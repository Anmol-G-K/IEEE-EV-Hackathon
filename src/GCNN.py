# gcn_pipeline_all_in_one.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import polars as pl
from pathlib import Path

# --- SETTINGS ---
MAX_BYTES = 8
WINDOW_SEC = 1.0
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Paths from data_viewer ---
DATA_ROOT = Path(__file__).resolve().parent / "data"
SPLIT_MAP = [
    ("train", DATA_ROOT / "0_Preliminary" / "0_Training"),
    ("test",  DATA_ROOT / "0_Preliminary" / "1_Submission"),
    ("val",   DATA_ROOT / "1_Final"),
]

def scan_split(split: str, base_dir: Path) -> pl.LazyFrame:
    if not base_dir.exists():
        return None
    files = sorted(base_dir.rglob("*.csv"))
    if not files:
        return None
    lfs = []
    for f in files:
        lf = pl.scan_csv(f.as_posix(), has_header=True, ignore_errors=True, infer_schema_length=5000)
        EXPECTED_COLS = ["Timestamp", "Arbitration_ID", "DLC", "Data", "Class", "SubClass"]
        for c in EXPECTED_COLS:
            if c not in lf.columns:
                lf = lf.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))
        lf = lf.with_columns(
            pl.lit(split).alias("split"),
            pl.lit(f.name).alias("source_file"),
            pl.lit(str(f.parent)).alias("source_dir"),
        )
        lfs.append(lf)
    return pl.concat(lfs, how="vertical_relaxed")

def normalize(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns([
        pl.col("Timestamp").cast(pl.Float64, strict=False),
        pl.col("Arbitration_ID").cast(pl.Utf8, strict=False),
        pl.col("DLC").cast(pl.Int64, strict=False),
        pl.col("Data").cast(pl.Utf8, strict=False),
        pl.col("Class").cast(pl.Utf8, strict=False),
        pl.col("SubClass").cast(pl.Utf8, strict=False),
    ])
    lf = lf.with_columns(
        pl.when(pl.col("Data").is_not_null())
        .then(pl.col("Data").str.replace_all(r"[^0-9A-Fa-f]", "").str.to_uppercase())
        .otherwise(None)
        .alias("Data")
    )
    lf = lf.with_columns(
        pl.when(pl.col("Data").is_not_null())
        .then((pl.col("Data").str.len_chars() // 2).cast(pl.Int64))
        .otherwise(None)
        .alias("data_len_bytes")
    )
    lf = (
        lf.group_by(["split", "source_dir", "source_file"])
        .agg([
            pl.col("Timestamp"),
            (pl.col("Timestamp") - pl.col("Timestamp").min()).alias("ts_rel_s"),
            pl.all().exclude("Timestamp")
        ])
        .explode(pl.all().exclude(["split", "source_dir", "source_file"]))
    )
    return lf

def _hex_to_padded_bytes_vector(hex_string: str, max_bytes: int = MAX_BYTES) -> np.ndarray:
    if hex_string is None:
        return np.zeros(max_bytes, dtype=np.float32)
    if len(hex_string) % 2 == 1:
        hex_string = '0' + hex_string
    try:
        byte_values = bytes.fromhex(hex_string)
    except Exception:
        return np.zeros(max_bytes, dtype=np.float32)
    byte_values = byte_values[:max_bytes] + bytes(max(0, max_bytes - len(byte_values)))
    return np.frombuffer(byte_values, dtype=np.uint8).astype(np.float32)

def _build_lazy_dataset() -> pl.LazyFrame:
    lfs = [scan_split(split, base) for split, base in SPLIT_MAP]
    lfs = [lf for lf in lfs if lf is not None]
    if not lfs:
        raise FileNotFoundError("No CSV files found.")
    return normalize(pl.concat(lfs, how="vertical_relaxed"))

def _build_node_features_from_lazy(lf: pl.LazyFrame):
    df = lf.select(["Arbitration_ID", "Data"]).collect(streaming=True)
    aid_to_vectors, counts = {}, {}
    for row in df.iter_rows():
        aid, data_hex = row
        vec = _hex_to_padded_bytes_vector(data_hex)
        if aid not in aid_to_vectors:
            aid_to_vectors[aid] = vec.astype(np.float64)
            counts[aid] = 1
        else:
            aid_to_vectors[aid] += vec
            counts[aid] += 1
    node_ids = list(aid_to_vectors.keys())
    sums = np.vstack([aid_to_vectors[aid] for aid in node_ids])
    freqs = np.array([counts[aid] for aid in node_ids], dtype=np.float64).reshape(-1, 1)
    payload_means = (sums / np.clip(freqs, 1.0, None)).astype(np.float32)

    scaler = StandardScaler()
    payload_means_scaled = scaler.fit_transform(payload_means)
    freq_log = np.log1p(freqs)
    freq_log = (freq_log - freq_log.mean()) / (freq_log.std() + 1e-9)
    X = np.hstack([payload_means_scaled, freq_log.astype(np.float32)])
    id_to_idx = {aid: i for i, aid in enumerate(node_ids)}
    return node_ids, X, id_to_idx

def _build_time_series_from_lazy(lf: pl.LazyFrame, node_ids, id_to_idx):
    tdf = (
        lf.select(["Arbitration_ID", pl.col("ts_rel_s").alias("t_rel")])
        .with_columns((pl.col("t_rel") // WINDOW_SEC).cast(pl.Int64).alias("win"))
        .group_by(["Arbitration_ID", "win"]).agg(pl.len().alias("cnt"))
        .collect(streaming=True)
    )
    n_wins = int(tdf["win"].max()) + 1
    M = np.zeros((len(node_ids), n_wins), dtype=np.float32)
    for aid, win, cnt in tdf.iter_rows():
        idx = id_to_idx.get(aid)
        if idx is not None and 0 <= win < n_wins:
            M[idx, int(win)] = float(cnt)
    M = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-9)
    return M

def _build_adj_from_timeseries(M, top_k=8, threshold=0.3):
    n = M.shape[0]
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    corr = np.corrcoef(M)
    corr = np.nan_to_num(corr)
    edges = []
    for i in range(n):
        scores = corr[i].copy()
        scores[i] = -1e9
        neighbors = np.argsort(scores)[-top_k:]
        for j in neighbors:
            if scores[j] >= threshold:
                edges.append((i, j))
                edges.append((j, i))
    if not edges:
        edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    return np.asarray(edges, dtype=np.int64).T

def preprocess_all():
    lf = _build_lazy_dataset()
    node_ids, X, id_to_idx = _build_node_features_from_lazy(lf)
    M = _build_time_series_from_lazy(lf, node_ids, id_to_idx)
    edge_index = _build_adj_from_timeseries(M)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "edge_index.npy", edge_index)
    np.save(OUTPUT_DIR / "node_ids.npy", np.array(node_ids, dtype=object))
    return X, edge_index

# --- GCN ---
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.lin_out = nn.Linear(out_dim, 1)

    def forward(self, x, A):
        h = A @ x
        h = self.W1(h)
        h = F.relu(h)
        h = A @ h
        h = self.W2(h)
        emb = h
        score = torch.sigmoid(self.lin_out(emb)).squeeze(-1)
        return emb, score

def train_gcn(X, edge_index):
    N = X.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    src, dst = edge_index
    for s, d in zip(src, dst):
        A[s, d] = 1.0
    A = A + np.eye(N, dtype=np.float32)
    D = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-9))
    A_norm = D @ A @ D

    x = torch.tensor(X, dtype=torch.float32)
    A_norm = torch.tensor(A_norm, dtype=torch.float32)

    model = SimpleGCN(in_dim=X.shape[1]).to("cpu")
    decoder = nn.Linear(32, x.shape[1]).to("cpu")
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)

    target = A_norm @ x
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        emb, score = model(x, A_norm)
        recon = decoder(emb)
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        emb, score = model(x, A_norm)
    np.save(OUTPUT_DIR / "node_embeddings_cpu.npy", emb.numpy())
    np.save(OUTPUT_DIR / "node_anomaly_score_cpu.npy", score.numpy())
    return emb.numpy(), score.numpy(), edge_index

# --- Visualization ---
def visualize_all(emb, score, edge_index):
    # PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=score, cmap='viridis', s=40)
    plt.colorbar(sc, label="Anomaly Score")
    plt.title("Node Embeddings (PCA) Colored by Anomaly Score")
    plt.savefig(OUTPUT_DIR / "embedding_pca.png")
    plt.close()

    # Histogram
    plt.figure()
    sns.histplot(score, bins=30, kde=True)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.savefig(OUTPUT_DIR / "score_histogram.png")
    plt.close()

    # Graph
    G = nx.DiGraph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=score, cmap="viridis")
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label="Anomaly Score")
    plt.title("Graph Structure Colored by Anomaly Score")
    plt.axis('off')
    plt.savefig(OUTPUT_DIR / "graph_structure.png")
    plt.close()

# --- Main ---
if __name__ == "__main__":
    print("Running full pipeline...")
    X, edge_index = preprocess_all()
    emb, score, edge_index = train_gcn(X, edge_index)
    visualize_all(emb, score, edge_index)
    print(f"✅ Done. Files and plots saved in {OUTPUT_DIR.resolve()}")
