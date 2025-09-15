
# preprocess_can.py (refactored to use helpers.data_viewer scanning/normalization)
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# SETTINGS
MAX_BYTES = 8
WINDOW_SEC = 1.0

# Import robust scanning/normalization from helpers
from helpers.data_viewer import scan_split, normalize, SPLIT_MAP

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
        raise FileNotFoundError("No CSV files found in configured dataset directories.")
    lf = pl.concat(lfs, how="vertical_relaxed")
    lf = normalize(lf)
    return lf

def _build_node_features_from_lazy(lf: pl.LazyFrame):
    # Compute mean payload bytes per Arbitration_ID and log-frequency
    # Convert hex Data to byte vectors using Polars list and then finalize to numpy
    lf_bytes = lf.with_columns(
        pl.when(pl.col("Data").is_not_null())
          .then(pl.col("Data")).otherwise(None)
          .alias("Data_clean")
    )

    # Aggregate mean of byte positions 0..7 via expression
    # Build columns b0..b7 on the fly using a small UDF after materialization for safety
    df = lf_bytes.select(["Arbitration_ID", "Data_clean"]).collect(streaming=True)

    # Group by Arbitration_ID on pandas-like side to create robust byte means
    # Convert to python for byte parsing (fast enough since aggregated at group level later)
    aid_to_vectors = {}
    counts = {}
    for row in df.iter_rows():
        aid, data_hex = row
        vec = _hex_to_padded_bytes_vector(data_hex, MAX_BYTES)
        if aid not in aid_to_vectors:
            aid_to_vectors[aid] = vec.astype(np.float64)
            counts[aid] = 1
        else:
            aid_to_vectors[aid] += vec
            counts[aid] += 1

    node_ids = list(aid_to_vectors.keys())
    sums = np.vstack([aid_to_vectors[aid] for aid in node_ids]).astype(np.float64)
    freqs = np.array([counts[aid] for aid in node_ids], dtype=np.float64).reshape(-1, 1)
    payload_means = (sums / np.clip(freqs, 1.0, None)).astype(np.float32)

    scaler = StandardScaler()
    payload_means_scaled = scaler.fit_transform(payload_means)
    freq_log = np.log1p(freqs).astype(np.float32)
    freq_log = (freq_log - freq_log.mean()) / (freq_log.std() + 1e-9)
    X = np.hstack([payload_means_scaled.astype(np.float32), freq_log.astype(np.float32)])

    id_to_idx = {aid: i for i, aid in enumerate(node_ids)}
    return node_ids, X, id_to_idx

def _build_time_series_from_lazy(lf: pl.LazyFrame, node_ids, id_to_idx, window_sec: float = WINDOW_SEC):
    tdf = (
        lf.select([
            pl.col("Arbitration_ID"),
            pl.col("ts_rel_s").alias("t_rel"),
        ])
        .with_columns((pl.col("t_rel") // window_sec).cast(pl.Int64).alias("win"))
        .group_by(["Arbitration_ID", "win"]).agg(pl.len().alias("cnt"))
        .collect(streaming=True)
    )

    if tdf.height == 0:
        return np.zeros((len(node_ids), 1), dtype=np.float32)

    n_wins = int(tdf["win"].max()) + 1
    M = np.zeros((len(node_ids), n_wins), dtype=np.float32)
    for aid, win, cnt in tdf.iter_rows():
        idx = id_to_idx.get(aid)
        if idx is None:
            continue
        if 0 <= win < n_wins:
            M[idx, int(win)] = float(cnt)

    row_mean = M.mean(axis=1, keepdims=True)
    row_std = M.std(axis=1, keepdims=True) + 1e-9
    M = (M - row_mean) / row_std
    return M

def _build_adj_from_timeseries(M: np.ndarray, top_k: int = 8, threshold: float = 0.3) -> np.ndarray:
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
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append((i, j))
    return np.asarray(edges, dtype=np.int64).T

def preprocess_all_splits(output_dir: str | Path | None = None, top_k: int = 8, threshold: float = 0.3):
    lf = _build_lazy_dataset()
    node_ids, X, id_to_idx = _build_node_features_from_lazy(lf)
    M = _build_time_series_from_lazy(lf, node_ids, id_to_idx, window_sec=WINDOW_SEC)
    edge_index = _build_adj_from_timeseries(M, top_k=top_k, threshold=threshold)

    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    (out_dir / "node_ids.npy").write_bytes(np.array(node_ids, dtype=object).tobytes())
    # Use np.save for arrays
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "edge_index.npy", edge_index)
    np.save(out_dir / "node_ids.npy", np.array(node_ids, dtype=object))
    return node_ids, X, edge_index

if __name__ == "__main__":
    preprocess_all_splits()
    print("Saved X.npy, edge_index.npy, node_ids.npy")
