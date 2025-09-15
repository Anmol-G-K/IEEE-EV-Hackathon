"""
Car Hacking IDS – Hybrid ML Framework
-------------------------------------
- Loads Car Hacking: Attack & Defense Challenge 2020 dataset
- Preprocesses CSV → NPZ windows
- Defines Hybrid model (Sequence + Graph + Contrastive + Fusion)
- Trains with PyTorch AMP for efficiency

Dependencies:
    pip install polars pandas numpy torch torch-geometric scikit-learn tqdm lightgbm

"""

import os
import math
import json
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeoData, Batch as GeoBatch
from torch_geometric.nn import GCNConv, global_mean_pool

# -------------------------------
# Paths
# -------------------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CACHE_DIR = ROOT / "cache_npz"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIRS = {
    "train": DATA_ROOT / "0_Preliminary" / "0_Training",
    "test":  DATA_ROOT / "0_Preliminary" / "1_Submission",
    "val":   DATA_ROOT / "1_Final",
}

EXPECTED_COLS = ["Timestamp", "Arbitration_ID", "DLC", "Data", "Class", "SubClass"]

# -------------------------------
# Utils: payload parsing
# -------------------------------
def parse_hex_payload(hex_str: str, max_len=8):
    if hex_str is None or (isinstance(hex_str, float) and math.isnan(hex_str)):
        return [0]*max_len
    s = str(hex_str).strip().upper()
    s = "".join([c for c in s if c in "0123456789ABCDEF"])
    if len(s) % 2 != 0:
        s = "0" + s
    arr = [int(s[i:i+2], 16) for i in range(0, min(len(s), max_len*2), 2)]
    arr += [0]*(max_len - len(arr))
    return arr[:max_len]

# -------------------------------
# Preprocessing: load + window
# -------------------------------
def scan_split(split: str, base_dir: Path) -> pl.DataFrame:
    files = sorted(base_dir.rglob("*.csv"))
    if not files:
        return None
    lfs = []
    for f in files:
        lf = pl.read_csv(f.as_posix(), has_header=True)
        for c in EXPECTED_COLS:
            if c not in lf.columns:
                lf = lf.with_columns(pl.lit(None).alias(c))
        lf = lf.with_columns(pl.lit(split).alias("split"))
        lfs.append(lf)
    return pl.concat(lfs)

def preprocess_split(split: str, window_size=128, step=64, max_rows=None):
    out_path = CACHE_DIR / f"{split}_w{window_size}_s{step}.npz"
    if out_path.exists():
        print(f"[Cache] {out_path}")
        return np.load(out_path, allow_pickle=True)

    df = scan_split(split, SPLIT_DIRS[split])
    if max_rows:
        df = df.head(max_rows)

    # Clean
    df = df.with_columns([
        pl.col("Timestamp").cast(pl.Float64, strict=False),
        pl.col("Arbitration_ID").cast(pl.Utf8, strict=False),
        pl.col("Data").cast(pl.Utf8, strict=False),
    ])
    df = df.with_columns(
        pl.col("Data").apply(lambda x: "".join([c for c in str(x).upper() if c in "0123456789ABCDEF"]) if x else None)
    )
    df = df.drop_nulls(subset=["Timestamp", "Arbitration_ID", "Data"])

    # Convert to python objects
    rows = df.to_dicts()

    X_arb, X_payload, X_iat, Y = [], [], [], []
    n = len(rows)
    idx = 0
    while idx + window_size <= n:
        chunk = rows[idx: idx+window_size]
        arbs = [r["Arbitration_ID"] for r in chunk]
        payloads = [parse_hex_payload(r["Data"]) for r in chunk]
        ts = [r["Timestamp"] for r in chunk]
        iat = np.diff(ts, prepend=ts[0]).tolist()
        lbls = [r["Class"] for r in chunk]
        label = "Attack" if "Attack" in lbls else "Normal"

        X_arb.append(arbs)
        X_payload.append(payloads)
        X_iat.append(iat)
        Y.append(label)
        idx += step

    np.savez_compressed(out_path, arb=X_arb, payload=X_payload, iat=X_iat, label=Y)
    print(f"[Saved] {out_path}")
    return np.load(out_path, allow_pickle=True)

# -------------------------------
# Dataset + Dataloader
# -------------------------------
class WindowDataset(Dataset):
    def __init__(self, data_npz, arb_encoder, label_map):
        self.X_arb = data_npz["arb"]
        self.X_payload = data_npz["payload"]
        self.X_iat = data_npz["iat"]
        self.Y = data_npz["label"]
        self.arb_encoder = arb_encoder
        self.label_map = label_map

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        arbs = self.arb_encoder.transform(self.X_arb[idx])
        payload = np.array(self.X_payload[idx], dtype=np.int64)
        iat = np.array(self.X_iat[idx], dtype=np.float32)
        y = self.label_map[self.Y[idx]]
        return torch.tensor(arbs), torch.tensor(payload), torch.tensor(iat), torch.tensor(y)

# -------------------------------
# Models
# -------------------------------
class SeqTransformerEncoder(nn.Module):
    def __init__(self, n_arbs, arb_dim=64, byte_dim=32, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.arb_emb = nn.Embedding(n_arbs, arb_dim)
        self.byte_emb = nn.Embedding(256, byte_dim)
        self.byte_proj = nn.Linear(byte_dim*8, 64)
        self.in_proj = nn.Linear(arb_dim+64+1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 256, batch_first=True)
        self.trans = nn.TransformerEncoder(layer, n_layers)
        self.out_dim = d_model

    def forward(self, arb, payload, iat):
        a = self.arb_emb(arb)
        b = self.byte_emb(payload).view(payload.size(0), payload.size(1), -1)
        b = self.byte_proj(b)
        i = iat.unsqueeze(-1)
        x = self.in_proj(torch.cat([a, b, i], -1))
        x = self.trans(x)
        return x.mean(1)

class WindowGNN(nn.Module):
    def __init__(self, in_dim=1, hid=64):
        super().__init__()
        self.fc = nn.Linear(in_dim, hid)
        self.conv1 = GCNConv(hid, hid)
        self.conv2 = GCNConv(hid, hid)
        self.out_dim = hid

    def forward(self, geo_batch):
        x = torch.relu(self.fc(geo_batch.x))
        x = torch.relu(self.conv1(x, geo_batch.edge_index))
        x = torch.relu(self.conv2(x, geo_batch.edge_index))
        return global_mean_pool(x, geo_batch.batch)

class ContrastiveEncoder(nn.Module):
    def __init__(self, n_arbs, dim=64):
        super().__init__()
        self.arb_emb = nn.Embedding(n_arbs, 64)
        self.byte_emb = nn.Embedding(256, 32)
        self.fc = nn.Sequential(
            nn.Linear(64+32*8, 128), nn.ReLU(), nn.Linear(128, dim)
        )
        self.out_dim = dim

    def forward(self, arb, payload):
        a = self.arb_emb(arb)
        b = self.byte_emb(payload).view(payload.size(0), -1)
        x = torch.cat([a, b], -1)
        return nn.functional.normalize(self.fc(x), dim=-1)

class FusionClassifier(nn.Module):
    def __init__(self, seq_dim, gnn_dim, contr_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_dim+gnn_dim+contr_dim, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, s, g, c):
        return self.fc(torch.cat([s,g,c],-1))

# -------------------------------
# Training loop
# -------------------------------
def train_epoch(seq, contr, fusion, loader, opt, device):
    seq.train(); contr.train(); fusion.train()
    crit = nn.CrossEntropyLoss()
    total, correct = 0, 0
    for arb,payload,iat,y in tqdm(loader):
        arb,payload,iat,y = arb.to(device),payload.to(device),iat.to(device),y.to(device)
        with torch.cuda.amp.autocast():
            s = seq(arb,payload,iat)
            # simple contrastive rep = first msg in window
            c = contr(arb[:,0], payload[:,0])
            g = torch.zeros((arb.size(0), seq.out_dim), device=device) # stub
            out = fusion(s,g,c)
            loss = crit(out,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += y.size(0)
        correct += (out.argmax(1)==y).sum().item()
    return correct/total

def eval_epoch(seq, contr, fusion, loader, device):
    seq.eval(); contr.eval(); fusion.eval()
    y_true,y_pred=[],[]
    for arb,payload,iat,y in loader:
        arb,payload,iat = arb.to(device),payload.to(device),iat.to(device)
        with torch.no_grad():
            s = seq(arb,payload,iat)
            c = contr(arb[:,0], payload[:,0])
            g = torch.zeros((arb.size(0), seq.out_dim), device=device)
            out = fusion(s,g,c)
        y_true += y.tolist()
        y_pred += out.argmax(1).cpu().tolist()
    print(classification_report(y_true,y_pred))

# -------------------------------
# Main
# -------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocess
    train_npz = preprocess_split("train", max_rows=50000)
    val_npz = preprocess_split("val", max_rows=20000)

    # Encode Arbitration_ID
    all_arbs = sum(train_npz["arb"].tolist(), [])
    arb_enc = LabelEncoder().fit(all_arbs)
    labels = {"Normal":0, "Attack":1}

    train_ds = WindowDataset(train_npz, arb_enc, labels)
    val_ds = WindowDataset(val_npz, arb_enc, labels)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    # Models
    seq = SeqTransformerEncoder(len(arb_enc.classes_)).to(device)
    contr = ContrastiveEncoder(len(arb_enc.classes_)).to(device)
    fusion = FusionClassifier(seq.out_dim, seq.out_dim, contr.out_dim, 2).to(device)

    opt = optim.Adam(list(seq.parameters())+list(contr.parameters())+list(fusion.parameters()), lr=1e-3)

    for epoch in range(3):
        acc = train_epoch(seq, contr, fusion, train_dl, opt, device)
        print(f"Epoch {epoch} Train Acc {acc:.3f}")
        eval_epoch(seq, contr, fusion, val_dl, device)

if __name__=="__main__":
    main()
