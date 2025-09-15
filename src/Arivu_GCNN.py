 
# gcn_can.py (end-to-end: runs preprocessing if needed, then trains GCN)
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

# Ensure we can import local preprocess when running from project root
try:
    from preprocess import preprocess_all_splits
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from preprocess import preprocess_all_splits

ARTIFACTS = ["X.npy", "edge_index.npy", "node_ids.npy"]
if not all(Path(a).exists() for a in ARTIFACTS):
    print("Preprocessed artifacts not found. Running preprocessing...")
    preprocess_all_splits()

# Load preprocessed files
X = np.load("X.npy")              # [N, 9]
edge_index_np = np.load("edge_index.npy")  # [2, E]
node_ids = np.load("node_ids.npy", allow_pickle=True).tolist()

x = torch.tensor(X, dtype=torch.float)
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.lin = torch.nn.Linear(out_dim, 1)  # anomaly score/regression or binary
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        emb = x
        score = torch.sigmoid(self.lin(emb)).squeeze(-1)
        return emb, score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = Data(x=x, edge_index=edge_index).to(device)
model = GCNNet(in_dim=X.shape[1], hid_dim=64, out_dim=32).to(device)
decoder = torch.nn.Linear(32, X.shape[1]).to(device)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(decoder.parameters()),
    lr=1e-3,
    weight_decay=1e-5,
)

# If you have per-node labels (e.g., majority Attack/Normal per Arbitration_ID),
# prepare y: a tensor of 0/1 labels same order as node_ids. Otherwise, run unsupervised:
# For demo we'll assume no labels and train with link-prediction-ish or embedding reconstruction loss.
# Here we'll train to reconstruct neighborhood-averaged features (simple self-supervised).
def neighborhood_target_matrix(x, edge_index):
    # compute for each node the mean of neighbor features (simple target)
    N = x.size(0)
    deg = torch.zeros(N, device=x.device)
    agg = torch.zeros_like(x)
    src, dst = edge_index
    for s,d in zip(src, dst):
        agg[d] += x[s]
        deg[d] += 1
    deg = deg.unsqueeze(-1).clamp(min=1.0)
    return (agg / deg)

# Training loop
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    emb, score = model(data.x, data.edge_index)
    # self-supervised target
    target = neighborhood_target_matrix(data.x, data.edge_index)
    # reconstruct target from embeddings via a small decoder:
    recon = decoder(emb)
    loss = F.mse_loss(recon, target)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch} loss {loss.item():.6f}")

# After training: get node embeddings and anomaly score
model.eval()
with torch.no_grad():
    emb, score = model(data.x, data.edge_index)
    emb = emb.cpu().numpy()
    score = score.cpu().numpy()  # between 0 and 1

# Save embeddings and scores
np.save("node_embeddings.npy", emb)
np.save("node_anomaly_score.npy", score)
print("Saved node_embeddings.npy and node_anomaly_score.npy")
