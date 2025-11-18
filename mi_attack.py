import argparse, torch, torch.nn as nn, numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------- tiny MLP attacker --------------------------------------------
class Attacker(nn.Module):
    def __init__(self, in_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.mlp(x).squeeze()

# ---------- args ----------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--dataset", default="Cora")
p.add_argument("--target_ckpt", required=True)
p.add_argument("--shadow_frac", type=float, default=0.5)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- load graph + target model ------------------------------------
data = Planetoid(root="data", name=args.dataset)[0]
from framework import get_model
ckpt = torch.load(args.target_ckpt, map_location=device)
saved_in_dim = ckpt["model_state"]["conv1.lin.weight"].shape[1]

if data.x.size(1) < saved_in_dim:
    pad = saved_in_dim - data.x.size(1)
    data.x = torch.cat([data.x, torch.zeros(data.num_nodes, pad)], dim=1)

gargs = argparse.Namespace(
        gnn="gcn",
        in_dim=saved_in_dim,
        hidden_dim=128,
        out_dim=64,
        num_edge_type=None,
        unlearning_model="baseline"
)

dummy = torch.zeros(data.num_nodes, dtype=torch.bool)
target = get_model(gargs, dummy, dummy,
                   num_nodes=data.num_nodes,
                   num_edge_type=None).to(device)
target.load_state_dict(ckpt["model_state"], strict=False)
target.eval()

# ---------- shadow / member labels ---------------------------------------
idx = np.arange(data.num_nodes)
shadow_idx, non_shadow_idx = train_test_split(idx,
                        test_size=1-args.shadow_frac, random_state=42)
member_mask       = torch.zeros(data.num_nodes, dtype=torch.bool)
member_mask[shadow_idx] = True           # “members” for attacker training

with torch.no_grad():
    logits = target(data.x.to(device), data.edge_index.to(device))  # [N, nclass]
    conf   = logits.softmax(dim=-1).max(dim=-1).values.cpu()        # [N]

X = conf.unsqueeze(-1)                    # feature = max-prob
y = member_mask.float()

# ---------- train attacker on 80 % of nodes ------------------------------
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0)
attacker = Attacker(1).to(device)
opt = torch.optim.Adam(attacker.parameters(), lr=1e-3)

for _ in range(1000):
    opt.zero_grad()
    loss = nn.BCEWithLogitsLoss()(attacker(X[train_idx].to(device)),
                                  y[train_idx].to(device))
    loss.backward(); opt.step()

# ---------- evaluate ------------------------------------------------------
attacker.eval()
with torch.no_grad():
    scores = attacker(X[test_idx].to(device)).sigmoid().cpu().numpy()
auc = roc_auc_score(y[test_idx].numpy(), scores)
print(f"MI-AUC on {args.dataset} = {auc:.4f}")
