import os
import copy
import json
import wandb
import pickle
import argparse
from pathlib import Path
import torch
import re
import torch.nn as nn
from torch_geometric.utils import to_undirected, to_networkx, k_hop_subgraph, is_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.seed import seed_everything

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.utils import *
from framework.data_loader import split_forget_retain, train_test_split_edges_no_neg_adj_mask, get_original_data
# from train_mi import load_mi_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _read_parsed_targets1(parsed_path, request_id, want):
    """Return (nodes, pairs) from parsed_*.jsonl filtered by request_id and mode."""
    nodes, pairs = set(), set()
    with open(parsed_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if request_id and ex.get("request_id") != request_id:
                continue
            tlist = ex.get("parsed", {}).get("targets") or ex.get("resolved_targets", [])
            for t in tlist:
                if want == "node" and t.get("type") == "node":
                    nodes.add(int(t["node_id"]))
                elif want == "edge" and t.get("type") == "edge":
                    u, v = int(t["src"]), int(t["dst"])
                    pairs.add((u, v))
    return nodes, pairs
def _read_parsed_targets(parsed_path, request_id, want):
    """
    Return (nodes, pairs) from parsed_*.jsonl filtered by request_id and mode.
    - nodes: set of node identifiers
    - pairs: set of (src, dst) pairs
    NOTE: 'attribute' and 'label' are treated as node-scoped deletions.
    """
    nodes, pairs = set(), set()
    with open(parsed_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if request_id and ex.get("request_id") != request_id:
                continue
            tlist = ex.get("parsed", {}).get("targets") or ex.get("resolved_targets", [])
            for t in tlist:
                ttype = t.get("type")
                if want == "node" and ttype in ("node", "attribute", "label"):
                    nodes.add(int(t["node_id"]))
                elif want == "edge" and ttype == "edge":
                    u, v = int(t["src"]), int(t["dst"])
                    pairs.add((u, v))
    return nodes, pairs

def _build_edge_map(edge_index, undirected=True):
    src, dst = edge_index
    mapping = {}
    for i in range(edge_index.size(1)):
        u = int(src[i]); v = int(dst[i])
        key = (min(u,v), max(u,v)) if undirected else (u, v)
        mapping.setdefault(key, []).append(i)
    return mapping

def _read_json_or_jsonl(path, max_items=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
        if data.startswith("[") or data.startswith("{"):
            try:
                obj = json.loads(data)
                if isinstance(obj, list):
                    items = obj
                else:
                    items = [obj]
                if max_items is not None:
                    items = items[:max_items]
                return items
            except json.JSONDecodeError:
                pass
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError:
                continue
            if max_items is not None and len(items) >= max_items:
                break
    return items

def _load_yelp_id_mapper():
    """
    Build external->int node id map for Yelp from files in YELP_DIR.
    Returns a dict ext_id -> int_id.
    """
    yelp_dir = os.environ.get("YELP_DIR", os.path.join("data", "yelp"))

    def _find_one(cands):
        for name in cands:
            p = os.path.join(yelp_dir, name)
            if os.path.exists(p):
                return p
        return None

    user_path = _find_one(("users.json","user.json","yelp_academic_dataset_user.json"))
    biz_path  = _find_one(("businesses.json","business.json","yelp_academic_dataset_business.json"))
    if not (user_path and biz_path):
        raise FileNotFoundError(
            f"Could not find Yelp user/business files in {yelp_dir}. "
            "Set YELP_DIR or place users.json/businesses.json (or yelp_academic_dataset_*) there."
        )

    # Optional caps for speed (same env vars we used in the loader)
    def _cap(env):
        v = os.getenv(env)
        return int(v) if v and v.isdigit() else None

    users = _read_json_or_jsonl(user_path, max_items=_cap("YELP_MAX_USERS"))
    biz   = _read_json_or_jsonl(biz_path,  max_items=_cap("YELP_MAX_BIZ"))

    def get_u_id(u):  return u.get("user_id") or u.get("userId") or u.get("userID")
    def get_b_id(b):  return b.get("business_id") or b.get("businessId") or b.get("businessID")

    ext2int = {}
    # users first: 0..U-1
    for u in users:
        uid = get_u_id(u)
        if uid is not None and uid not in ext2int:
            ext2int[uid] = len(ext2int)
    # businesses next: offset..offset+B-1
    offset = len(ext2int)
    for b in biz:
        bid = get_b_id(b)
        if bid is not None and bid not in ext2int:
            ext2int[bid] = offset + (len(ext2int) - offset)
    return ext2int


def get_processed_data(d, val_ratio, test_ratio, df_ratio, subset='in'):
    '''pend for future use'''
    data = get_original_data(d)

    data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
    data = split_forget_retain(data, df_ratio, subset)
    return data


torch.autograd.set_detect_anomaly(True)
def main():
    import sys

    # -------------------------
    # (A) Extract extra flags BEFORE framework.parse_args()
    # -------------------------
    del_mode = "node"
    if "--del_mode" in sys.argv:
        i = sys.argv.index("--del_mode"); del_mode = sys.argv[i+1]; del sys.argv[i:i+2]

    parsed_file = None
    if "--parsed_file" in sys.argv:
        i = sys.argv.index("--parsed_file"); parsed_file = sys.argv[i+1]; del sys.argv[i:i+2]

    req_id = None
    if "--request_id" in sys.argv:
        i = sys.argv.index("--request_id"); req_id = sys.argv[i+1]; del sys.argv[i:i+2]

    # -------------------------
    # (B) Framework args as-is
    # -------------------------
    # ---- BEGIN viz flags ----
    viz = False
    viz_out = None
    if "--viz" in sys.argv:
        i = sys.argv.index("--viz"); viz = True; del sys.argv[i:i+1]
    if "--viz_out" in sys.argv:
        i = sys.argv.index("--viz_out"); viz_out = sys.argv[i+1]; del sys.argv[i:i+2]
    # ---- END viz flags ----
    args = parse_args()
    args.del_mode = del_mode

    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_all', str(args.random_seed))
    args.attack_dir = attack_path_all; os.makedirs(attack_path_all, exist_ok=True)
    shadow_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'shadow_all', str(args.random_seed))
    args.shadow_dir = shadow_path_all; os.makedirs(shadow_path_all, exist_ok=True)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
                                       '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # -------------------------
    # (C) Defer reading forget_file until after data is loaded,
    #     because we may need the edge_index to map (u,v) pairs.
    # -------------------------
    # Build dataset (df_ratio=0 if we supply IDs)
    data = get_processed_data(
        args.dataset, val_ratio=0.05, test_ratio=0.05,
        df_ratio=0.0 if (parsed_file or args.forget_file) else args.df_size
    )

    forget_ids = None              # final list[int] of node IDs OR edge indices
    pending_pairs = None           # if we get (u,v) pairs, map to indices below
    auto = (del_mode.lower() == "auto")
    # Case 1: parsed_file provided → read targets directly
    if parsed_file:
        # Peek both scopes to decide the actual mode if 'auto' or if the chosen mode is empty
        n_nodes, _  = _read_parsed_targets(parsed_file, req_id, "node")
        _, e_pairs  = _read_parsed_targets(parsed_file, req_id, "edge")
        print(f"[parsed] request {req_id}: node_targets={len(n_nodes)}, edge_targets={len(e_pairs)}")
        if auto or (del_mode == "node" and len(n_nodes) == 0 and len(e_pairs) > 0):
            del_mode = "edge"
            args.del_mode = "edge"
            print("[auto] switched del_mode → edge (no node targets; edges present)")
        elif auto or (del_mode == "edge" and len(e_pairs) == 0 and len(n_nodes) > 0):
            del_mode = "node"
            args.del_mode = "node"
            print("[auto] switched del_mode → node (no edge targets; nodes present)")

        want = "node" if args.del_mode == "node" else "edge"
        nodes, pairs = _read_parsed_targets(parsed_file, req_id, want)
        # --- NEW: Yelp mapping for external string IDs ---
        if args.dataset.lower() == "yelp":
            ext2int = _load_yelp_id_mapper()
            if want == "node":
                mapped = set()
                for nid in nodes:
                    # allow both int and str
                    if isinstance(nid, int) or (isinstance(nid, str) and nid.isdigit()):
                        mapped.add(int(nid))
                    else:
                        if nid in ext2int:
                            mapped.add(ext2int[nid])
                        else:
                            # silently skip unknown ids (or you can raise)
                            continue
                nodes = mapped
            else:  # edge mode
                mapped_pairs = set()
                for (u, v) in pairs:
                    def _to_int(x):
                        if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
                            return int(x)
                        return ext2int.get(x, None)
                    ui = _to_int(u); vi = _to_int(v)
                    if ui is not None and vi is not None:
                        mapped_pairs.add((ui, vi))
                pairs = mapped_pairs
        # --- end Yelp mapping ---
        if want == "node":
            forget_ids = sorted(nodes)
        else:
            pending_pairs = pairs

    # Case 2: legacy forget_file path
    elif args.forget_file and Path(args.forget_file).exists():
        raw = Path(args.forget_file).read_text().strip()
        if raw == "":
            print("Nothing to delete – request already satisfied.\n")
            return
        # Detect pairs vs single-column IDs
        if "\t" in raw or "," in raw:
            # format: "u\tv" per line or "u,v"
            pairs = set()
            for line in raw.splitlines():
                if not line.strip(): continue
                toks = re.split(r"[\t, ]+", line.strip())
                if len(toks) >= 2:
                    pairs.add((int(toks[0]), int(toks[1])))
            pending_pairs = pairs
        else:
            forget_ids = sorted({int(x) for x in raw.split()})

    # If we have pairs → map to edge indices (handles undirected graphs)
    if pending_pairs and args.del_mode == "edge":
        from torch_geometric.utils import is_undirected
        # Full graph
        und_full = is_undirected(data.edge_index)
        emap_full = _build_edge_map(data.edge_index, undirected=und_full)
        idxs_full = set()
        for (u,v) in pending_pairs:
            key = (min(u,v), max(u,v)) if und_full else (u,v)
            idxs_full.update(emap_full.get(key, []))

        # Train graph
        idxs_train = set()
        if hasattr(data, "train_pos_edge_index"):
            und_train = is_undirected(data.train_pos_edge_index)
            emap_train = _build_edge_map(data.train_pos_edge_index, undirected=und_train)
            for (u,v) in pending_pairs:
                key = (min(u,v), max(u,v)) if und_train else (u,v)
                idxs_train.update(emap_train.get(key, []))

        forget_ids = sorted(idxs_full)  # df_mask uses full edge_index by default
        print(f"[map] matched edges: full={len(idxs_full)}, train={len(idxs_train)}")

    # Early exit if nothing to delete
    # Attach deletion masks
    if forget_ids is not None:
        if args.del_mode == 'node':
            print(f"[delete] {len(forget_ids)} node IDs to delete")
            data.sdf_nodes = torch.tensor(forget_ids, dtype=torch.long)

            # node mask
            data.sdf_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            if len(forget_ids) > 0:
                data.sdf_mask[data.sdf_nodes] = True

            # retain masks (some trainers use these)
            data.retain_mask  = ~data.sdf_mask
            data.retain_nodes = torch.nonzero(data.retain_mask, as_tuple=False).view(-1)

            # derive an edge-level view (incident edges) for evaluators/UtU
            src, dst = data.edge_index
            df_edge_mask = data.sdf_mask[src] | data.sdf_mask[dst]
            data.df_mask = df_edge_mask
            data.directed_df_edge_index = data.edge_index[:, df_edge_mask]

            # Only UtU expects sdf_mask to mean "edge deletions"
            if args.unlearning_model.lower().startswith("utu"):
                data.sdf_mask = df_edge_mask

        else:  # EDGE deletion
            print(f"[delete] {len(forget_ids)} edge indices to delete")
            df_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
            if len(forget_ids) > 0:
                df_mask[torch.tensor(forget_ids, dtype=torch.long)] = True

            data.df_mask = df_mask
            data.directed_df_edge_index = data.edge_index[:, df_mask]
            data.dr_mask = ~data.df_mask

            # keep node masks empty for edge-only deletion
            data.sdf_nodes = torch.tensor([], dtype=torch.long)
            data.sdf_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.retain_mask  = torch.ones(data.num_nodes, dtype=torch.bool)
            data.retain_nodes = data.retain_mask.nonzero(as_tuple=False).view(-1)

            # UtU reads edge mask from sdf_mask as well
            if args.unlearning_model.lower().startswith("utu"):
                data.sdf_mask = df_mask

        del_nodes = int(getattr(data, "sdf_mask", torch.zeros(data.num_nodes, dtype=torch.bool)).sum())
        del_edges = int(getattr(data, "df_mask",  torch.zeros(data.edge_index.size(1), dtype=torch.bool)).sum())
        train_e   = int(data.train_pos_edge_index.size(1)) if hasattr(data, "train_pos_edge_index") else int(data.edge_index.size(1))
        tot_e     = int(data.edge_index.size(1))
        print(f"[summary] deleted_nodes={del_nodes} "
            f"deleted_edges={del_edges} "
            f"train_edges={train_e} "
            f"total_edges={tot_e}")

    # -------------------------
    # (D) Continue with your existing logic (unchanged)
    #     - sets data.sdf_mask / df_mask, trains, evaluates, ...
    # -------------------------
    print('Directed dataset:', data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]
    print('Training args', args)

    model = get_model(args, data.sdf_node_1hop_mask, data.sdf_node_2hop_mask,
                      num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
            logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))   # logits_ori: tensor.shape([num_nodes, num_nodes]), represent probability of edge existence between any two nodes
            if logits_ori is not None:
                logits_ori = logits_ori.to(device)
        else:
            logits_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        model.load_state_dict(model_ckpt['model_state'], strict=False)
   
    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)
    # data = data.to(device)

    # Optimizer
    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(model.deletion1.parameters(), lr=args.lr)
            optimizer2 = torch.optim.Adam(model.deletion2.parameters(), lr=args.lr)
            optimizer = [optimizer1, optimizer2]
        else:
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters()])
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    
    wandb.init(config=args, project="GNNDelete", group="over_unlearn", name=get_run_name(args), mode=args.mode)
    wandb.watch(model, log_freq=100)


    # MI attack model
    attack_model_all = None
    attack_model_sub = None

    # Train
    trainer = get_trainer(args)
    trainer.train(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    # Test
    if args.unlearning_model != 'retrain':
        retrain_path = os.path.join(
            'checkpoint', args.dataset, args.gnn, 'retrain', 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]), 
            'model_best.pt')
        if os.path.exists(retrain_path):
            retrain_ckpt = torch.load(retrain_path, map_location=device)
            retrain_args = copy.deepcopy(args)
            retrain_args.unlearning_model = 'retrain'
            retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
            retrain.load_state_dict(retrain_ckpt['model_state'])
            retrain = retrain.to(device)
            retrain.eval()
        else:
            retrain = None
    else:
        retrain = None
    
    test_results = trainer.test(model, data, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    print(test_results[-1])

    trainer.save_log()
    wandb.finish()
    # ---- BEGIN: optional PCA viz (after deletion)
    if viz:
        try:
            from viz_projection import plot_after_with_pca
            # forget_ids may be None if early-exited; only plot when we actually had targets
            if 'forget_ids' in locals() and forget_ids is not None:
                out_dir = viz_out or os.path.join(args.checkpoint_dir, "viz")
                plot_after_with_pca(
                    args=args,
                    data=data,
                    forget_ids=forget_ids,
                    del_mode=args.del_mode,
                    original_ckpt_dir=original_path,   # points to .../<dataset>/<gnn>/original/<seed>
                    request_id=req_id,
                    out_dir=out_dir
                )
            else:
                print("[viz] skip: no targets to delete")
        except Exception as e:
            print(f"[viz] WARNING: could not render after-deletion figure: {e}")
    # ---- END
  


if __name__ == "__main__":
    main()
