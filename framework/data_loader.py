# framework/data_loader.py
from ogb.linkproppred import PygLinkPropPredDataset  # must put this in 1st line to avoid stall on running
import os, json, torch, pathlib
from torch_geometric.data import Data
import math
import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import k_hop_subgraph, is_undirected, to_undirected, negative_sampling, subgraph
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Planetoid, Reddit2, Flickr
import io

# ------------------------ small Yelp helpers ------------------------
def _read_json_or_jsonl(path, max_items=None):
    items = []
    if not os.path.exists(path):
        return items
    # try JSON array/object first
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().lstrip()
            if s.startswith("["):
                obj = json.loads(s)
                if isinstance(obj, list):
                    return obj[: (max_items or len(obj))]
            elif s.startswith("{"):
                obj = json.loads(s)
                return [obj]
    except Exception:
        pass
    # fallback JSONL
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_items is not None and len(items) >= max_items:
                break
    return items

def _save_yelp_index(user2idx: dict, biz2idx: dict, yelp_dir: str):
    """
    Persist a compact, reproducible mapping so deletion can map external IDs
    to the exact internal integer ids used by the trained graph.

    user2idx: user_id -> internal id in [0 .. U-1]
    biz2idx : business_id -> internal id in [U .. U+B-1]  (NOTE: absolute ids)
    """
    try:
        U = (max(user2idx.values()) + 1) if user2idx else 0
        B = 0
        if biz2idx:
            max_b_abs = max(biz2idx.values())  # == U + (B-1)
            if max_b_abs + 1 >= U:
                B = (max_b_abs + 1) - U

        users = [None] * U
        for uid, i in user2idx.items():
            if 0 <= i < U:
                users[i] = uid

        biz = [None] * B
        for bid, i_abs in biz2idx.items():
            j = i_abs - U if i_abs >= U else i_abs
            if 0 <= j < B:
                biz[j] = bid

        idx_payload = {"user_ids": users, "business_ids": biz}

        os.makedirs(yelp_dir, exist_ok=True)
        p1 = os.path.join(yelp_dir, "yelp_index.json")
        with open(p1, "w", encoding="utf-8") as f:
            json.dump(idx_payload, f)
        print(f"[yelp] wrote {p1}")

        mirror_dir = os.path.join("data", "Yelp")
        os.makedirs(mirror_dir, exist_ok=True)
        p2 = os.path.join(mirror_dir, "yelp_index.json")
        with open(p2, "w", encoding="utf-8") as f:
            json.dump(idx_payload, f)
        print(f"[yelp] mirrored → {p2}")
    except Exception as e:
        print(f"[yelp] WARN: failed to write yelp_index.json: {e}")



def _load_yelp_simplified(yelp_dir: str) -> Data:
    """
    Minimal Yelp loader (user–business review graph).
    - Nodes: users + businesses, contiguous ids [0..N-1]
    - Edges: undirected U--B from reviews
    - Features: [is_user, is_business, deg_norm]
    Also saves an index file yelp_index.json so deletion can map ext IDs → int nodes.
    """
    # locate files
    def find_one(cands):
        for name in cands:
            p = os.path.join(yelp_dir, name)
            if os.path.exists(p):
                return p
        return None

    user_path = find_one(("users.json", "user.json", "yelp_academic_dataset_user.json", "yelp_academic_dataset_user.jsonl"))
    biz_path  = find_one(("businesses.json", "business.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_business.jsonl"))
    rev_path  = find_one(("reviews.json", "review.json", "yelp_academic_dataset_review.json", "yelp_academic_dataset_review.jsonl"))

    if not (user_path and biz_path and rev_path):
        raise FileNotFoundError(
            f"Could not find user/business/review files under {yelp_dir}. "
            f"Looked for users.json|user.json|yelp_academic_dataset_user.json(l) etc."
        )

    # optional caps (for prototyping)
    def _cap(env, default=None):
        v = os.getenv(env, None)
        return int(v) if (v and v.isdigit()) else default

    users = _read_json_or_jsonl(user_path, max_items=_cap("YELP_MAX_USERS"))
    biz   = _read_json_or_jsonl(biz_path,  max_items=_cap("YELP_MAX_BIZ"))
    revs  = _read_json_or_jsonl(rev_path,  max_items=_cap("YELP_MAX_REVS"))

    # map external IDs -> integer node IDs
    def get_u_id(u):  return u.get("user_id") or u.get("id") or u.get("userId") or u.get("userID")
    def get_b_id(b):  return b.get("business_id") or b.get("id") or b.get("businessId") or b.get("businessID")
    def get_r_uid(r): return r.get("user_id") or r.get("userId") or r.get("userID")
    def get_r_bid(r): return r.get("business_id") or r.get("businessId") or r.get("businessID")

    uid2nid = {}
    for u in users:
        uid = get_u_id(u)
        if uid is not None and uid not in uid2nid:
            uid2nid[uid] = len(uid2nid)

    offset   = len(uid2nid)
    bid2nid  = {}
    for b in biz:
        bid = get_b_id(b)
        if bid is not None and bid not in bid2nid:
            bid2nid[bid] = offset + len(bid2nid)

    N_users = len(uid2nid)
    N_biz   = len(bid2nid)
    N       = N_users + N_biz
    if N == 0:
        raise RuntimeError("Yelp loader: built zero nodes; check input files/keys.")

    # edges from reviews (user_id, business_id); undirected
    src, dst = [], []
    miss_u = miss_b = 0
    for r in revs:
        uext = get_r_uid(r); bext = get_r_bid(r)
        u = uid2nid.get(uext); b = bid2nid.get(bext)
        if u is None: miss_u += 1
        if b is None: miss_b += 1
        if (u is None) or (b is None):
            continue
        src.append(u); dst.append(b)
        src.append(b); dst.append(u)

    if not src:
        raise RuntimeError(
            "Yelp loader: no edges constructed — "
            f"unmatched IDs? (missing users={miss_u}, missing businesses={miss_b})"
        )

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # simple features: [is_user, is_business, deg_norm]
    deg = torch.bincount(edge_index[0], minlength=N).float()
    deg_norm = (deg / (deg.max() if deg.max() > 0 else 1.0)).unsqueeze(1)
    is_user = torch.zeros((N, 1), dtype=torch.float32); is_user[:N_users] = 1.0
    is_biz  = 1.0 - is_user
    x = torch.cat([is_user, is_biz, deg_norm], dim=1)  # [N,3]

    # save index so deletion can map string IDs → internal ints
    _save_yelp_index(uid2nid, bid2nid, yelp_dir)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = N
    data.num_edge_type = 1
    return data

# ------------------------ existing code below (unchanged for other datasets) ------------------------

def get_original_data(d):
    data_dir = './data'
    if d in ['Cora', 'PubMed', 'DBLP']:
        dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d == 'CiteSeer' or d in ['Cora_p', 'PubMed_p', 'Citeseer_p']:
        dataset = Planetoid(os.path.join(data_dir, d), d.split('_')[0], transform=T.NormalizeFeatures())
    elif d in ['CS', 'Physics']:
        dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Amazon']:
        dataset = Amazon(os.path.join(data_dir, d), 'Photo', transform=T.NormalizeFeatures())
    elif d in ['Reddit']:
        dataset = Reddit2(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif d in ['Flickr']:
        dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif 'ogbl' in d:
        dataset = PygLinkPropPredDataset(root=os.path.join(data_dir, d), name=d)
    elif d == "Yelp":
        yelp_dir = os.environ.get("YELP_DIR", os.path.join("data", "yelp"))
        candidates = (
            "users.json", "user.json", "yelp_academic_dataset_user.json", "yelp_academic_dataset_user.jsonl",
            "businesses.json", "business.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_business.jsonl",
            "reviews.json", "review.json", "yelp_academic_dataset_review.json", "yelp_academic_dataset_review.jsonl"
        )
        found_any = any(os.path.exists(os.path.join(yelp_dir, c)) for c in candidates)
        if not found_any:
            raise FileNotFoundError(
                f"Yelp JSON/JSONL not found. Set YELP_DIR or place users.json/businesses.json/reviews.json "
                f"(or yelp_academic_dataset_* files) under {yelp_dir}"
            )
        return _load_yelp_simplified(yelp_dir)
    else:
        raise NotImplementedError(f"{d} not supported.")
    data = dataset[0]
    return data

def gen_inout_mask(data):
    _, local_edges, _, mask = k_hop_subgraph(
        data.val_pos_edge_index.flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    distant_edges = data.train_pos_edge_index[:, ~mask]
    print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])
    in_mask = mask
    out_mask = ~mask
    return {'in': in_mask, 'out': out_mask}

def split_forget_retain(data, df_size, subset='in'):
    if df_size >= 100:  # df_size is number of nodes/edges to be deleted
        df_size = int(df_size)
    else:
        df_size = int(df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')
    df_mask_all = gen_inout_mask(data)[subset]
    df_nonzero = df_mask_all.nonzero().squeeze()
    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]
    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False
    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # Collect enclosing subgraph of Df for loss computation
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        1,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop
    assert not is_undirected(data.train_pos_edge_index)

    train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(
        data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
    two_hop_mask = two_hop_mask.bool()
    df_mask = df_mask.bool()
    dr_mask = ~df_mask

    data.train_pos_edge_index = train_pos_edge_index
    data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask
    return data

def split_shadow_target(data):
    if data.y != None:
        labels = data.y.numpy()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for target_index, shadow_index in sss.split(torch.zeros(labels.shape[0]), labels):
            target_nodes = torch.from_numpy(target_index)
            shadow_nodes = torch.from_numpy(shadow_index)
    else:
        ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        all_indices = torch.arange(data.x.shape[0])
        for target_index, shadow_index in ss.split(all_indices):
            target_nodes = all_indices[target_index]
            shadow_nodes = all_indices[shadow_index]

    target_edge_index, target_edge_attr = subgraph(
        target_nodes, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True
    )
    shadow_edge_index, shadow_edge_attr = subgraph(
        shadow_nodes, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True
    )

    target_data = Data(
        x=data.x[target_nodes],
        edge_index=target_edge_index,
        edge_attr=target_edge_attr,
        y=data.y[target_nodes] if data.y != None else None,
    )
    shadow_data = Data(
        x=data.x[shadow_nodes],
        edge_index=shadow_edge_index,
        edge_attr=shadow_edge_attr,
        y=data.y[shadow_nodes] if data.y != None else None,
    )
    return shadow_data, target_data

def train_test_split_edges_no_neg_adj_mask(data, val_ratio: float = 0.05, test_ratio: float = 0.05, two_hop_degree=None):
    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    mask = row < col
    row, col = row[mask], col[mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    if two_hop_degree is not None:
        low_degree_mask = two_hop_degree < 50
        low = low_degree_mask.nonzero().squeeze()
        high = (~low_degree_mask).nonzero().squeeze()
        low = low[torch.randperm(low.size(0))]
        high = high[torch.randperm(high.size(0))]
        perm = torch.cat([low, high])
    else:
        perm = torch.randperm(row.size(0))
    row = row[perm]
    col = col[perm]
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.train_pos_edge_index, data.train_pos_edge_attr = None
    else:
        data.train_pos_edge_index = data.train_pos_edge_index
    assert not is_undirected(data.train_pos_edge_index)

    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])
    data.test_neg_edge_index = neg_edge_index

    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])
    data.val_neg_edge_index = neg_edge_index
    return data

# ---- the rest of original helpers (unchanged) ----
def load_dict(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            mapping[l[0]] = l[1]
    return mapping

def load_edges(filename):
    with open(filename, 'r') as f:
        r = f.readlines()
    r = [i.strip().split('\t') for i in r]
    return r

def generate_true_dict(all_triples):
    heads = {(r, t) : [] for _, r, t in all_triples}
    tails = {(h, r) : [] for h, r, _ in all_triples}
    for h, r, t in all_triples:
        heads[r, t].append(h)
        tails[h, r].append(t)
    return heads, tails

def get_loader(args, delete=[]):
    prefix = os.path.join('./data', args.dataset)
    train = load_edges(os.path.join(prefix, 'train.txt'))
    valid = load_edges(os.path.join(prefix, 'valid.txt'))
    test = load_edges(os.path.join(prefix, 'test.txt'))
    train = [(int(i[0]), int(i[1]), int(i[2])) for i in train]
    valid = [(int(i[0]), int(i[1]), int(i[2])) for i in valid]
    test = [(int(i[0]), int(i[1]), int(i[2])) for i in test]
    train_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in train]
    valid_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in valid]
    test_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in test]
    train = train + train_rev
    valid = valid + valid_rev
    test = test + test_rev
    all_edge = train + valid + test
    true_triples = generate_true_dict(all_edge)
    edge = torch.tensor([(int(i[0]), int(i[2])) for i in all_edge], dtype=torch.long).t()
    edge_type = torch.tensor([int(i[1]) for i in all_edge], dtype=torch.long)
    train_size = len(train)
    valid_size = len(valid)
    test_size = len(test)
    total_size = train_size + valid_size + test_size
    train_mask = torch.zeros((total_size,)).bool()
    train_mask[:train_size] = True
    valid_mask = torch.zeros((total_size,)).bool()
    valid_mask[train_size:train_size + valid_size] = True
    test_mask = torch.zeros((total_size,)).bool()
    test_mask[-test_size:] = True
    num_nodes = edge.flatten().unique().shape[0]
    num_edges = edge.shape[1]
    num_edge_type = edge_type.unique().shape[0]
    x = torch.rand((num_nodes, args.in_dim))
    if len(delete) > 0:
        delete_idx = torch.tensor(delete, dtype=torch.long)
        num_train_edges = train_size // 2
        train_mask[delete_idx] = False
        train_mask[delete_idx + num_train_edges] = False
        train_size -= 2 * len(delete)
    node_id = torch.arange(num_nodes)
    dataset = Data(
        edge_index=edge, edge_type=edge_type, x=x, node_id=node_id,
        train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    dataloader = GraphSAINTRandomWalkSampler(
        dataset, batch_size=args.batch_size, walk_length=args.walk_length, num_steps=args.num_steps)
    print(f'Dataset: {args.dataset}, Num nodes: {num_nodes}, Num edges: {num_edges//2}, Num relation types: {num_edge_type}')
    print(f'Train edges: {train_size//2}, Valid edges: {valid_size//2}, Test edges: {test_size//2}')
    return dataloader, valid, test, true_triples, num_nodes, num_edges, num_edge_type
