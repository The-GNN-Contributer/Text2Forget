import os, re, json, csv, argparse, random
from typing import List, Dict, Any, Tuple
from torch_geometric.utils import to_networkx

def load_corafull(seed: int):
    try:
        from torch_geometric.datasets import CoraFull
    except Exception as e:
        raise RuntimeError("CoraFull requires torch-geometric.") from e
    ds = CoraFull(root="./data/corafull")
    data = ds[0]
    G = to_networkx(data, to_undirected=True)

    import random
    FIRST = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Heidi","Ivan","Judy"]
    LAST  = ["Kim","Lee","Patel","Garcia","Mueller","Singh","Chen","Khan","Novak","Ito"]
    ADJ   = ["Efficient","Robust","Scalable","Neural","Probabilistic","Graph","Attention-based","Bayesian"]
    NOUN  = ["Embeddings","Classifiers","Networks","Representations","Inference","Reasoning","Link Prediction","GNNs"]
    DOMAIN= ["for Citation Networks","for Graph Mining","for Text Classification","for Scientific Articles"]
    rng = random.Random(seed)
    def synth_title(i):
        return f"{rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice(DOMAIN)}"
    def synth_year(i): return 1998 + (i % 21)
    for i in range(G.number_of_nodes()):
        title = synth_title(i)
        year  = synth_year(i)
        G.nodes[i]["type"] = "paper"
        G.nodes[i]["title"] = title
        G.nodes[i]["year"]  = year
        G.nodes[i]["name"]  = f"“{title}” ({year})"
    return G

def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

import networkx as nx

# -------------------- Dataset loaders --------------------

FIRST = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Heidi","Ivan","Judy","Katerina","Luis","Maya","Noah","Omar","Priya","Quinn","Ravi","Sara","Tariq"]
LAST  = ["Kim","Lee","Patel","Garcia","Mueller","Singh","Chen","Khan","Novak","Ito","Silva","Haddad","Sato","Nguyen","Kumar","Huang","Rossi","Gonzalez","Kowalski","Hansen"]
ADJ   = ["Efficient","Robust","Scalable","Neural","Probabilistic","Graph","Attention-based","Bayesian","Contrastive","Self-supervised","Hierarchical","Distributed","Incremental","Adaptive","Interpretable"]
NOUN  = ["Embeddings","Classifiers","Networks","Representations","Inference","Reasoning","Link Prediction","GNNs","Transformers","Regularization","Sampling","Clustering","Label Propagation"]
DOMAIN= ["for Citation Networks","for Graph Mining","for Text Classification","for Bioinformatics","for Recommendation","for Social Graphs","for Semi-supervised Learning","for Scientific Articles"]

def synth_title_det(i: int, seed: int) -> str:
    r = random.Random((seed+1)*(i+17))
    return f"{r.choice(ADJ)} {r.choice(NOUN)} {r.choice(DOMAIN)}"

def synth_authors_det(i: int, seed: int, k=2) -> str:
    r = random.Random((seed+3)*(i+23))
    names = [f"{r.choice(FIRST)} {r.choice(LAST)}" for _ in range(r.choice([2,3]))]
    return ", ".join(names)

def synth_year_det(i: int, seed: int) -> int:
    r = random.Random((seed+5)*(i+31))
    return r.randint(1998, 2018)

def load_planetoid(name: str, seed: int):
    try:
        import torch
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_networkx
    except Exception as e:
        raise RuntimeError("Planetoid requires PyTorch Geometric. Install it to use cora/citeseer/pubmed.") from e
    ds = Planetoid(root=f"./data/{name}", name=name.capitalize())
    data = ds[0]
    G = to_networkx(data, to_undirected=True)
    for i in range(G.number_of_nodes()):
        title = synth_title_det(i, seed)
        authors = synth_authors_det(i, seed)
        year = synth_year_det(i, seed)
        G.nodes[i]["type"] = "paper"
        G.nodes[i]["title"] = title
        G.nodes[i]["authors"] = authors
        G.nodes[i]["year"] = year
        G.nodes[i]["name"] = f"“{title}” ({year})"
    return G

def load_toy(seed=42):
    r = random.Random(seed+101)
    G = nx.Graph()
    for i in range(12):
        G.add_node(f"u{i}", type="user", name=f"{r.choice(FIRST)} {r.choice(LAST)}")
    cafes = ["Cafe Luna","Blue Bottle","Sunrise Deli","Bean There","Grounded Lab","Brew & Bloom"]
    for j in range(6):
        G.add_node(f"b{j}", type="business", name=cafes[j])
    rid = 0
    for u in range(12):
        for _ in range(r.randint(1,3)):
            b = r.randint(0,5)
            date = f"{r.randint(2017,2021)}-{r.randint(1,12):02d}-{r.randint(1,28):02d}"
            G.add_edge(f"u{u}", f"b{b}", type="review", review_id=f"r{rid}", stars=r.randint(1,5), date=date)
            rid += 1
    return G

def load_yelp(folder: str):
    import json, os
    G = nx.Graph()

    def _is_json_array(path: str) -> bool:
        # True if first non-space char is '['
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                ch = f.read(1)
                if not ch:
                    return False
                if ch.isspace():
                    continue
                return ch == "["

    def _iter_jsonl(path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _iter_array(path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
            for rec in obj:
                yield rec

    def _iter_smart(path: str):
        return _iter_array(path) if _is_json_array(path) else _iter_jsonl(path)

    def _pick_file(preferred: str, fallback: str) -> str:
        p = os.path.join(folder, preferred)
        if os.path.exists(p): return p
        q = os.path.join(folder, fallback)
        if os.path.exists(q): return q
        raise FileNotFoundError(f"Neither {preferred} nor {fallback} found in {folder}")

    users_path = _pick_file("yelp_academic_dataset_user.json", "users.json")
    biz_path   = _pick_file("yelp_academic_dataset_business.json", "businesses.json")
    rev_path   = _pick_file("yelp_academic_dataset_review.json", "reviews.json")

    # Users
    for u in _iter_smart(users_path):
        uid = u.get("user_id")
        if not uid: continue
        G.add_node(uid, type="user", name=u.get("name","User"))

    # Businesses
    for b in _iter_smart(biz_path):
        bid = b.get("business_id")
        if not bid: continue
        G.add_node(bid, type="business", name=b.get("name","Business"))

    # Reviews (edges)
    for r in _iter_smart(rev_path):
        uid = r.get("user_id"); bid = r.get("business_id")
        if not uid or not bid: continue
        if uid not in G: G.add_node(uid, type="user", name="User")
        if bid not in G: G.add_node(bid, type="business", name="Business")
        G.add_edge(uid, bid,
                   type="review",
                   review_id=r.get("review_id",""),
                   stars=r.get("stars",None),
                   date=r.get("date") or r.get("date_added") or "")
    return G

def build_surface_forms(G: nx.Graph) -> List[Dict[str,Any]]:
    recs = []
    for n, a in G.nodes(data=True):
        if a.get("type") == "paper":
            disp = f"“{a.get('title','Untitled')}” ({a.get('year','')})"
            extra = f"authors {a.get('authors','')}"
        else:
            disp = a.get("name", str(n)); extra = ""
        recs.append({
            "doc_id": f"node::{n}",
            "type": "node",
            "node_id": str(n),
            "display": disp,
            "text": f"{a.get('type','node')} named {disp} {extra}".strip()
        })
    for u, v, a in G.edges(data=True):
        et = a.get("type","edge")
        ud = G.nodes[u].get("name", str(u))
        vd = G.nodes[v].get("name", str(v))
        if G.nodes[u].get("type")=="paper" and "title" in G.nodes[u]:
            ud = f"“{G.nodes[u]['title']}” ({G.nodes[u]['year']})"
        if G.nodes[v].get("type")=="paper" and "title" in G.nodes[v]:
            vd = f"“{G.nodes[v]['title']}” ({G.nodes[v]['year']})"
        if et == "review":
            title = f"review by {ud} about {vd} on {a.get('date','')}"
        else:
            title = f"citation from {ud} to {vd}"
        recs.append({
            "doc_id": f"edge::{u}::{v}::{a.get('review_id','')}",
            "type": "edge",
            "src": str(u), "dst": str(v),
            "edge_type": et, "review_id": a.get("review_id",""),
            "display": title,
            "text": f"{title} (hidden ids: {u},{v})"
        })
    return recs

# -------------------- Retrieval --------------------

class HybridIndex:
    def __init__(self, model="BAAI/bge-small-en-v1.5", use_bm25=True, normalize=True):
        self.model_id = model; self.use_bm25 = use_bm25; self.normalize = normalize
        self.records = []; self._bm25=None; self._emb=None; self._X=None; self._nn=None
    def _device(self): return "cuda" if has_cuda() else "cpu"
    def fit(self, records):
        self.records = records
        texts = [r.get("display","") + " " + r.get("text","") for r in records]
        if self.use_bm25:
            from rank_bm25 import BM25Okapi
            tokenized = [t.lower().split() for t in texts]
            self._bm25 = BM25Okapi(tokenized)
        from sentence_transformers import SentenceTransformer
        from sklearn.neighbors import NearestNeighbors
        self._emb = SentenceTransformer(self.model_id, device=self._device())
        X = self._emb.encode(texts, normalize_embeddings=self.normalize, convert_to_numpy=True, show_progress_bar=False)
        self._X = X
        self._nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(X)
    def bm25_scores(self, query: str):
        if self._bm25 is None: return []
        tok = query.lower().split()
        s = self._bm25.get_scores(tok)
        return list(enumerate(s))  # [(idx, score), ...]
    def emb_scores(self, query: str, k=10):
        import numpy as np
        qv = self._emb.encode([query], normalize_embeddings=self.normalize, convert_to_numpy=True)
        d, idx = self._nn.kneighbors(qv, n_neighbors=min(k, len(self.records)))
        out = []
        for j, i in enumerate(idx[0]):
            sim = 1.0 - float(d[0][j])
            out.append((int(i), sim))
        return out

# -------------------- Mention extraction --------------------

def extract_mentions(text: str) -> List[str]:
    quoted = re.findall(r"“([^”]+)”|\"([^\"]+)\"", text)
    mentions = [q[0] or q[1] for q in quoted if (q[0] or q[1])]
    caps = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,4})\b", text)
    m2 = re.findall(r"review of ([A-Z][^\n,]+)", text, flags=re.IGNORECASE)
    for m in caps + m2:
        s = m.strip()
        if len(s) >= 3:
            mentions.append(s)
    uniq = []
    for m in mentions:
        if m not in uniq:
            uniq.append(m)
    return uniq[:5]

def guess_scope(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["connection","link","review","citation","reference","relationship"]): return "edge"
    if any(k in t for k in ["attribute","profile","field","location","bio","metadata"]): return "attribute"
    if any(k in t for k in ["category","tag","label"]): return "label"
    return "node"

# -------------------- Grounding confidence --------------------

def normalize_margin(top1, top2, eps=1e-6):
    if top1 is None or top2 is None: return 0.0
    if top1 <= 0 and top2 <= 0:
        return 0.0
    return max(0.0, (top1 - top2) / (abs(top1) + abs(top2) + eps))

def grounding_confidence(hi: HybridIndex, text: str, scope_hint: str, resolved_doc_id: str = None) -> Dict[str, float]:
    mentions = extract_mentions(text) or [text]
    # Aggregate over mentions: pick the strongest margin
    best = {"bm25_margin":0.0,"emb_margin":0.0,"agreement":0.0,"consistency":0.0}
    doc_counts = {}
    for q in mentions:
        bm = hi.bm25_scores(q)
        em = hi.emb_scores(q, k=5)
        bm_sorted = sorted(bm, key=lambda x:x[1], reverse=True)[:2] if bm else []
        em_sorted = em[:2] if em else []
        b1 = bm_sorted[0][1] if len(bm_sorted)>0 else None
        b2 = bm_sorted[1][1] if len(bm_sorted)>1 else None
        e1 = em_sorted[0][1] if len(em_sorted)>0 else None
        e2 = em_sorted[1][1] if len(em_sorted)>1 else None
        bm_doc1 = hi.records[bm_sorted[0][0]]["doc_id"] if len(bm_sorted)>0 else None
        em_doc1 = hi.records[em_sorted[0][0]]["doc_id"] if len(em_sorted)>0 else None
        bm25_margin = normalize_margin(b1, b2)
        emb_margin  = normalize_margin(e1, e2)
        agr = 1.0 if (bm_doc1 is not None and em_doc1 is not None and bm_doc1 == em_doc1) else 0.0
        best["bm25_margin"] = max(best["bm25_margin"], bm25_margin)
        best["emb_margin"]  = max(best["emb_margin"],  emb_margin)
        best["agreement"]   = max(best["agreement"], agr)
        # count for consistency
        for d in [bm_doc1, em_doc1]:
            if d: doc_counts[d] = doc_counts.get(d,0)+1
    # consistency: top doc across mentions/modalities
    total_votes = sum(doc_counts.values()) or 1
    top_votes = max(doc_counts.values()) if doc_counts else 0
    best["consistency"] = top_votes / total_votes
    # final confidence
    conf = 0.4*best["bm25_margin"] + 0.4*best["emb_margin"] + 0.1*best["agreement"] + 0.1*best["consistency"]
    best["confidence"] = conf
    return best

# -------------------- Main ranking logic --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_edges", type=int, default=0)
    ap.add_argument("--max_index", type=int, default=300000)
    ap.add_argument("--no-bm25", action="store_true")
    
    ap.add_argument("--dataset", choices=["toy","cora", "corafull", "citeseer","pubmed","yelp"], required=True)
    ap.add_argument("--yelp-dir", type=str, default=None, help="Folder for Yelp JSONs (if dataset=yelp)")
    ap.add_argument("--in", dest="in_dir", type=str, required=True, help="Directory with requests_*.jsonl or scored_*.jsonl")
    ap.add_argument("--out", dest="out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--file", type=str, default="", help="Specify a particular file to rank (overrides auto-detect)")
    ap.add_argument("--topk", type=int, default=0, help="If >0, also write topK jsonl")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    #recs = build_surface_forms(G)
    # Load dataset & build index
    if args.dataset == "toy":
        G = load_toy(args.seed)
        recs = build_surface_forms(G)
    elif args.dataset in ("cora","citeseer","pubmed"):
        G = load_planetoid(args.dataset, args.seed)
        recs = build_surface_forms(G)
    elif args.dataset == "corafull":
        G = load_corafull(args.seed)
        recs = build_surface_forms(G)
    elif args.dataset == "yelp":
        G = load_yelp(args.yelp_dir)
        recs = build_surface_forms(G)
        if not args.index_edges:
            recs = [r for r in recs if r["type"] == "node"]  # skip 6.7M review edges
        if args.max_index and len(recs) > args.max_index:
            import random
            random.seed(args.seed)
            recs = random.sample(recs, args.max_index)
        print(f"[ranker] yelp records after filtering: {len(recs)}")
    else:
        if not args.yelp_dir:
            raise SystemExit("--yelp-dir is required for dataset=yelp")
        G = load_yelp(args.yelp_dir)
    #recs = build_surface_forms(G)
    hi = HybridIndex(model=args.embed_model, use_bm25=not args.no_bm25, normalize=True)
    hi.fit(recs)

    # Decide input file
    if args.file:
        in_path = args.file
    else:
        cand1 = os.path.join(args.in_dir, f"scored_{args.dataset}.jsonl")
        cand2 = os.path.join(args.in_dir, f"requests_{args.dataset}.jsonl")
        in_path = cand1 if os.path.exists(cand1) else cand2
        if not os.path.exists(in_path):
            raise SystemExit("Could not find scored_*.jsonl or requests_*.jsonl in --in directory.")

    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            text = ex["text"]
            scope = ex.get("scope_hint") or guess_scope(text)
            # AHL components if present; else recompute simple proxies
            ahl = ex.get("ahl", {})
            legal = float(ahl.get("legal", 0.5))
            variety = float(ahl.get("variety", 0.5))
            detector = float(ahl.get("detector", 0.5))

            # resolved doc id (if present) to check agreement (optional)
            resolved = ex.get("resolved_targets") or ex.get("parsed",{}).get("targets",[])
            resolved_doc = None
            if resolved:
                t = resolved[0]
                if t["type"] == "edge":
                    resolved_doc = f"edge::{t['src']}::{t['dst']}::{t.get('review_id','')}"
                else:
                    resolved_doc = f"node::{t['node_id']}"

            gc = grounding_confidence(hi, text, scope, resolved_doc)
            surface = 0.5*variety + 0.5*detector
            composite = 0.45*gc["confidence"] + 0.35*legal + 0.20*surface

            rows.append({
                "request_id": ex.get("request_id",""),
                "dataset": ex.get("dataset", args.dataset),
                "scope": scope,
                "legal_regime": ex.get("legal_regime",""),
                "legal_basis_code": ex.get("legal_basis_code",""),
                "GroundingConfidence": round(gc["confidence"], 4),
                "BM25_margin": round(gc["bm25_margin"], 4),
                "EMB_margin": round(gc["emb_margin"], 4),
                "Agreement": round(gc["agreement"], 4),
                "Consistency": round(gc["consistency"], 4),
                "Legal": round(legal, 4),
                "Variety": round(variety, 4),
                "Detector": round(detector, 4),
                "CompositeScore": round(composite, 4),
                "text": text.replace("\n"," ").strip()[:500]
            })

    rows.sort(key=lambda r: r["CompositeScore"], reverse=True)

    # Write CSV
    out_csv = os.path.join(args.out_dir, f"ranked_{args.dataset}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Optional topK JSONL
    if args.topk and args.topk > 0:
        out_top = os.path.join(args.out_dir, f"topk_{args.dataset}.jsonl")
        with open(out_top, "w", encoding="utf-8") as f:
            for r in rows[:args.topk]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ranker] wrote {out_csv} (top {args.topk} also written if requested)")
    print(f"[ranker] example top 3:")
    for r in rows[:3]:
        print(f"  - {r['request_id']} | {r['CompositeScore']:.3f} | {r['text'][:80]}...")

if __name__ == "__main__":
    main()
