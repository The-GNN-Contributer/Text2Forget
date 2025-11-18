import os, re, json, uuid, random, argparse
from typing import List, Dict, Any, Tuple
from collections import Counter, deque

# -------------------- utils --------------------
def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass

# -------------------- humanization helpers --------------------
FIRST = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Heidi","Ivan","Judy","Katerina","Luis","Maya","Noah","Omar","Priya","Quinn","Ravi","Sara","Tariq"]
LAST  = ["Kim","Lee","Patel","Garcia","Mueller","Singh","Chen","Khan","Novak","Ito","Silva","Haddad","Sato","Nguyen","Kumar","Huang","Rossi","Gonzalez","Kowalski","Hansen"]
ADJ   = ["Efficient","Robust","Scalable","Neural","Probabilistic","Graph","Attention-based","Bayesian","Contrastive","Self-supervised","Hierarchical","Distributed","Incremental","Adaptive","Interpretable"]
NOUN  = ["Embeddings","Classifiers","Networks","Representations","Inference","Reasoning","Link Prediction","GNNs","Transformers","Regularization","Sampling","Clustering","Label Propagation"]
DOMAIN= ["for Citation Networks","for Graph Mining","for Text Classification","for Bioinformatics","for Recommendation","for Social Graphs","for Semi-supervised Learning","for Scientific Articles"]

def synth_title_det(i: int, seed: int) -> str:
    r = random.Random((seed+1)*(i+17))
    return f"{r.choice(ADJ)} {r.choice(NOUN)} {r.choice(DOMAIN)}"

def synth_authors_det(i: int, seed: int) -> str:
    r = random.Random((seed+3)*(i+23))
    names = [f"{r.choice(FIRST)} {r.choice(LAST)}" for _ in range(r.choice([2,3]))]
    return ", ".join(names)

def synth_year_det(i: int, seed: int) -> int:
    r = random.Random((seed+5)*(i+31))
    return r.randint(1998, 2018)

# -------------------- datasets --------------------
import networkx as nx

def load_toy(seed=42):
    set_seed(seed)
    G = nx.Graph()
    r = random.Random(seed+101)
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
    meta = {"name":"toy","node_count":G.number_of_nodes(),"edge_count":G.number_of_edges()}
    return G, meta

def load_planetoid(name: str, seed: int):
    try:
        import torch
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_networkx
    except Exception as e:
        raise RuntimeError("Planetoid requires PyTorch Geometric (cora/citeseer/pubmed).") from e
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
    meta = {"name":name,"node_count":G.number_of_nodes(),"edge_count":G.number_of_edges()}
    return G, meta

def load_corafull(seed: int):
    try:
        from torch_geometric.datasets import CoraFull
        from torch_geometric.utils import to_networkx
    except Exception as e:
        raise RuntimeError("CoraFull requires PyTorch Geometric.") from e
    ds = CoraFull(root="./data/corafull")
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
    meta = {"name":"corafull","node_count":G.number_of_nodes(),"edge_count":G.number_of_edges()}
    return G, meta

# ---- robust Yelp readers (JSON / JSONL, UTF-8) ----
def _file_starts_with_array(path: str) -> bool:
    with open(path, "rb") as fb:
        head = fb.read(1024)
    return head.lstrip().startswith(b"[")

def _iter_jsonl(path: str, limit: int | None = None):
    cnt = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
                cnt += 1
                if limit and cnt >= limit: break
            except json.JSONDecodeError:
                continue

def _load_json_array(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

def load_yelp(folder: str):
    """
    Accepts either:
      - simplified arrays: users.json, businesses.json, reviews.json
      - official: yelp_academic_dataset_user/business/review.json or .jsonl
    Always UTF-8 and robust to JSON vs JSONL.
    """
    users_path = os.path.join(folder, "users.json")
    biz_path   = os.path.join(folder, "businesses.json")
    revs_path  = os.path.join(folder, "reviews.json")

    if not (os.path.exists(users_path) and os.path.exists(biz_path) and os.path.exists(revs_path)):
        cand_users = ["yelp_academic_dataset_user.json", "yelp_academic_dataset_user.jsonl"]
        cand_biz   = ["yelp_academic_dataset_business.json", "yelp_academic_dataset_business.jsonl"]
        cand_revs  = ["yelp_academic_dataset_review.json", "yelp_academic_dataset_review.jsonl"]
        users_path = next((os.path.join(folder, p) for p in cand_users if os.path.exists(os.path.join(folder, p))), None)
        biz_path   = next((os.path.join(folder, p) for p in cand_biz   if os.path.exists(os.path.join(folder, p))), None)
        revs_path  = next((os.path.join(folder, p) for p in cand_revs  if os.path.exists(os.path.join(folder, p))), None)
        if not (users_path and biz_path and revs_path):
            raise SystemExit(f"Could not find Yelp files in {folder}.")

    G = nx.Graph()

    # users
    if users_path.endswith(".json") and _file_starts_with_array(users_path):
        users = _load_json_array(users_path)
        for u in users:
            uid = u.get("user_id") or u.get("id")
            if uid: G.add_node(uid, type="user", name=u.get("name","User"))
    else:
        for u in _iter_jsonl(users_path):
            uid = u.get("user_id") or u.get("id")
            if uid: G.add_node(uid, type="user", name=u.get("name","User"))

    # businesses
    if biz_path.endswith(".json") and _file_starts_with_array(biz_path):
        biz = _load_json_array(biz_path)
        for b in biz:
            bid = b.get("business_id") or b.get("id")
            if bid: G.add_node(bid, type="business", name=b.get("name","Business"))
    else:
        for b in _iter_jsonl(biz_path):
            bid = b.get("business_id") or b.get("id")
            if bid: G.add_node(bid, type="business", name=b.get("name","Business"))

    # reviews (edges)
    if revs_path.endswith(".json") and _file_starts_with_array(revs_path):
        it = _load_json_array(revs_path)
    else:
        it = _iter_jsonl(revs_path)

    for r in it:
        uid = r.get("user_id"); bid = r.get("business_id")
        if uid and bid:
            G.add_edge(uid, bid, type="review",
                       review_id=r.get("review_id") or r.get("reviewId") or "",
                       stars=r.get("stars"), date=r.get("date") or "")

    meta = {"name":"yelp","node_count":G.number_of_nodes(),"edge_count":G.number_of_edges()}
    return G, meta

# -------------------- surface forms (split nodes / edges) --------------------
def build_surface_forms_split(G: nx.Graph):
    nodes, edges = [], []
    for n, a in G.nodes(data=True):
        if a.get("type") == "paper":
            disp = f"“{a.get('title','Untitled')}” ({a.get('year','')})"
            extra = f"authors {a.get('authors','')}"
        else:
            disp = a.get("name", str(n)); extra = ""
        nodes.append({
            "doc_id": f"node::{n}",
            "type": "node", "node_id": str(n),
            "display": disp, "text": f"{a.get('type','node')} named {disp} {extra}".strip()
        })
    for u, v, a in G.edges(data=True):
        et = a.get("type","edge")
        ud = G.nodes[u].get("name", str(u))
        vd = G.nodes[v].get("name", str(v))
        if G.nodes[u].get("type")=="paper" and "title" in G.nodes[u]:
            ud = f"“{G.nodes[u]['title']}” ({G.nodes[u]['year']})"
        if G.nodes[v].get("type")=="paper" and "title" in G.nodes[v]:
            vd = f"“{G.nodes[v]['title']}” ({G.nodes[v]['year']})"
        title = f"review by {ud} about {vd} on {a.get('date','')}" if et=="review" else f"citation from {ud} to {vd}"
        edges.append({
            "doc_id": f"edge::{u}::{v}::{a.get('review_id','')}",
            "type": "edge", "src": str(u), "dst": str(v),
            "edge_type": et, "review_id": a.get("review_id",""),
            "display": title, "text": f"{title} (hidden ids: {u},{v})"
        })
    return nodes, edges

# -------------------- HybridIndex (BM25 + embeddings + optional FAISS) --------------------
class HybridIndex:
    def __init__(self, model="BAAI/bge-small-en-v1.5", use_bm25=True, normalize=True, batch_size=256, use_faiss=True):
        self.model_id = model; self.use_bm25 = use_bm25; self.normalize = normalize
        self.batch_size = batch_size; self.use_faiss = use_faiss
        self.records = []; self._bm25=None; self._emb=None; self._X=None; self._nn=None; self._faiss=None
    def _device(self): return "cuda" if has_cuda() else "cpu"
    def fit(self, records):
        self.records = records
        texts = [r.get("display","") + " " + r.get("text","") for r in records]
        if self.use_bm25:
            from rank_bm25 import BM25Okapi
            tokenized = [t.lower().split() for t in texts]
            self._bm25 = BM25Okapi(tokenized)
        from sentence_transformers import SentenceTransformer
        self._emb = SentenceTransformer(self.model_id, device=self._device())
        X = self._emb.encode(texts, convert_to_numpy=True, normalize_embeddings=self.normalize,
                             batch_size=self.batch_size, show_progress_bar=True)
        self._X = X
        if self.use_faiss:
            try:
                import faiss
                xb = X.astype('float32')
                if self.normalize:
                    faiss.normalize_L2(xb)
                    self._faiss = faiss.IndexFlatIP(xb.shape[1])
                else:
                    self._faiss = faiss.IndexFlatL2(xb.shape[1])
                self._faiss.add(xb); return
            except Exception:
                self._faiss = None
        from sklearn.neighbors import NearestNeighbors
        self._nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(X)
    def search(self, query: str, k: int = 10):
        scores = []
        if self._bm25 is not None:
            tok = query.lower().split()
            s = self._bm25.get_scores(tok)
            for i, sc in enumerate(s): scores.append((float(sc), ("bm25", i)))
        qv = self._emb.encode([query], convert_to_numpy=True, normalize_embeddings=self.normalize)
        if self._faiss is not None:
            import faiss
            q = qv.astype('float32'); 
            if self.normalize: faiss.normalize_L2(q)
            D, I = self._faiss.search(q, min(k, len(self.records)))
            for j, i in enumerate(I[0]): scores.append((1.0 - float(D[0][j]), ("emb", int(i))))
        else:
            d, idx = self._nn.kneighbors(qv, n_neighbors=min(k, len(self.records)))
            for j, i in enumerate(idx[0]): scores.append((1.0 - float(d[0][j]), ("emb", int(i))))
        scores.sort(key=lambda x: x[0], reverse=True)
        out, seen = [], set()
        for sc, (_, i) in scores:
            if i in seen: continue
            out.append((sc, self.records[i])); seen.add(i)
            if len(out) >= k: break
        return out

# -------------------- multi-regime legal grounds & style bank --------------------
REGIMES = {
    "gdpr": {
        "label": "GDPR Article 17",
        "deadline": ["within one month","within 30 days","as soon as reasonably possible"],
        "grounds": {
            "gdpr-no-longer-necessary": [
                "because the data is no longer necessary for its original purpose.",
                "as this information is no longer needed.",
                "because the original purpose for processing no longer applies."
            ],
            "gdpr-withdraw-consent": [
                "since I withdraw consent to its processing.",
                "as I no longer consent to processing this information.",
                "because I have withdrawn my consent."
            ],
            "gdpr-objection": [
                "since I object to the processing and ask that it cease.",
                "as my objection to processing should be honored.",
                "because I have formally objected to this processing."
            ],
            "gdpr-unlawful": [
                "because I believe the processing is unlawful.",
                "as the data appears to have been processed unlawfully.",
                "because processing seems to contravene applicable law."
            ],
            "gdpr-legal-obligation": [
                "because erasure is legally required.",
                "as retention is not permitted by law.",
                "because there is a legal obligation to erase it."
            ],
            "gdpr-children": [
                "since this relates to data collected from a minor.",
                "as this concerns a child’s data.",
                "because a child’s data is involved."
            ]
        },
        "weights": {
            "gdpr-no-longer-necessary": 0.45,
            "gdpr-withdraw-consent":    0.25,
            "gdpr-objection":           0.15,
            "gdpr-unlawful":            0.07,
            "gdpr-legal-obligation":    0.06,
            "gdpr-children":            0.02
        }
    },
    "ccpa": {
        "label": "California CCPA/CPRA",
        "deadline": ["within 45 days","within forty-five (45) days","as soon as reasonably possible"],
        "grounds": {
            "ccpa-delete": [
                "as I am exercising my right to delete personal information under the CCPA/CPRA.",
                "because I request deletion of my personal information pursuant to the CCPA/CPRA."
            ],
            "ccpa-correct": [
                "as I request correction of inaccurate personal information under the CCPA/CPRA."
            ],
            "ccpa-limit-sensitive": [
                "as I request that you limit the use and disclosure of my sensitive personal information."
            ],
            "ccpa-opt-out-sale": [
                "as I am opting out of the sale or sharing of my personal information."
            ]
        },
        "weights": {"ccpa-delete": 0.70,"ccpa-correct": 0.10,"ccpa-limit-sensitive": 0.10,"ccpa-opt-out-sale": 0.10}
    },
    "pipeda": {
        "label": "Canada PIPEDA",
        "deadline": ["within 30 days","within a reasonable time","as soon as reasonably possible"],
        "grounds": {
            "pipeda-withdraw-consent": ["since I withdraw consent to your collection, use, or disclosure of this information."],
            "pipeda-no-longer-necessary": ["because the information is no longer necessary for the identified purposes."],
            "pipeda-correct": ["as I request correction of inaccurate or incomplete personal information under PIPEDA."]
        },
        "weights": {"pipeda-withdraw-consent": 0.45,"pipeda-no-longer-necessary": 0.40,"pipeda-correct": 0.15}
    },
    "lgpd": {
        "label": "Brazil LGPD",
        "deadline": ["within 15 days","as soon as reasonably possible"],
        "grounds": {
            "lgpd-delete-consent": ["as I request the elimination of personal data processed with my consent under the LGPD."],
            "lgpd-unnecessary": ["because the data is unnecessary, excessive, or processed in noncompliance with the LGPD."]
        },
        "weights": {"lgpd-delete-consent": 0.7,"lgpd-unnecessary": 0.3}
    },
    "pdpa": {
        "label": "Singapore PDPA",
        "deadline": ["within 30 days","within a reasonable time","as soon as reasonably possible"],
        "grounds": {
            "pdpa-withdraw-consent": ["since I withdraw consent for the purposes previously stated under the PDPA."],
            "pdpa-retention-limit": ["because the purpose for retention no longer applies and the data should be deleted or anonymized."],
            "pdpa-correct": ["as I request correction of inaccurate or incomplete personal data under the PDPA."]
        },
        "weights": {"pdpa-withdraw-consent": 0.5,"pdpa-retention-limit": 0.3,"pdpa-correct": 0.2}
    }
}
DEFAULT_REGIME_WEIGHTS = {"gdpr": 0.5, "ccpa": 0.25, "pipeda": 0.1, "lgpd": 0.1, "pdpa": 0.05}

TIME_LIMITS_GENERIC = ["within one month","within 30 days","within 45 days","as soon as reasonably possible","at your earliest convenience"]
CLOSINGS = ["Regards,","Many thanks,","Thank you,","Sincerely,","Best,"]
IDENTITY_SENTENCES = [
    "I can verify my identity via my account email if needed.",
    "I can confirm my account details upon request.",
    "I can provide additional verification if required."
]


TEMPLATES = [
    "Subject: Request for erasure\n\nTo whom it may concern,\n\n{lead} This concerns {anchor}. {tagline} I am requesting erasure {cause}\nPlease remove the related {scope_phrase} {deadline}\n{identity}\n\n{closing}",
    "{lead}\n\nSpecifically, this concerns {anchor}. {tagline} I'm asking for erasure {cause} Please remove the related {scope_phrase} {deadline}\n{identity}\n\n{closing}",
    "{lead} Regarding {anchor}. {tagline} Please erase the associated {scope_phrase}. {identity} {closing}",
    "Hi team — I’m submitting a data erasure request relating to {anchor}. {tagline} The basis is that {cause} Please remove the {scope_phrase} and let me know once done. {identity} {closing}",
    "Please process the following request. Context: {anchor}. {tagline} Ground: {cause} Action: remove {scope_phrase}. {identity} {closing}",
    "Hello, I recently reviewed {anchor}. {tagline} I’d like you to remove the {scope_phrase} {cause} {deadline} {identity} {closing}",
    "Erase request concerning {anchor}. {tagline} Basis: {cause} Action: remove {scope_phrase}. {identity} {closing}",
    "I need to request deletion. With respect to {anchor}, {tagline} I request erasure {cause} Please remove the {scope_phrase} {deadline} {identity} {closing}"
]

ABOUT_STEMS = ["it's about", "it’s about", "this is about"]  # used by throttle (case-insensitive)

# -------------------- sampling helpers --------------------
def sample_regime(enabled: List[str], uniform: bool=False) -> str:
    keys = [k for k in enabled if k in REGIMES] or list(REGIMES.keys())
    if uniform:
        return random.choice(keys)
    w = [DEFAULT_REGIME_WEIGHTS.get(k, 0.05) for k in keys]
    return random.choices(keys, weights=w, k=1)[0]

def sample_basis(regime: str, uniform: bool=False) -> str:
    g = REGIMES[regime]["grounds"]; w = REGIMES[regime]["weights"]
    keys = list(g.keys())
    if uniform:
        return random.choice(keys)
    weights = [w[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]

def pick_deadline(regime: str) -> str:
    opts = REGIMES[regime]["deadline"]
    return random.choice(opts) if random.random() < 0.75 else ""

def regime_tagline(regime: str) -> str:
    return f"(pursuant to {REGIMES[regime]['label']})" if random.random() < 0.4 else ""

def cause_sentence(regime: str, basis: str) -> str:
    return random.choice(REGIMES[regime]["grounds"][basis])

def prefer_scope_for_basis(basis: str, default: str) -> str:
    if "correct" in basis: return "attribute"
    if "opt-out-sale" in basis: return "edge"
    return default

# -------------------- small humanization (typos) --------------------
def maybe_typos(text: str, prob: float=0.05) -> str:
    if prob <= 0.0 or random.random() > prob: return text
    def swap(word):
        if len(word)<5: return word
        i = random.randint(1, len(word)-2)
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    return " ".join(swap(w) if random.random()<0.08 else w for w in text.split())

# -------------------- anchor + scope helpers --------------------
def build_anchor(rec: Dict[str,Any]) -> str:
    if rec["type"] == "node":
        return rec["display"]
    d = rec["display"]
    if "review by" in d and "about" in d and "on" in d:
        m = re.search(r"review by (.+?) about (.+?) on (\d{4}-\d{2}-\d{2})", d)
        if m:
            reviewer, biz, date = m.groups()
            return f"my review of {biz} from {date}"
    return d

def scope_phrase_for(rec: Dict[str,Any], scope: str) -> str:
    txt = (rec.get("display","") + " " + rec.get("text","")).lower()
    if scope == "edge":
        if "review" in txt: return random.choice(["review","review content","link between my account and that review"])
        if "citation" in txt: return random.choice(["citation link","reference","link between the two publications"])
        return "link/connection"
    if scope == "attribute":
        if "user" in txt or "profile" in txt: return random.choice(["profile field","bio","location field"])
        if "paper" in txt or "publication" in txt: return random.choice(["metadata field","keyword field","author field"])
        return random.choice(["metadata field","profile field"])
    if scope == "label": return random.choice(["category","topic tag","label"])
    if "paper" in txt or "publication" in txt: return random.choice(["publication record","paper entry"])
    if "business" in txt: return random.choice(["business page entry"])
    if "user" in txt or "profile" in txt: return random.choice(["account","profile entry"])
    return "item"

# -------------------- generation (with optional paraphrase) --------------------
def generate_text(provider: str, model: str, max_tokens: int, temperature: float, top_p: float,
                  rec: Dict[str,Any], scope: str, regime: str, basis: str, reflection: bool,
                  typo_prob: float) -> Tuple[str,Dict[str,Any]]:
    lead_options = [
        "I’m writing to ask for deletion of certain information.",
        "I’d like to have specific data removed from my account.",
        "I’m submitting a data erasure request.",
        "Could you please help me remove some data associated with my account?",
        "I need to request deletion of some content."
    ]
    lead = random.choice(lead_options)
    anchor = build_anchor(rec)
    cause = cause_sentence(regime, basis)
    cause_cap = cause[0].upper() + cause[1:] if cause else ""
    scope = prefer_scope_for_basis(basis, scope)
    scope_phrase = scope_phrase_for(rec, scope)
    deadline = pick_deadline(regime)
    identity = random.choice(IDENTITY_SENTENCES) if random.random() < 0.85 else ""
    tagline = regime_tagline(regime)
    closing = random.choice(CLOSINGS)

    template = random.choice(TEMPLATES)
    body = template.format(lead=lead, anchor=anchor, cause=cause, cause_cap=cause_cap, scope_phrase=scope_phrase,
                           deadline=f"{deadline}." if deadline and not deadline.strip().endswith(('.', '!', '?')) else deadline,
                           identity=identity, closing=closing, tagline=tagline)

    # keep dataset names/IDs out of the text
    if re.search(r"\b(Cora|Citeseer|Pubmed|CoraFull|Yelp)\b|\bPaper\s*\d+\b", body, flags=re.IGNORECASE):
        template = random.choice(TEMPLATES)
        body = template.format(lead=random.choice(lead_options), anchor=anchor, cause=cause, cause_cap=cause_cap,
                               scope_phrase=scope_phrase, deadline=deadline, identity=identity, closing=closing, tagline=tagline)

    if provider in (None,"","none"):
        return maybe_typos(body, typo_prob), {"provider":"rule","regime":regime,"basis":basis}

    # Optional LLM paraphrase for extra variety
    try:
        if provider == "openai":
            from openai import OpenAI; import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = ("Rewrite the following request naturally without adding dataset names or IDs; keep meaning:\n\n"
                      f"{body}\n\nReturn only the rewritten text.")
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=temperature, max_tokens=max_tokens
            )
            text = resp.choices[0].message.content.strip()
            if reflection:
                r = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role":"user","content":"Ensure the request includes a legal basis and an identity note; add one concise sentence if missing.\n\n"+text}],
                    temperature=0.2, max_tokens=120
                )
                text = text + "\n\n" + r.choices[0].message.content.strip()
            return maybe_typos(text, typo_prob*0.6), {"provider":"openai","regime":regime,"basis":basis}
        elif provider == "ollama":
            import requests, os
            url = os.getenv("OLLAMA_URL","http://localhost:11434/api/generate")
            r = requests.post(url, json={"model": model or "llama3", "prompt": f"Paraphrase naturally (no dataset names/IDs):\n{body}\nReturn only the rewritten text.", "stream": False}, timeout=180)
            r.raise_for_status()
            text = r.json().get("response","").strip()
            return maybe_typos(text, typo_prob*0.6), {"provider":"ollama","regime":regime,"basis":basis}
        elif provider == "transformers":
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            tok = AutoTokenizer.from_pretrained(model or "gpt2")
            m = AutoModelForCausalLM.from_pretrained(model or "gpt2", device_map="auto", torch_dtype=torch.float16 if has_cuda() else None)
            gen = pipeline("text-generation", model=m, tokenizer=tok, device=0 if has_cuda() else -1)
            prompt = f"Paraphrase naturally (no dataset names/IDs):\n{body}\n\nReturn only the rewritten text."
            out = gen(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=top_p)
            text = out[0]["generated_text"][len(prompt):].strip()
            return maybe_typos(text, typo_prob*0.6), {"provider":"transformers","regime":regime,"basis":basis}
    except Exception as e:
        return maybe_typos(body, typo_prob), {"provider":"rule","regime":regime,"basis":basis,"error":str(e)}

# -------------------- AHL scoring + anti-overlap + about-throttle --------------------
def legal_form_score(text: str) -> float:
    t = text.lower()
    cues = ["no longer necessary","withdraw consent","objection","unlawful","legal obligation","child",
            "delete personal information","opt out","limit the use","correction","rectify","retention no longer"]
    ground = any(x in t for x in cues)
    identity = any(x in t for x in ["identity","email","account","verify"])
    deadline = any(x in t for x in ["one month","30 days","45 days","15 days","as soon as"])
    return (1.0 if ground else 0.0) + (0.5 if identity else 0.0) + (0.5 if deadline else 0.0)

def variety_score(text: str) -> float:
    toks = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if not toks: return 0.5
    unq = len(set(toks)); total = len(toks)
    bigrams = list(zip(toks, toks[1:])); unq_bg = len(set(bigrams)); total_bg = max(1,len(bigrams))
    sents = max(1, text.count(".")+text.count("!")+text.count("?"))
    v = 0.35*(unq/total) + 0.35*(unq_bg/total_bg) + 0.3*(min(sents,4)/4.0)
    return min(1.0, max(0.0, v))

def detector_proxy_score(text: str) -> float:
    try:
        import textstat
        flesch = textstat.flesch_reading_ease(text)
        sent_len = textstat.words_per_sentence(text)
        words = textstat.lexicon_count(text)
        s = 0.0
        if words >= 25: s += 0.25
        if 50 <= flesch <= 85: s += 0.35
        if 10 <= sent_len <= 30: s += 0.40
        return min(1.0, s)
    except Exception:
        return 0.6

def compute_ahl(text: str) -> Dict[str,float]:
    scores = {"legal": legal_form_score(text)/2.0, "variety": variety_score(text), "detector": detector_proxy_score(text)}
    w = {"legal":0.4,"variety":0.3,"detector":0.3}
    scores["ahl"] = sum(w[k]*scores[k] for k in w.keys())
    return scores

RECENT = deque(maxlen=32)
def char_overlap(a: str, b: str) -> float:
    A, B = set(a.lower()), set(b.lower())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# Track "about" frequency
ABOUT_STATS = {"used": 0, "total": 0}

def text_has_about(text: str) -> bool:
    t = text.lower()
    return any(stem in t for stem in ABOUT_STEMS)

def pick_best_candidate(cands: list, about_max_freq: float) -> Tuple[str, Dict[str,float]]:
    best, best_score, best_ahl = None, -1e9, None
    current_ratio = (ABOUT_STATS["used"] / max(1, ABOUT_STATS["total"])) if ABOUT_STATS["total"] else 0.0
    for txt, ahl in cands:
        v = ahl.get("variety", 0.5)
        penalty = 0.0
        # anti-overlap
        for prev in RECENT:
            penalty = max(penalty, char_overlap(txt, prev))
        score = v - 0.4*penalty + 0.2*ahl.get("detector", 0.5) + 0.2*ahl.get("legal", 0.5)
        # about throttle
        if about_max_freq >= 0.0 and text_has_about(txt) and current_ratio > about_max_freq:
            score -= 0.25  # discourage once we've exceeded target share
        if score > best_score:
            best, best_score, best_ahl = txt, score, ahl
    RECENT.append(best)
    ABOUT_STATS["total"] += 1
    if text_has_about(best): ABOUT_STATS["used"] += 1
    return best, best_ahl

# -------------------- (optional) hybrid resolver; direct resolver stays fastest --------------------
def extract_mentions(text: str) -> List[str]:
    quoted = re.findall(r"“([^”]+)”|\"([^\"]+)\"", text)
    mentions = [q[0] or q[1] for q in quoted if (q[0] or q[1])]
    caps = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,4})\b", text)
    m2 = re.findall(r"review of ([A-Z][^\n,]+)", text, flags=re.IGNORECASE)
    for m in caps + m2:
        s = m.strip()
        if len(s) >= 3: mentions.append(s)
    uniq = []
    for m in mentions:
        if m not in uniq: uniq.append(m)
    return uniq[:5]

def resolve_targets(hi: "HybridIndex", text: str, scope_hint: str) -> List[Dict[str,Any]]:
    mentions = extract_mentions(text) or [text]
    hits = []
    for q in mentions:
        for sc, rec in hi.search(q, k=3):
            hits.append(rec)
    preferred = []
    for rec in hits:
        if scope_hint == "edge" and rec["type"] == "edge":
            preferred.append(rec)
        elif scope_hint in ("node","attribute","label") and rec["type"] == "node":
            preferred.append(rec)
    sel = preferred[0] if preferred else (hits[0] if hits else None)
    targets = []
    if sel:
        if sel["type"] == "edge":
            targets.append({"type":"edge","src":sel["src"],"dst":sel["dst"],"edge_type":sel.get("edge_type"),"review_id":sel.get("review_id")})
        else:
            if scope_hint == "attribute":
                targets.append({"type":"attribute","node_id":sel["node_id"],"attr_key":"profile"})
            elif scope_hint == "label":
                targets.append({"type":"label","node_id":sel["node_id"]})
            else:
                targets.append({"type":"node","node_id":sel["node_id"]})
    return targets

def direct_targets_from_rec(rec: Dict[str,Any], scope: str) -> List[Dict[str,Any]]:
    if rec["type"] == "edge":
        return [{"type":"edge","src":rec["src"],"dst":rec["dst"],"edge_type":rec.get("edge_type"),"review_id":rec.get("review_id","")}]
    else:
        if scope == "attribute": return [{"type":"attribute","node_id":rec["node_id"],"attr_key":"profile"}]
        if scope == "label":     return [{"type":"label","node_id":rec["node_id"]}]
        return [{"type":"node","node_id":rec["node_id"]}]

# -------------------- CLI & main --------------------
def main():
    ap = argparse.ArgumentParser(description="Request generator with variety-improved templates + CoraFull + Yelp speed knobs")
    ap.add_argument("--dataset", choices=["toy","cora","corafull","citeseer","pubmed","yelp"], default="toy")
    ap.add_argument("--yelp-dir", type=str, default=None, help="Folder with simplified Yelp JSONs or official JSON/JSONL")
    ap.add_argument("--out", type=str, default="./outputs_v5")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    # Yelp speed knobs
    ap.add_argument("--resolve-mode", choices=["hybrid","direct"], default="hybrid",
                    help="direct = NO index build; resolve from sampled record (recommended for Yelp)")
    ap.add_argument("--index-edges", type=int, default=1, help="0 = index nodes only (big Yelp speedup)")
    ap.add_argument("--cap-nodes", type=int, default=0)
    ap.add_argument("--cap-edges", type=int, default=0)
    ap.add_argument("--cap-seed", type=int, default=42)

    # Hybrid index tuning
    ap.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-faiss", action="store_true")

    # Variety / paraphrase / realism
    ap.add_argument("--alts", type=int, default=3)
    ap.add_argument("--provider", choices=["none","openai","ollama","transformers"], default="none")
    ap.add_argument("--gen-model", type=str, default="")
    ap.add_argument("--max-new-tokens", type=int, default=196)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--self-reflection", action="store_true")
    ap.add_argument("--no-typos", action="store_true", help="disable small typos")

    # Regime/basis controls for bias audit
    ap.add_argument("--uniform-regimes", action="store_true", help="sample regimes uniformly (ignore default weights)")
    ap.add_argument("--uniform-basis", action="store_true", help="sample legal bases uniformly within each regime")

    # About-stem throttle (0.35 = about ~35% max). Set -1 to disable.
    ap.add_argument("--about-max-freq", type=float, default=0.35)

    # regimes subset
    ap.add_argument("--regimes", type=str, default="gdpr,ccpa,pipeda,lgpd,pdpa")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Load graph
    if args.dataset == "toy":
        G, meta = load_toy(args.seed)
    elif args.dataset == "cora":
        G, meta = load_planetoid("cora", args.seed)
    elif args.dataset == "citeseer":
        G, meta = load_planetoid("citeseer", args.seed)
    elif args.dataset == "pubmed":
        G, meta = load_planetoid("pubmed", args.seed)
    elif args.dataset == "corafull":
        G, meta = load_corafull(args.seed)
    else:
        if not args.yelp_dir:
            raise SystemExit("--yelp-dir is required for dataset=yelp (same directory you’ll use in deletion)")
        G, meta = load_yelp(args.yelp_dir)
    print(f"[data] {meta} | CUDA: {has_cuda()}")

    # Build surface forms with edge control + caps
    rec_nodes, rec_edges = build_surface_forms_split(G)
    rng = random.Random(args.cap_seed)
    if args.cap_nodes > 0 and len(rec_nodes) > args.cap_nodes:
        rng.shuffle(rec_nodes); rec_nodes = rec_nodes[:args.cap_nodes]
    if args.dataset == "yelp" and args.index_edges == 1 and args.cap_edges > 0 and len(rec_edges) > args.cap_edges:
        rng.shuffle(rec_edges); rec_edges = rec_edges[:args.cap_edges]
    recs = rec_nodes + (rec_edges if args.index_edges == 1 else [])
    print(f"[index-plan] nodes={len(rec_nodes)}  edges={(len(rec_edges) if args.index_edges==1 else 0)}  total={len(recs)}")

    regimes_enabled = [r.strip() for r in args.regimes.split(",") if r.strip()]

    # Build index only for hybrid mode
    hi = None
    if args.resolve_mode == "hybrid":
        hi = HybridIndex(model=args.embed_model, use_bm25=not args.no_bm25,
                         normalize=True, batch_size=args.batch_size, use_faiss=not args.no_faiss)
        print("[index] building hybrid index ...")
        hi.fit(recs); print("[index] ready. records:", len(recs))
    else:
        print("[index] skipped (resolve-mode=direct)")

    # Generate requests with variety & anti-overlap & about throttle
    outputs = []
    drops = 0
    #typo_prob = 0.0 if args.no-typos else 0.05  # noqa: E999 (dash in var name not allowed, handled below)

    # fix typo_prob var (argument name cannot be attribute with dash)
    typo_prob = 0.0 if args.no_typos else 0.05

    for i in range(args.n):
        rec = random.choice(recs)
        regime = sample_regime(regimes_enabled, uniform=args.uniform_regimes)
        basis  = sample_basis(regime, uniform=args.uniform_basis)
        scope  = random.choice(["node","edge","attribute","label"])
        scope  = prefer_scope_for_basis(basis, scope)

        # sample alternatives then keep best
        cands = []
        meta_gen = None
        for _ in range(max(1, args.alts)):
            tx, mg = generate_text(args.provider, args.gen_model, args.max_new_tokens, args.temperature, args.top_p,
                                   rec, scope, regime, basis, args.self_reflection, typo_prob)
            cands.append((tx, compute_ahl(tx)))
            if meta_gen is None: meta_gen = mg
        text, ahl = pick_best_candidate(cands, about_max_freq=args.about_max_freq)

        # Resolve targets
        if args.resolve_mode == "direct":
            targets = direct_targets_from_rec(rec, scope)
        else:
            targets = resolve_targets(hi, text, scope) or direct_targets_from_rec(rec, scope)

        if not targets:
            drops += 1
            continue

        outputs.append({
            "request_id": str(uuid.uuid4()),
            "dataset": meta["name"],
            "language": "en",
            "text": text,
            "scope_hint": scope,
            "legal_regime": regime,
            "legal_basis_code": basis,
            "target_evidence": {"doc_id": rec["doc_id"], "display": rec["display"]},
            "ahl": ahl,
            "gen_meta": meta_gen,
            "resolved_targets": targets
        })

    req_path = os.path.join(args.out, f"requests_{meta['name']}.jsonl")
    sc_path  = os.path.join(args.out, f"scored_{meta['name']}.jsonl")
    pa_path  = os.path.join(args.out, f"parsed_{meta['name']}.jsonl")
    with open(req_path, "w", encoding="utf-8") as f:
        for ex in outputs: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    with open(sc_path, "w", encoding="utf-8") as f:
        for ex in outputs: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    with open(pa_path, "w", encoding="utf-8") as f:
        for ex in outputs: f.write(json.dumps(ex, ensure_ascii=False)+"\n")

    print(f"[gen] wrote {len(outputs)} requests (dropped {drops}) → {req_path}")
    c = Counter(x["scope_hint"] for x in outputs)
    about_ratio = (ABOUT_STATS['used']/max(1,ABOUT_STATS['total'])) if ABOUT_STATS['total'] else 0.0
    print("[harness] actions by scope:", dict(c))
    print(f"[harness] about-stem usage: {ABOUT_STATS['used']}/{ABOUT_STATS['total']} = {about_ratio:.2%} (target ≤ {args.about_max_freq:.0%} if enabled)")

if __name__ == "__main__":
    main()
