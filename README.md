# Text2Forget: From Random Forget-Sets to Natural-Language Requests for Graph Unlearning

This repository contains the code and synthetic corpus used in our paper:

Text2Forget-GU provides a **DSAR-style, ID‑free request corpus** and a **request→target resolver** for graph unlearning on CoraFull, PubMed, and Yelp. It is designed to replace “delete random k% of nodes/edges” baselines with **realistic natural-language deletion requests** while remaining fully reproducible.

The repo also ships the **JSONL corpora** (or scripts to regenerate them) and all metadata needed to reproduce the benchmark.

---

## 2. Installation

We recommend Python 3.9+ and a conda environment with GPU support if available.

pip install torch torchvision torchaudio          # choose CUDA build if you have a GPU
pip install torch-geometric                       # follow PyG install instructions
pip install sentence-transformers rank-bm25
pip install scikit-learn textstat networkx

You need either:

The official Yelp Academic Dataset JSON files:

yelp_academic_dataset_user.json
yelp_academic_dataset_business.json
yelp_academic_dataset_review.json

or

Simplified JSON arrays:

users.json
businesses.json
reviews.json

Place them in the folder yelp_json

## 3. Running Code 

**Request Generation**
python request_gen.py --dataset corafull --n 3000 --seed 42 --out ./outputs_v5 --provider none

**Request Ranking**
python ranker.py --dataset corafull --in ./outputs_v5 --out ./ranked --topk 100

**Training**
python train_gnn.py --dataset corafull --gnn gcn --mode disabled --random_seed 42

**Unlearing**
python delete_gnn.py --dataset corafull --gnn gcv --del_mode node --parsed_file ./outputs_v5/parsed_corafull.jsonl --request_id 518cea2b-cc42-4474-ac3f-ba8d54bfcbaa --unlearning_model retrain --df_size 0 --mode disabled --viz


