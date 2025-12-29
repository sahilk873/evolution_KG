```md
# agent.md — Evolutionary Subgraph Discovery on Hetionet (Proof of Concept, GPU-enabled)

## Goal
Build a small-scale, end-to-end research prototype that:
1) Predicts **Compound → treats → Disease** links using a KG + downstream classifier.
2) Produces **top-K diverse, compact mechanistic subgraphs** (hypotheses) per (drug, disease) query via **evolutionary search**.
3) Demonstrates new capability vs baselines: diversity/compactness/robustness + competitive predictive performance.

Primary output: a `results/` folder containing paper-style tables, plots, and case studies.

---

## Core Deliverables
- `scripts/download_hetionet.py`: download Hetionet edge/node files into `data/raw/`
- `scripts/build_triples.py`: parse raw data → canonical triples `data/processed/triples.tsv`
- `scripts/make_splits.py`: random edge split + disease-holdout split with typed negatives
- `kg/graph.py`: typed adjacency graph with budgets/caps
- `embeddings/train_pykeen.py`: GPU training of RotatE/ComplEx; export entity embeddings
- `retrieval/candidate_pool.py`: bounded per-query candidate pool builder + caching
- `evolution/evolve.py`: evolutionary subgraph search; outputs top-K subgraphs per query
- `features/featurize.py`: fast, cached subgraph featurization using pretrained embeddings
- `models/classifier.py`: logistic regression baseline + small GPU MLP option
- `baselines/*.py`: full-embedding, k-hop, random-walk, shortest-path baselines
- `eval/metrics.py`: ROC-AUC, PR-AUC, size, diversity, robustness
- `scripts/run_experiments.py`: one command to run pipeline and create `results/`

---

## Repo Structure
```

.
├── agent.md
├── README.md
├── pyproject.toml (or requirements.txt)
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── artifacts/
│   ├── pykeen/
│   ├── cand/
│   ├── features/
│   └── evo/
├── kg/
│   ├── graph.py
│   └── schema.py
├── embeddings/
│   ├── train_pykeen.py
│   └── export_embeddings.py
├── retrieval/
│   ├── candidate_pool.py
│   └── constraints.py
├── evolution/
│   ├── representation.py
│   ├── operators.py
│   └── evolve.py
├── features/
│   ├── featurize.py
│   └── path_features.py
├── baselines/
│   ├── full_embed.py
│   ├── khop.py
│   ├── random_walk.py
│   └── shortest_paths.py
├── models/
│   ├── classifier.py
│   └── train.py
├── eval/
│   ├── metrics.py
│   ├── robustness.py
│   └── viz.py
└── scripts/
├── download_hetionet.py
├── build_triples.py
├── make_splits.py
├── run_experiments.py
└── make_case_studies.py

```

---

## Environment (GPU-enabled)
Python 3.10+.

Dependencies:
- numpy, pandas
- scikit-learn
- networkx (ONLY for shortest paths within candidate pools)
- torch (GPU)
- pykeen
- matplotlib
- tqdm
- joblib

GPU usage:
- PyKEEN training on GPU (RotatE default)
- optional MLP classifier on GPU (torch)

---

## Data Source (Hetionet v1.0)
Use Hetionet v1.0 nodes/edges (Compound–treats–Disease exists in the Hetionet schema). Download must be scripted and reproducible.

Raw files go to:
- `data/raw/hetionet/`

Processed outputs:
- `data/processed/triples.tsv` with columns:
  - `head`, `relation`, `tail`, `head_type`, `tail_type`
- `data/processed/entity2id.json`, `data/processed/relation2id.json`
- `data/processed/node_types.json` (optional)

Implementation note:
- Hetionet may be distributed as multiple edge tables; merge them.
- Normalize IDs as strings; preserve prefixes if present.

---

## Task Definition: Drug–Disease Link Prediction
Predict whether a (Compound, TREATS, Disease) edge holds.

- Positives: all existing `(Compound, TREATS, Disease)` edges in the training KG.
- Negatives: sampled `(Compound, Disease)` pairs not present in positives (typed sampling only).

Default negative ratios:
- train: 1:5 (pos:neg)
- val/test: 1:1

Two splits:
1) Random edge split: 80/10/10 on positive TREATS edges
2) Disease-holdout: hold out 20% of diseases for test (all their TREATS edges → test)

Store splits (JSONL):
- `data/splits/random/{train,val,test}.jsonl`
- `data/splits/disease_holdout/{train,val,test}.jsonl`

Each row:
- `drug_id`, `disease_id`, `label`, `split`, `split_name`

---

## Step-by-Step Build Plan

### Phase 1 — Download + Parse Hetionet
1) `scripts/download_hetionet.py`
   - downloads required files into `data/raw/hetionet/`
   - idempotent: does nothing if files already exist
   - prints what it downloaded

2) `scripts/build_triples.py`
   - parse raw edge/node data → canonical triple list
   - emit:
     - `data/processed/triples.tsv`
     - `entity2id.json`, `relation2id.json`
     - `node_types.json`
   - validate:
     - no null head/tail
     - relation vocabulary is stable
     - node types exist for compounds/diseases

3) `kg/graph.py`
   - load `triples.tsv`
   - build adjacency maps:
     - `out_adj[node] -> list[(rel, nbr)]`
     - `in_adj[node]  -> list[(rel, nbr)]`
   - build typed indices:
     - `nodes_by_type[type] -> list[node]`
   - support degree caps when retrieving neighbors.

---

### Phase 2 — Splits + Typed Negatives
4) `scripts/make_splits.py`
   - extract all TREATS edges as positives
   - build:
     - random edge split
     - disease-holdout split
   - generate negatives with typed sampling:
     - sample compound from Compound nodes
     - sample disease from Disease nodes
     - reject if (compound, TREATS, disease) is a known positive in any split
   - save JSONL splits

Sanity checks:
- no overlap of positive edges between train/val/test
- disease-holdout: test diseases not in train/val
- consistent negative sampling ratios

---

### Phase 3 — Pretrain Global Embeddings (PyKEEN on GPU)
5) `embeddings/train_pykeen.py`
   - input: training KG triples (exclude val/test TREATS positives from KG during training)
   - model: RotatE default (ComplEx optional)
   - GPU enabled: `device=cuda`
   - default hyperparams (small-scale, fast):
     - embed_dim: 200
     - epochs: 30
     - batch_size: 1024
     - learning_rate: 1e-3
     - negative_sampler: default
   - output:
     - `artifacts/pykeen/{split_name}/model.pt`
     - `artifacts/pykeen/{split_name}/entity_embeddings.npy` (aligned to entity2id)
     - `artifacts/pykeen/{split_name}/relation_embeddings.npy`

6) `embeddings/export_embeddings.py`
   - verifies entity embedding array aligns with entity2id length
   - writes a small report: embedding norms, shape, dtype

---

### Phase 4 — Candidate Pool Subgraph (bounded and cached)
7) `retrieval/candidate_pool.py`
   For each query (drug d, disease y):
   - build candidate graph `G_cand(d,y)` as union of:
     - 2-hop neighborhood around d
     - 2-hop neighborhood around y
   - enforce budgets:
     - `max_neighbors_per_node`: 200
     - `max_edges_total`: 10,000
     - `max_nodes_total`: 10,000 (hard stop)
   - optional: type-aware preference:
     - always include Gene/Pathway/Anatomy-related neighbors if available
   - cache:
     - `artifacts/cand/{split_name}/{query_id}.npz` with:
       - `nodes` (list of node IDs)
       - `edges` (arrays: src_idx, rel_id, dst_idx)

Note:
- Candidate pools are built from the TRAIN KG adjacency only (no test leakage).
- Candidate building must be deterministic given a seed.

---

### Phase 5 — Subgraph Representation + Fast Features
8) `evolution/representation.py`
   - an individual subgraph S is a set of edge indices into `G_cand`
   - invariant: `|E(S)| <= B` (default B=100)
   - store:
     - `edge_idx_set`
     - derived `node_idx_set` (cached)
     - `hash()` stable for caching features

9) `features/featurize.py`
   Given (d,y,S), pretrained entity embeddings `E`, plus candidate pool nodes mapping:
   - Pair features:
     - `[E_d, E_y, E_d ⊙ E_y, |E_d - E_y|]`
   - Subgraph pooled node features (over nodes in S):
     - mean(E_nodes), max(E_nodes)
   - Relation histogram (within S):
     - counts over relation IDs (fixed-length = num_relations OR top-R relations + bucket)
   - Node-type histogram (within S):
     - counts over known node types (fixed-length)
   Return fixed-length vector `x(d,y,S)`.

Caching:
- `artifacts/features/{split_name}/{query_id}/{subgraph_hash}.npy`

---

### Phase 6 — Downstream Classifier (GPU optional)
10) `models/classifier.py`
   - Baseline: sklearn LogisticRegression (fast, interpretable)
   - Optional: torch MLP (GPU):
     - 2 layers, hidden=256, dropout=0.1, ReLU
     - BCEWithLogitsLoss

11) `models/train.py`
   - trains classifier on training examples using a chosen subgraph method:
     - baseline methods: full_embed, khop, random_walk, shortest_paths
   - outputs:
     - trained model
     - val metrics
     - test predictions
   - store predictions:
     - `artifacts/preds/{split_name}/{method}.jsonl`

Important:
- Keep the classifier fixed during evolutionary search (for PoC).
  (Later improvements can co-train, but PoC should remain simple.)

---

### Phase 7 — Evolutionary Subgraph Discovery (main contribution)
12) `evolution/operators.py`
   Mutation operators:
   - ADD_FRONTIER: add a random edge adjacent to current node set
   - DELETE: remove a random edge
   - SWAP: delete one edge, add one frontier edge
   - TYPE_AWARE_ADD: prefer adding edges that bring in Gene/Pathway nodes (if node types exist)

   Crossover:
   - union parents’ edge sets
   - prune to budget B using a heuristic:
     - keep edges on paths between d and y first (if any)
     - then keep edges with rare relation types
     - then fill randomly (seeded)

13) `evolution/evolve.py`
   For each query (d,y):
   - load `G_cand(d,y)`
   - initialize population (pop_size default 80):
     - 40% random-walk seeded (start from d and y)
     - 40% shortest-path seeded (within candidate pool; if no path, fallback to random)
     - 20% random edge samples (budgeted)
   - fitness evaluation uses classifier score:
     - `p = classifier_proba(x(d,y,S))`
     - `Fitness(S) = log(p + 1e-9) - λ * |E(S)|`
     - default λ = 0.001 (tunable)
   - evolution:
     - generations default 25
     - elitism: keep top 15%
     - mutation rate: 0.6
     - crossover rate: 0.3
     - random immigrants: 0.1
   - output top-K subgraphs (K=5):
     - `artifacts/evo/{split_name}/{query_id}/topk.json`

Also store per-generation stats:
- best fitness, median fitness, avg size

---

## Baselines (must implement)
All baselines must share:
- same pretrained embeddings
- same featurizer
- same classifier type (LR by default; MLP optional)

1) `baselines/full_embed.py`
   - no subgraph; use only pair features (+ optional degree/type stats)

2) `baselines/khop.py`
   - deterministic 2-hop union around d and y
   - prune to edge budget B with BFS ordering

3) `baselines/random_walk.py`
   - sample B edges via random walks within candidate pool

4) `baselines/shortest_paths.py`
   - extract up to P shortest paths between d and y within candidate pool (P=5)
   - if insufficient edges, fill with frontier edges deterministically

---

## Required Experiments (Publication-Ready PoC)
Run on both splits: `random` and `disease_holdout`.

### Experiment 1 — Predictive Performance
Metrics:
- ROC-AUC
- PR-AUC
Report mean ± std across 3 seeds.

Output:
- `results/tables/main_metrics.csv`

### Experiment 2 — Explanation Size vs Performance Tradeoff
Sweep edge budget:
- B ∈ {50, 100, 200}

For each method:
- AUC/AUPRC vs B
- median explanation edges/nodes

Output:
- `results/tables/size_tradeoff.csv`
- `results/plots/size_tradeoff.png`

### Experiment 3 — Mechanistic Diversity (Evolution)
For each query’s top-K hypotheses (K=5):
- average pairwise Jaccard distance (edge sets)
- unique gene count across top-K
- unique pathway count across top-K (if types present)

Output:
- `results/tables/diversity.csv`
- `results/plots/diversity_boxplot.png`

### Experiment 4 — Robustness to KG Noise
Corrupt candidate pool construction by randomly dropping edges from the TRAIN KG adjacency used to build candidates:
- p ∈ {0.05, 0.10, 0.20}

Re-run candidate building + evolution, evaluate:
- ΔROC-AUC, ΔPR-AUC vs clean KG
- explanation stability: Jaccard(top1_edges_clean, top1_edges_corrupt)

Output:
- `results/tables/robustness.csv`
- `results/plots/robustness_curves.png`

### Experiment 5 — Case Studies
Select 5–10 known positive drug–disease pairs in test.
For each:
- show top-1 and top-3 evolved subgraphs (edge list)
- show 1–2 shortest mechanistic paths inside each subgraph:
  - (node_id, relation, node_id, ...)
Write as markdown.

Output:
- `results/case_studies/{query_id}.md`

---

## Metrics Implementation Details
Implement in `eval/metrics.py`:
- ROC-AUC: `sklearn.metrics.roc_auc_score`
- PR-AUC: `sklearn.metrics.average_precision_score`
- Jaccard distance:
  - `1 - |A ∩ B| / |A ∪ B|`
- Explanation size:
  - edges = |E(S)|, nodes = |V(S)|
- Robustness stability:
  - Jaccard of top-1 edge set pre/post corruption
- Aggregate metrics by split and by seed.

Implement plotting in `eval/viz.py` using matplotlib only.

---

## One-Command Runner
`scripts/run_experiments.py` must support:

Required flags:
- `--split random|disease_holdout`
- `--method baseline|evolution|all`
- `--model rotate|complex`
- `--device cuda|cpu`
- `--budget 100`
- `--pop_size 80`
- `--generations 25`
- `--topk 5`
- `--seeds 3`

Pipeline:
1) download (if missing)
2) build triples (if missing)
3) make splits (if missing)
4) train embeddings (if missing)
5) build candidate pools (cached)
6) train classifier (baseline features)
7) run baselines and/or evolution
8) compute metrics + plots + case studies into `results/`

---

## Default Small-Scale Settings (GPU available)
To keep runtime reasonable while still meaningful:

Data subset (initial PoC):
- train positives: 5,000
- val positives: 500
- test positives: 1,000
- negatives: 1:5 train, 1:1 val/test

PyKEEN:
- model: RotatE
- embed_dim: 200
- epochs: 30
- batch_size: 1024
- lr: 1e-3
- device: cuda

Candidate pool:
- hops: 2
- max_neighbors_per_node: 200
- max_edges_total: 10,000

Evolution:
- pop_size: 80
- generations: 25
- budget B: 100
- topK: 5

---

## README Requirements
README must include:
- installation (pip/uv)
- single command to reproduce results (default config)
- expected output files and where to find them
- how to switch splits, budgets, and model type
- citation note for Hetionet

---

## Pitfalls to Avoid
- Do NOT train embeddings on test positives (leakage).
- Candidate pools must be built using only the training KG adjacency for the split.
- networkx should only run on candidate pools (never full KG).
- Enforce strict budgets everywhere; fail fast if exceeded.
- Handle missing paths gracefully (shortest-path baseline fallback).
- Ensure subgraph hashes are stable and cache-safe.

---

## Stretch Goals (Optional)
- Add MLP classifier trained on GPU and compare vs LR
- Add uncertainty via multiple evolution runs or KG corruption ensembles
- Add ECE + calibration plots

End.
```

