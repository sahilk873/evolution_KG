# Evolutionary Subgraph Discovery on Hetionet (PoC)

This repository delivers the scaffolding for an evolutionary search over mechanistic subgraphs that explain `Compound - TREATS - Disease` predictions on Hetionet v1.0.

## Installation
1. Create a virtual environment (Python 3.10+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install from `pyproject.toml` or `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Reproducible run
The primary entry point is `scripts/run_experiments.py`. After preparing Hetionet (downloaded by the same script), run the default PoC pipeline with
```
python -m scripts.run_experiments \
  --split random \
  --method all \
  --model rotate \
  --device cuda \
  --budget 100 \
  --pop_size 80 \
  --generations 25 \
  --topk 5 \
  --seeds 3
```

## Expected outputs
- `data/raw/hetionet/` (downloaded Hetionet tables)
- `data/processed/triples.tsv`, `data/processed/entity2id.json`, `data/processed/relation2id.json`
- `data/splits/{random,disease_holdout}/{train,val,test}.jsonl`
- `artifacts/pykeen/{split}/` (models + embeddings)
- `artifacts/cand/{split}/{query_id}.npz`
- `artifacts/features/{split}/{query_id}/{subgraph_hash}.npy`
- `artifacts/evo/{split}/{query_id}/topk.json`
- `artifacts/preds/{split}/{method}.jsonl`
- `results/` tables, plots, and case studies per the AGENT instructions

## Switching modes
Adjust behavior with the `scripts/run_experiments.py` flags:
- `--split random|disease_holdout`
- `--method baseline|evolution|all`
- `--model rotate|complex`
- `--device cuda|cpu`
- `--budget` (edge budget for subgraphs)
- `--pop_size`, `--generations`, `--topk`, `--seeds`

Each flag is passed downstream to the matching scripts so you can rerun single components by reusing helper functions.

## Co-training mode (Level 1)
Level 1 co-training alternates baseline classifier updates with evolutionary subgraph search to better align the fitness model with evolved hypotheses. Enable it with `--cotrain_rounds` and tune how positives and negatives are sourced:
- `--train_subgraph_method {khop,random_walk,shortest_paths}` determines the baseline subgraphs in round 0.
- `--evo_train_topk` keeps the top-K evolved subgraphs per positive query for retraining.
- `--evo_negatives {khop,random_walk,none}` picks the baseline handling for negatives.
- `--mix_khop_frac` controls the fraction of training examples drawn from baseline subgraphs versus evolved ones.
- `--evo_train_max_cases` caps how many positives are evolved per round.

Example command:
```
python -m scripts.run_experiments \
  --split random \
  --method all \
  --model rotate \
  --device cuda \
  --budget 100 \
  --pop_size 80 \
  --generations 25 \
  --topk 5 \
  --seeds 3 \
  --cotrain_rounds 2 \
  --train_subgraph_method khop \
  --evo_train_topk 1 \
  --evo_negatives khop \
  --mix_khop_frac 0.5
```

## Citation
Hetionet v1.0: Himmelstein DS, et al. Sci Data 2017. Please cite Hetionet when publishing results that rely on it.
