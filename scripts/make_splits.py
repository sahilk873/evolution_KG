"""Build random and disease-holdout splits with typed negatives."""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NEGATIVE_RATIOS = {"train": 5, "val": 1, "test": 1}


def load_node_types(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_relation(df: pd.DataFrame, relation_name: str) -> pd.DataFrame:
    mask = df["relation"].str.upper() == relation_name.upper()
    return df[mask].reset_index(drop=True)


def sample_typed_negatives(
    positive_pairs: Sequence[Tuple[str, str]],
    compounds: Sequence[str],
    diseases: Sequence[str],
    ratio: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    positives = set(positive_pairs)
    target = ratio * len(positive_pairs)
    negatives = []
    attempts = 0
    max_attempts = target * 20 + 100
    while len(negatives) < target and attempts < max_attempts:
        drug = rng.choice(compounds)
        disease = rng.choice(diseases)
        if (drug, disease) in positives:
            attempts += 1
            continue
        negatives.append((drug, disease))
        attempts += 1
    return negatives


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_dataset_rows(split_name: str, split: str, positives: Sequence[Tuple[str, str]], negatives: Sequence[Tuple[str, str]]) -> List[dict]:
    rows: List[dict] = []
    rows.extend(
        {
            "drug_id": drug,
            "disease_id": disease,
            "label": 1,
            "split": split,
            "split_name": split_name,
        }
        for drug, disease in positives
    )
    rows.extend(
        {
            "drug_id": drug,
            "disease_id": disease,
            "label": 0,
            "split": split,
            "split_name": split_name,
        }
        for drug, disease in negatives
    )
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    return rows


def build_random_split(df: pd.DataFrame, node_types: Dict[str, str], rng: np.random.Generator) -> Dict[str, List[dict]]:
    positives = [(row["head"], row["tail"]) for _, row in df.iterrows()]
    rng.shuffle(positives)
    total = len(positives)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    partitions = {
        "train": positives[:train_end],
        "val": positives[train_end:val_end],
        "test": positives[val_end:],
    }
    compounds = [node for node, typ in node_types.items() if typ.upper().startswith("COMPO")]
    diseases = [node for node, typ in node_types.items() if typ.upper().startswith("DISEASE")]
    datasets: Dict[str, List[dict]] = {}
    for split, entries in partitions.items():
        ratio = NEGATIVE_RATIOS[split]
        negatives = sample_typed_negatives(entries, compounds, diseases, ratio, rng)
        datasets[split] = build_dataset_rows("random", split, entries, negatives)
    return datasets


def build_disease_holdout(df: pd.DataFrame, node_types: Dict[str, str], rng: np.random.Generator) -> Dict[str, List[dict]]:
    disease_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for _, row in df.iterrows():
        disease_index[row["tail"]].append((row["head"], row["tail"]))
    diseases = list(disease_index.keys())
    rng.shuffle(diseases)
    holdout_count = max(1, int(len(diseases) * 0.2))
    holdout = set(diseases[:holdout_count])
    train_val = []
    test = []
    for disease, edges in disease_index.items():
        if disease in holdout:
            test.extend(edges)
        else:
            train_val.extend(edges)
    split_point = int(len(train_val) * 0.1)
    splits = {
        "train": train_val[split_point:],
        "val": train_val[:split_point],
        "test": test,
    }
    compounds = [node for node, typ in node_types.items() if typ.upper().startswith("COMPO")]
    diseases_all = [node for node, typ in node_types.items() if typ.upper().startswith("DISEASE")]
    datasets: Dict[str, List[dict]] = {}
    for split, entries in splits.items():
        ratio = NEGATIVE_RATIOS[split]
        negatives = sample_typed_negatives(entries, compounds, diseases_all, ratio, rng)
        datasets[split] = build_dataset_rows("disease_holdout", split, entries, negatives)
    return datasets


def build_splits(processed_dir: Path, seed: int = 42) -> None:
    triples_path = processed_dir / "triples.tsv"
    node_types_path = processed_dir / "node_types.json"
    df = pd.read_csv(triples_path, sep="\t", dtype=str)
    treats = filter_relation(df, "TREATS")
    node_types = load_node_types(node_types_path)
    rng = np.random.default_rng(seed)
    random_split = build_random_split(treats, node_types, rng)
    disease_split = build_disease_holdout(treats, node_types, rng)
    for split, rows in random_split.items():
        write_jsonl(processed_dir.parent / "splits" / "random" / f"{split}.jsonl", rows)
    for split, rows in disease_split.items():
        write_jsonl(processed_dir.parent / "splits" / "disease_holdout" / f"{split}.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create KG splits")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_splits(args.processed_dir, args.seed)


if __name__ == "__main__":
    main()
