"""Train a PyKEEN model (RotatE/ComplEx) on the processed triples."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_PARAMS = {
    "rotate": {
        "model": "RotatE",
        "embedding_dim": 200,
        "learning_rate": 1e-3,
    },
    "complex": {
        "model": "ComplEx",
        "embedding_dim": 200,
        "learning_rate": 1e-3,
    },
}


def _load_excluded_pairs(split: str) -> set[tuple[str, str]]:
    excluded = set()
    split_dir = Path("data/splits") / split
    for subset in ("val", "test"):
        path = split_dir / f"{subset}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = pd.read_json(line, typ="series")
            if int(row.get("label", 0)) == 1:
                excluded.add((row["drug_id"], row["disease_id"]))
    return excluded


def train_pykeen(
    split_name: str,
    model_choice: Literal["rotate", "complex"] = "rotate",
    device: str = "cuda",
    epochs: int = 30,
    batch_size: int = 1024,
    data_split: str | None = None,
) -> None:
    triples_path = Path("data/processed/triples.tsv")
    df = pd.read_csv(triples_path, sep="\t", dtype=str)
    base_split = data_split or split_name
    excluded_pairs = _load_excluded_pairs(base_split)
    if excluded_pairs:
        mask = ~(
            df["relation"].str.upper() == "TREATS"
        ) | ~df.apply(lambda r: (r["head"], r["tail"]) in excluded_pairs, axis=1)
        df = df[mask].reset_index(drop=True)
    triples = df[["head", "relation", "tail"]].values
    tf = TriplesFactory.from_labeled_triples(triples)
    params = DEFAULT_PARAMS[model_choice]
    artifact_dir = Path("artifacts/pykeen") / split_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Training %s on %s triples", params["model"], tf.num_triples)
    result = pipeline(
        training=tf,
        model=params["model"],
        optimizer="Adam",
        epochs=epochs,
        optimizer_kwargs={"lr": params["learning_rate"]},
        training_kwargs={"batch_size": batch_size},
        device=device if torch.cuda.is_available() and "cuda" in device else "cpu",
    )
    result.save_to_directory(str(artifact_dir))
    entity_embeddings = result.model.entity_embeddings.weight.detach().cpu().numpy()
    relation_embeddings = result.model.relation_embeddings.weight.detach().cpu().numpy()
    np.save(artifact_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(artifact_dir / "relation_embeddings.npy", relation_embeddings)
    logging.info("Exported embeddings to %s", artifact_dir)
