"""Train a PyKEEN model (RotatE/ComplEx) on the processed triples."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
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


def load_metadata(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def train_pykeen(
    split_name: str,
    model_choice: Literal["rotate", "complex"] = "rotate",
    device: str = "cuda",
    epochs: int = 30,
    batch_size: int = 1024,
) -> None:
    triples_path = Path("data/processed/triples.tsv")
    tf = TriplesFactory.from_path(triples_path, create_inverse_triples=False, delimiter="\t")
    params = DEFAULT_PARAMS[model_choice]
    artifact_dir = Path("artifacts/pykeen") / split_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Training %s on %s triples", params["model"], tf.num_triples)
    result = pipeline(
        training=tf,
        model=params["model"],
        optimizer="Adam",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=params["learning_rate"],
        device=device if torch.cuda.is_available() and "cuda" in device else "cpu",
    )
    result.save_to_directory(str(artifact_dir))
    entity_embeddings = result.model.entity_embeddings.weight.detach().cpu().numpy()
    relation_embeddings = result.model.relation_embeddings.weight.detach().cpu().numpy()
    np.save(artifact_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(artifact_dir / "relation_embeddings.npy", relation_embeddings)
    logging.info("Exported embeddings to %s", artifact_dir)
