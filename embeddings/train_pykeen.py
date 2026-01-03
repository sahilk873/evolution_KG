"""Train a PyKEEN model (RotatE/ComplEx) on the processed triples."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from torch.nn import ModuleList

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

TREATS_RELATION = "TREATS"


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _candidate_tensor(candidate: torch.nn.Module) -> torch.Tensor | None:
    weight = getattr(candidate, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    embeddings = getattr(candidate, "_embeddings", None)
    if embeddings is not None:
        emb_weight = getattr(embeddings, "weight", None)
        if isinstance(emb_weight, torch.Tensor):
            return emb_weight
    try:
        with torch.no_grad():
            tensor = candidate()
    except TypeError:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor
    return None


def _extract_embedding_weights(model: torch.nn.Module, attr_names: Sequence[str]) -> np.ndarray:
    def _iter_candidates(attribute: torch.nn.Module | Sequence[torch.nn.Module]) -> Sequence[torch.nn.Module]:
        if isinstance(attribute, ModuleList):
            return list(attribute)
        if isinstance(attribute, Sequence) and not isinstance(attribute, (str, bytes)):
            return list(attribute)
        return [attribute]

    for name in attr_names:
        repr_attribute = getattr(model, name, None)
        if repr_attribute is None:
            continue
        for candidate in _iter_candidates(repr_attribute):
            if candidate is None:
                continue
            tensor = _candidate_tensor(candidate)
            if tensor is None:
                continue
            return _tensor_to_numpy(tensor)
    raise AttributeError(f"Could not find embedding weights on model for {attr_names}")


def _load_excluded_pairs(split: str) -> set[tuple[str, str]]:
    excluded = set()
    split_dir = Path("data/splits") / split
    for subset in ("val", "test"):
        path = split_dir / f"{subset}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if int(row.get("label", 0)) == 1:
                excluded.add((row["drug_id"], row["disease_id"]))
    return excluded


def _load_split_positive_triples(split: str, subset: str, relation: str = TREATS_RELATION) -> list[tuple[str, str, str]]:
    path = Path("data/splits") / split / f"{subset}.jsonl"
    if not path.exists():
        return []
    triples: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("label", 0)) == 1:
                triples.append((row["drug_id"], relation, row["disease_id"]))
    return triples


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
            df["relation"].str.upper() == TREATS_RELATION
        ) | ~df.apply(lambda r: (r["head"], r["tail"]) in excluded_pairs, axis=1)
        df = df[mask].reset_index(drop=True)
    triples = df[["head", "relation", "tail"]].values
    tf = TriplesFactory.from_labeled_triples(triples)
    params = DEFAULT_PARAMS[model_choice]
    artifact_dir = Path("artifacts/pykeen") / split_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    testing_triples = _load_split_positive_triples(base_split, "test")
    if not testing_triples:
        raise ValueError(f"No positive test triples found for split '{base_split}'")
    testing_tf = TriplesFactory.from_labeled_triples(
        np.asarray(testing_triples, dtype=str)
    )
    validation_triples = _load_split_positive_triples(base_split, "val")
    validation_tf = (
        TriplesFactory.from_labeled_triples(
            np.asarray(validation_triples, dtype=str)
        )
        if validation_triples
        else None
    )

    logging.info("Training %s on %s triples", params["model"], tf.num_triples)
    result = pipeline(
        training=tf,
        testing=testing_tf,
        validation=validation_tf,
        model=params["model"],
        optimizer="Adam",
        epochs=epochs,
        optimizer_kwargs={"lr": params["learning_rate"]},
        training_kwargs={"batch_size": batch_size},
        device=device if torch.cuda.is_available() and "cuda" in device else "cpu",
    )
    result.save_to_directory(str(artifact_dir))
    entity_embeddings = _extract_embedding_weights(
        result.model,
        ("entity_embeddings", "entity_representations"),
    )
    relation_embeddings = _extract_embedding_weights(
        result.model,
        ("relation_embeddings", "relation_representations"),
    )
    np.save(artifact_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(artifact_dir / "relation_embeddings.npy", relation_embeddings)
    logging.info("Exported embeddings to %s", artifact_dir)
