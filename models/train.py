"""Train classifiers from baseline subgraphs and archive predictions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from baselines import full_embed, khop, random_walk, shortest_paths
from features.featurize import build_feature_vector
from kg.graph import TypedGraph
from models.classifier import Classifier
from retrieval.candidate_pool import CandidatePool, build_candidate_pool
from retrieval.constraints import CandidateBudget

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASELINE_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_baseline(name: str):
    def inner(func: Callable[..., Any]) -> Callable[..., Any]:
        BASELINE_REGISTRY[name] = func
        return func

    return inner


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_examples(
    split_name: str,
    subset: str,
    baseline: Callable[..., Any],
    budgets: CandidateBudget,
    graph: TypedGraph,
    artifact_tag: str,
    limit: int = 200,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    path = Path("data/splits") / split_name / f"{subset}.jsonl"
    rows = load_jsonl(path)[:limit]
    if not rows:
        return np.empty((0, 0)), np.empty((0,)), []
    features: List[np.ndarray] = []
    labels: List[int] = []
    cache: Dict[str, CandidatePool] = {}
    for row in rows:
        drug = row["drug_id"]
        disease = row["disease_id"]
        query_id = f"{drug}__{disease}"
        if query_id not in cache:
            cache[query_id] = build_candidate_pool(graph, split_name, drug, disease, budgets)
        pool = cache[query_id]
        subgraph = baseline(drug, disease, pool, budgets.max_edges_total)
        feat = build_feature_vector(artifact_tag, query_id, drug, disease, pool, subgraph)
        features.append(feat)
        labels.append(int(row.get("label", 1)))
    if not features:
        return np.empty((0, 0)), np.array(labels), rows
    return np.vstack(features), np.array(labels), rows


def train_classifier(
    split_name: str,
    method: str,
    device: str,
    model_type: str,
    budget: int,
    artifact_tag: str | None = None,
    limit: int = 200,
) -> Classifier:
    if method not in BASELINE_REGISTRY:
        raise ValueError("Unknown baseline %r" % method)
    baseline = BASELINE_REGISTRY[method]
    graph = TypedGraph.from_triples(Path("data/processed/triples.tsv"))
    budgets = CandidateBudget(max_edges_total=budget)
    artifact_tag = artifact_tag or split_name
    classifier = Classifier(mode=model_type, device=device)
    X_train, y_train, _ = build_examples(split_name, "train", baseline, budgets, graph, artifact_tag, limit)
    if X_train.size == 0:
        raise RuntimeError("No training examples")
    classifier.fit(X_train, y_train)
    for subset in ("train", "val", "test"):
        X_subset, _, rows = build_examples(split_name, subset, baseline, budgets, graph, artifact_tag, limit)
        if X_subset.size == 0:
            continue
        preds = classifier.predict_proba(X_subset)
        write_predictions(artifact_tag, method, subset, rows, preds)
    return classifier


def write_predictions(tag: str, method: str, subset: str, rows: List[Dict[str, Any]], preds: Sequence[float]) -> None:
    out_path = Path("artifacts/preds") / tag / f"{method}_{subset}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row, score in zip(rows, preds):
            record = dict(row)
            record["score"] = float(score)
            handle.write(json.dumps(record) + "\n")
    logging.info("Saved predictions to %s", out_path)


register_baseline("full_embed")(full_embed.explain)
register_baseline("khop")(khop.explain)
register_baseline("random_walk")(random_walk.explain)
register_baseline("shortest_paths")(shortest_paths.explain)
