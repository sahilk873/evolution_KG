"""Integration test covering the major pipeline phases."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from baselines import khop
from evolution.evolve import EvolutionConfig, evolve_query
from features.featurize import build_feature_vector
from kg.graph import TypedGraph
from models.classifier import Classifier
from retrieval.candidate_pool import CandidateBudget, build_candidate_pool

os.environ.setdefault("USE_PYKEEN_DATASET", "0")
from scripts.build_triples import build_triples
from scripts.make_splits import build_splits


def _create_sample_raw(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "hetionet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "sample.tsv", "w", encoding="utf-8") as handle:
        handle.write("subject\tpredicate\tobject\tsubject_category\tobject_category\n")
        for i in range(10):
            drug = f"Compound{i % 3}"
            disease = f"Disease{i % 4}"
            handle.write(f"{drug}\tTREATS\t{disease}\tCompound\tDisease\n")


def _prepare_embeddings(artifact_root: Path) -> None:
    artifact_dir = artifact_root / "artifacts" / "pykeen" / "test"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_map_path = artifact_root / "data" / "processed" / "entity2id.json"
    rel_map_path = artifact_root / "data" / "processed" / "relation2id.json"
    with open(entity_map_path, "r", encoding="utf-8") as handle:
        entity_map = json.load(handle)
    with open(rel_map_path, "r", encoding="utf-8") as handle:
        rel_map = json.load(handle)
    embedding_dim = 16
    np.save(artifact_dir / "entity_embeddings.npy", np.random.RandomState(0).rand(len(entity_map), embedding_dim))
    np.save(artifact_dir / "relation_embeddings.npy", np.random.RandomState(1).rand(len(rel_map), embedding_dim))


def test_pipeline_creates_artifacts(tmp_path: Path) -> None:
    root = tmp_path
    cwd = Path.cwd()
    try:
        os.chdir(root)
        _create_sample_raw(root)
        build_triples(root / "data" / "raw" / "hetionet", root / "data" / "processed")
        assert (root / "data" / "processed" / "triples.tsv").exists()
        assert (root / "data" / "processed" / "entity2id.json").exists()
        build_splits(root / "data" / "processed")
        assert (root / "data" / "splits" / "random" / "train.jsonl").exists()
        _prepare_embeddings(root)
        graph = TypedGraph.from_triples(root / "data" / "processed" / "triples.tsv")
        budgets = CandidateBudget()
        cp = build_candidate_pool(graph, "test", "Compound0", "Disease1", budgets)
        assert (root / "artifacts" / "cand" / "test" / "Compound0__Disease1.npz").exists()
        baseline = khop.explain("Compound0", "Disease1", cp, budgets.max_edges_total)
        vector = build_feature_vector("test", "Compound0__Disease1", "Compound0", "Disease1", cp, baseline)
        assert vector.ndim == 1
        classifier = Classifier(mode="logistic", device="cpu")
        X = np.vstack([vector, vector + 0.1])
        y = np.array([1, 0])
        classifier.fit(X, y)
        predictions = classifier.predict_proba(X)
        assert predictions.shape == (2,)
        config = EvolutionConfig(pop_size=8, generations=2, budget=5, topk=2)
        rng = random.Random(0)
        evolutions = evolve_query("Compound0", "Disease1", "test", cp, classifier, config, rng)
        assert len(evolutions) == 2
        base_path = root / "artifacts" / "evo" / "test" / "Compound0__Disease1"
        assert (base_path / "topk.json").exists()
        assert (base_path / "stats.json").exists()
        feature_dir = root / "artifacts" / "features" / "test" / "Compound0__Disease1"
        assert feature_dir.exists()
        assert any(feature_dir.glob("*.npy"))
    finally:
        os.chdir(cwd)
