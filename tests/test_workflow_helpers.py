import json
from pathlib import Path

import numpy as np
import pandas as pd

from kg.graph import TypedGraph
from retrieval.candidate_pool import build_candidate_pool
from retrieval.constraints import CandidateBudget
from scripts import build_triples, make_splits


def test_infer_type_recognizes_prefixes():
    assert build_triples._infer_type("Compound:123") == "COMPOUND"
    assert build_triples._infer_type("DISEASE::XYZ") == "DISEASE"
    assert build_triples._infer_type("FooBar") == "UNKNOWN"


def test_read_raw_triples_skips_invalid(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    good = raw_dir / "good.tsv"
    good.write_text("subject\tpredicate\tobject\nA\tTREATS\tB\n")
    bad = raw_dir / "bad.tsv"
    bad.write_text("foo\tbar\n1\t2\n")
    triples, node_types = build_triples._read_raw_triples(raw_dir)
    assert triples == [("A", "TREATS", "B")]
    assert node_types["A"] == "UNKNOWN" or isinstance(node_types["A"], str)


def test_build_triples_raw_writes_processed(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "sample.tsv"
    raw_file.write_text("subject\tpredicate\tobject\nC0001\tTREATS\tD0001\n")
    processed_dir = tmp_path / "processed"
    build_triples.build_triples(raw_dir, processed_dir, use_pykeen=False)
    triples_path = processed_dir / "triples.tsv"
    assert triples_path.exists()
    contents = pd.read_csv(triples_path, sep="\t", dtype=str)
    assert len(contents) == 1
    entity2id = json.loads((processed_dir / "entity2id.json").read_text())
    relation2id = json.loads((processed_dir / "relation2id.json").read_text())
    node_types = json.loads((processed_dir / "node_types.json").read_text())
    assert set(entity2id) == {"C0001", "D0001"}
    assert "TREATS" in relation2id
    assert "C0001" in node_types


def test_sample_typed_negatives_respects_ratios():
    positives = [("A", "D1"), ("B", "D2")]
    compounds = ["A", "B"]
    diseases = ["D1", "D2", "D3"]
    rng = np.random.default_rng(0)
    negatives = make_splits.sample_typed_negatives(positives, compounds, diseases, ratio=2, rng=rng)
    assert len(negatives) == 4
    assert all(pair not in positives for pair in negatives)


def test_build_random_split_counts():
    rows = [
        {"head": "A", "relation": "TREATS", "tail": "D1"},
        {"head": "B", "relation": "TREATS", "tail": "D2"},
        {"head": "C", "relation": "TREATS", "tail": "D3"},
        {"head": "D", "relation": "TREATS", "tail": "D4"},
        {"head": "E", "relation": "TREATS", "tail": "D5"},
    ]
    df = pd.DataFrame(rows)
    node_types = {f"C{i}": "COMPOUND" for i in range(1, 6)}
    node_types.update({f"D{i}": "DISEASE" for i in range(1, 6)})
    rng = np.random.default_rng(0)
    datasets = make_splits.build_random_split(df, node_types, rng)
    total_positives = len(rows)
    for split, records in datasets.items():
        positives = [r for r in records if r["label"] == 1]
        expected_negatives = make_splits.NEGATIVE_RATIOS[split] * len(positives)
        assert len(records) == len(positives) + expected_negatives
    assert sum(len(records) for records in datasets.values()) > total_positives


def test_typed_graph_neighbors():
    df = pd.DataFrame(
        [
            {"head": "A", "relation": "TREATS", "tail": "D"},
            {"head": "B", "relation": "TREATS", "tail": "D"},
        ]
    )
    graph = TypedGraph.from_triples(df)
    assert graph.neighbors("A") == [("TREATS", "D")]
    assert graph.neighbors("D", direction="in") == [("TREATS", "A"), ("TREATS", "B")]
    sampled = graph.sample_bounded_neighbors("D", direction="in", cap=1)
    assert len(sampled) == 1


def test_candidate_pool_builds_and_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame(
        [
            {"head": "DrugA", "relation": "TREATS", "tail": "DiseaseX", "head_type": "COMPOUND", "tail_type": "DISEASE"},
            {"head": "DrugA", "relation": "REL", "tail": "GeneY", "head_type": "COMPOUND", "tail_type": "GENE"},
        ]
    )
    graph = TypedGraph.from_triples(df)
    budgets = CandidateBudget(max_edges_total=4, hops=1)
    pool = build_candidate_pool(graph, "test_split", "DrugA", "DiseaseX", budgets, seed=0)
    assert "DrugA" in pool.nodes
    cache_file = Path("artifacts/cand/test_split/DrugA__DiseaseX.npz")
    assert cache_file.exists()
    second_pool = build_candidate_pool(graph, "test_split", "DrugA", "DiseaseX", budgets, seed=0)
    assert second_pool.nodes == pool.nodes
