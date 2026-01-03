import importlib
import json
import os
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

from retrieval.candidate_pool import CandidatePool

_mpl_cache = Path("tests/.mpl-cache")
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache.resolve()))

# pykeen imports torch in a few places, so provide a lightweight fake module before tests run.
if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.device = type("device", (), {})
    fake_torch.Generator = type("Generator", (), {})
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = type("Module", (), {})
    fake_torch.nn = fake_nn
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")
    fake_data.Dataset = type("Dataset", (), {})
    fake_data.DataLoader = type("DataLoader", (), {})
    fake_utils.data = fake_data
    fake_torch.utils = fake_utils
    def _stub_op(*args, **kwargs):
        return 0
    fake_torch.sum = fake_torch.max = fake_torch.mean = fake_torch.logsumexp = _stub_op
    fake_optim = types.ModuleType("torch.optim")
    fake_optimizer = types.ModuleType("torch.optim.optimizer")
    fake_optimizer.Optimizer = type("Optimizer", (), {})
    fake_optim.optimizer = fake_optimizer
    fake_torch.optim = fake_optim
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.utils"] = fake_utils
    sys.modules["torch.utils.data"] = fake_data
    sys.modules["torch.optim"] = fake_optim
    sys.modules["torch.optim.optimizer"] = fake_optimizer

fake_train_pykeen = types.ModuleType("embeddings.train_pykeen")
fake_train_pykeen.train_pykeen = lambda *args, **kwargs: None
sys.modules["embeddings.train_pykeen"] = fake_train_pykeen


@pytest.fixture(scope="module")
def runner():
    if "scripts.run_experiments" in sys.modules:
        del sys.modules["scripts.run_experiments"]
    return importlib.import_module("scripts.run_experiments")

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_ensure_directories(tmp_path, monkeypatch, runner):
    run_experiments = runner
    monkeypatch.chdir(tmp_path)
    run_experiments.ensure_directories()
    for expected in ("data/raw", "data/processed", "data/splits", "artifacts", "results"):
        assert (tmp_path / expected).is_dir()


def test_parse_args_defaults(monkeypatch, runner):
    run_experiments = runner
    monkeypatch.setattr(sys, "argv", ["run_experiments.py"])
    args = run_experiments.parse_args()
    assert args.split == "random"
    assert args.method == "all"
    assert args.model == "rotate"


def test_parse_args_custom(monkeypatch, runner):
    run_experiments = runner
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments.py",
            "--split",
            "disease_holdout",
            "--method",
            "baseline",
            "--model",
            "complex",
            "--pop_size",
            "5",
        ],
    )
    args = run_experiments.parse_args()
    assert args.split == "disease_holdout"
    assert args.method == "baseline"
    assert args.model == "complex"
    assert args.pop_size == 5


def test_load_predictions(tmp_path, runner):
    run_experiments = runner
    path = tmp_path / "preds.jsonl"
    assert run_experiments.load_predictions(path) == []
    write_jsonl(path, [{"label": 1}, {"label": 0}])
    preds = run_experiments.load_predictions(path)
    assert len(preds) == 2
    assert preds[0]["label"] == 1


def test_load_filtered_triples_filters_positive(tmp_path, monkeypatch, runner):
    run_experiments = runner
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    triples_path = processed / "triples.tsv"
    triples_path.write_text("head\trelation\ttail\nA\tTREATS\tD1\nB\tTREATS\tD2\nC\tOTHER\tD3\n")
    split_dir = tmp_path / "data/splits" / "random"
    val_rows = [{"drug_id": "A", "disease_id": "D1", "label": 1}]
    write_jsonl(split_dir / "val.jsonl", val_rows)
    filtered = run_experiments.load_filtered_triples(triples_path, "random", ("val", "test"))
    assert "D1" not in filtered["tail"].values
    assert len(filtered) == 2


def test_run_evolution_phase(monkeypatch, tmp_path, runner):
    run_experiments = runner
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        split="random",
        max_cases=2,
        budget=1,
        pop_size=1,
        generations=1,
        topk=2,
    )
    monkeypatch.setattr(run_experiments, "load_split_filtered_graph", lambda _: "graph")
    candidate_pool = CandidatePool(nodes=["A", "D1"], edges=[(0, 1, 1), (1, 2, 1)])
    monkeypatch.setattr(run_experiments, "build_candidate_pool", lambda *_: candidate_pool)
    monkeypatch.setattr(run_experiments, "load_jsonl", lambda _: [{"drug_id": "A", "disease_id": "D1"}])

    class FakeSubgraph:
        def __init__(self, edge_indices):
            self.edge_idx_set = frozenset(edge_indices)

    def fake_evolve(*_args, **_kwargs):
        return [FakeSubgraph({0}), FakeSubgraph({1})]

    monkeypatch.setattr(run_experiments, "evolve_query", fake_evolve)
    diversity_scores, reference_pool = run_experiments.run_evolution_phase(
        args, "random_seed0", classifier=object(), seed=0
    )
    assert reference_pool is candidate_pool
    assert diversity_scores == [1.0]


def test_write_metric_and_table_helpers(tmp_path, monkeypatch, runner):
    run_experiments = runner
    monkeypatch.chdir(tmp_path)
    run_experiments.write_metrics_table([{"split": "random", "seed": 0, "method": "full_embed", "roc_auc": 0.5, "pr_auc": 0.4}])
    contents = Path("results/tables/main_metrics.csv").read_text()
    assert "split,seed,method,roc_auc,pr_auc" in contents
    run_experiments.write_size_tradeoff([10, 20], "full_embed", 0.6, Path("results/tables/size_tradeoff.csv"))
    assert "full_embed,10,0.6,5" in Path("results/tables/size_tradeoff.csv").read_text()
    run_experiments.write_diversity_table([{"split": "random", "seed": 0, "avg_jaccard": 0.8}])
    assert "avg_jaccard" in Path("results/tables/diversity.csv").read_text()


def test_write_robustness_and_case_studies(tmp_path, monkeypatch, runner):
    run_experiments = runner
    monkeypatch.chdir(tmp_path)
    pool = CandidatePool(nodes=["A", "D"], edges=[(0, 1, 1), (1, 2, 1)])
    monkeypatch.setattr(run_experiments, "corrupt_candidate", lambda base, _: CandidatePool(base.nodes, base.edges[:1]))
    run_experiments.write_robustness_table(pool, [0.1, 0.2], Path("results/tables/robustness.csv"))
    assert "drop_prob,stability" in Path("results/tables/robustness.csv").read_text()
    artifacts_dir = tmp_path / "artifacts/evo/random_seed0/query"
    artifacts_dir.mkdir(parents=True)
    with open(artifacts_dir / "topk.json", "w", encoding="utf-8") as handle:
        json.dump([{"edges": [{"src": "A", "dst": "D", "rel_id": 1}], "rank": 1}], handle)
    run_experiments.write_case_studies("random_seed0")
    case_file = Path("results/case_studies/query.md")
    assert case_file.exists()
    assert "Top hypotheses" in case_file.read_text()
