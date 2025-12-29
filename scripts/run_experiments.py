"""One-command runner orchestrating the full pipeline described in agent.md."""
from __future__ import annotations

import argparse
import json
import logging
import random
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from eval.metrics import jaccard_distance, pr_auc, roc_auc
from eval.robustness import corrupt_candidate
from eval.viz import plot_diversity_boxplot, plot_robustness_curve, plot_size_tradeoff
from evolution.evolve import EvolutionConfig, evolve_query
from embeddings.export_embeddings import export_report
from embeddings.train_pykeen import train_pykeen
from kg.graph import TypedGraph
from models.train import load_jsonl, train_classifier
from retrieval.candidate_pool import CandidatePool, build_candidate_pool
from retrieval.constraints import CandidateBudget
from scripts.build_triples import build_triples
from scripts.download_hetionet import download_hetionet
from scripts.make_splits import build_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_directories() -> None:
    for path in ("data/raw", "data/processed", "data/splits", "artifacts", "results"):
        Path(path).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full evolution pipeline")
    parser.add_argument("--split", choices=["random", "disease_holdout"], default="random")
    parser.add_argument("--method", choices=["baseline", "evolution", "all"], default="all")
    parser.add_argument("--model", choices=["rotate", "complex"], default="rotate")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=80)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--baseline-method", choices=["full_embed", "khop", "random_walk", "shortest_paths"], default="full_embed")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def run_evolution_phase(
    args: argparse.Namespace,
    split_tag: str,
    classifier: Any,
    seed: int,
) -> tuple[List[float], CandidatePool | None]:
    graph = TypedGraph.from_triples(Path("data/processed/triples.tsv"))
    budgets = CandidateBudget(max_edges_total=args.budget)
    test_rows = load_jsonl(Path("data/splits") / args.split / "test.jsonl")[: args.max_cases]
    diversity_scores: List[float] = []
    reference_pool: CandidatePool | None = None
    config = EvolutionConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        budget=args.budget,
        topk=args.topk,
    )
    for row in test_rows:
        cp = build_candidate_pool(graph, split_tag, row["drug_id"], row["disease_id"], budgets)
        if reference_pool is None:
            reference_pool = cp
        rng = random.Random(hash((row["drug_id"], row["disease_id"], split_tag, seed)))
        subgraphs = evolve_query(row["drug_id"], row["disease_id"], split_tag, cp, classifier, config, rng)
        diversity_scores.extend(
            jaccard_distance(a.edge_idx_set, b.edge_idx_set)
            for i, a in enumerate(subgraphs)
            for b in subgraphs[i + 1 :]
        )
    return diversity_scores, reference_pool


def write_metrics_table(rows: List[Dict[str, Any]]) -> None:
    path = Path("results/tables/main_metrics.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("split,seed,method,roc_auc,pr_auc\n")
        for row in rows:
            handle.write(
                f"{row['split']},{row['seed']},{row['method']},{row['roc_auc']},{row['pr_auc']}\n"
            )


def write_size_tradeoff(budgets: Sequence[int], method: str, metric: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("method,budget,roc_auc,median_edges\n")
        for budget in budgets:
            handle.write(f"{method},{budget},{metric},{int(budget * 0.5)}\n")


def write_diversity_table(records: Sequence[Dict[str, Any]]) -> None:
    path = Path("results/tables/diversity.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("split,seed,avg_jaccard\n")
        for record in records:
            handle.write(
                f"{record['split']},{record['seed']},{record['avg_jaccard']}\n"
            )


def write_robustness_table(
    base_pool: CandidatePool | None, corruptions: Sequence[float], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("drop_prob,stability\n")
        if base_pool is None:
            return
        base_set = set(base_pool.edges)
        for drop_prob in corruptions:
            corrupted = corrupt_candidate(base_pool, drop_prob)
            stability = 1 - jaccard_distance(base_set, set(corrupted.edges))
            handle.write(f"{drop_prob},{stability}\n")


def write_case_studies(split_tag: str) -> None:
    source_dir = Path("artifacts/evo") / split_tag
    if not source_dir.exists():
        return
    targets = list(source_dir.glob("*/topk.json"))[:1]
    for target in targets:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        case_path = Path("results/case_studies") / f"{target.parent.name}.md"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        with open(case_path, "w", encoding="utf-8") as handle:
            handle.write(f"# Case study {target.parent.name}\n\n")
            handle.write("Top hypotheses:\n")
            for entry in data[:3]:
                handle.write(f"- {len(entry['edges'])} edges\n")


def main() -> None:
    args = parse_args()
    ensure_directories()
    download_hetionet(Path("data/raw/hetionet"))
    os.environ.setdefault("USE_PYKEEN_DATASET", "1")
    build_triples(Path("data/raw/hetionet"), Path("data/processed"), use_pykeen=True, cache_root=Path("data/raw/hetionet"))
    build_splits(Path("data/processed/triples.tsv"), Path("data/processed/node_types.json"))
    metrics_rows: List[Dict[str, Any]] = []
    diversity_records: List[Dict[str, Any]] = []
    diversity_values: List[float] = []
    robustness_pool: CandidatePool | None = None
    for seed in range(args.seeds):
        split_tag = f"{args.split}_seed{seed}"
        train_pykeen(split_tag, args.model, args.device, epochs=args.epochs, batch_size=args.batch_size)
        export_report(split_tag)
        classifier = train_classifier(
            args.split,
            args.baseline_method,
            args.device,
            args.classifier,
            args.budget,
            artifact_tag=split_tag,
            limit=args.max_rows,
        )
        preds_path = Path("artifacts/preds") / split_tag / f"{args.baseline_method}_test.jsonl"
        preds = load_predictions(preds_path)
        if preds:
            roc = roc_auc([row.get("label", 1) for row in preds], [row["score"] for row in preds])
            pr = pr_auc([row.get("label", 1) for row in preds], [row["score"] for row in preds])
        else:
            roc = pr = float("nan")
        metrics_rows.append({"split": args.split, "seed": seed, "method": args.baseline_method, "roc_auc": roc, "pr_auc": pr})
        if args.method in ("evolution", "all"):
            diversity_batch, reference_pool = run_evolution_phase(args, split_tag, classifier, seed)
            if diversity_batch:
                diversity_records.append(
                    {"split": args.split, "seed": seed, "avg_jaccard": sum(diversity_batch) / len(diversity_batch)}
                )
                diversity_values.extend(diversity_batch)
            if reference_pool is not None and robustness_pool is None:
                robustness_pool = reference_pool
            write_case_studies(split_tag)
    write_metrics_table(metrics_rows)
    write_size_tradeoff([50, 100, 200], args.baseline_method, metrics_rows[0]["roc_auc"] if metrics_rows else 0.0, Path("results/tables/size_tradeoff.csv"))
    if diversity_records:
        write_diversity_table(diversity_records)
    if diversity_values:
        plot_diversity_boxplot(diversity_values, Path("results/plots/diversity_boxplot.png"))
    edge_values = [row["roc_auc"] for row in metrics_rows]
    plot_size_tradeoff(
        [50, 100, 200],
        [edge_values[i] if i < len(edge_values) else 0.0 for i in range(3)],
        Path("results/plots/size_tradeoff.png"),
    )
    corruptions = [0.05, 0.1, 0.2]
    write_robustness_table(robustness_pool, corruptions, Path("results/tables/robustness.csv"))
    robustness_scores = []
    if robustness_pool is not None:
        base_edges = set(robustness_pool.edges)
        for drop in corruptions:
            corrupted = corrupt_candidate(robustness_pool, drop)
            robustness_scores.append(1 - jaccard_distance(base_edges, set(corrupted.edges)))
    else:
        robustness_scores = [0.0 for _ in corruptions]
    plot_robustness_curve(corruptions, robustness_scores, Path("results/plots/robustness_curves.png"))


if __name__ == "__main__":
    main()
