"""One-command runner orchestrating the full pipeline described in agent.md."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from eval.metrics import jaccard_distance, pr_auc, roc_auc
from eval.robustness import corrupt_candidate
from eval.viz import plot_diversity_boxplot, plot_robustness_curve, plot_size_tradeoff
from evolution.evolve import EvolutionConfig, evolve_query
from embeddings.export_embeddings import export_report
from embeddings.train_pykeen import train_pykeen
from features.featurize import build_feature_vector, set_feature_cache_persistence
from kg.graph import TypedGraph
from models.classifier import Classifier
from models.train import BASELINE_REGISTRY, load_jsonl, train_classifier
from retrieval.candidate_pool import CandidatePool, build_candidate_pool, set_candidate_cache_persistence
from retrieval.constraints import CandidateBudget
from scripts.build_triples import build_triples
from scripts.download_hetionet import download_hetionet
from scripts.make_splits import build_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SIZE_TRADEOFF_BUDGETS = (50, 100, 200)


def ensure_directories(results_root: Path | None = None) -> None:
    root = Path(results_root) if results_root else Path("results")
    for path in ("data/raw", "data/processed", "data/splits", "artifacts"):
        Path(path).mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)


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
    parser.add_argument("--cotrain_rounds", type=int, default=0, help="Number of co-training rounds after the initial baseline round.")
    parser.add_argument(
        "--train_subgraph_method",
        choices=["khop", "random_walk", "shortest_paths"],
        default="khop",
        help="Baseline subgraph method used for initial classifier training and positive mix examples.",
    )
    parser.add_argument("--evo_train_topk", type=int, default=1, help="Number of evolved subgraphs per training pair used for retraining.")
    parser.add_argument(
        "--evo_negatives",
        choices=["khop", "random_walk", "none"],
        default="khop",
        help="Baseline method for training negatives during co-training; 'none' reuses the previous negative features.",
    )
    parser.add_argument(
        "--mix_khop_frac",
        type=float,
        default=0.5,
        help="Fraction of training examples drawn from baseline subgraphs versus evolved subgraphs.",
    )
    parser.add_argument(
        "--evo_train_max_cases",
        type=int,
        default=2000,
        help="Max number of training positives to evolve during each co-training round.",
    )
    parser.add_argument(
        "--skip-size-tradeoff",
        action="store_true",
        dest="skip_size_tradeoff",
        help="Skip the budget sweep that generates size tradeoff tables/plots.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        dest="skip_prep",
        help="Assume triples/splits already exist and skip download/build steps",
    )
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        dest="reuse_artifacts",
        help="Skip PyKEEN training and reuse cached embeddings/classifier artifacts (requires existing artifacts/predictions)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed passed to PyKEEN training for repeatability",
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Disable on-disk caching of per-subgraph feature vectors.",
    )
    parser.add_argument(
        "--no-candidate-cache",
        action="store_true",
        help="Disable on-disk caching of candidate pools.",
    )
    parser.add_argument(
        "--no-evo-artifacts",
        action="store_true",
        help="Skip writing evolution top-k/stats artifacts (evo/evo_train).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to store tables, plots, and other derived outputs",
    )
    return parser.parse_args()


def _get_candidate_pool_for_row(
    row: Dict[str, Any],
    graph: TypedGraph,
    split_tag: str,
    budgets: CandidateBudget,
    cache: Dict[str, CandidatePool],
) -> CandidatePool:
    drug = row["drug_id"]
    disease = row["disease_id"]
    query_id = f"{drug}__{disease}"
    if query_id not in cache:
        cache[query_id] = build_candidate_pool(graph, split_tag, drug, disease, budgets)
    return cache[query_id]


def _build_examples_for_rows(
    rows: Sequence[Dict[str, Any]],
    baseline_name: str,
    budgets: CandidateBudget,
    graph: TypedGraph,
    split_tag: str,
    candidate_cache: Dict[str, CandidatePool],
    limit: int | None = None,
    target_label: int | None = None,
) -> tuple[List[np.ndarray], List[int]]:
    if baseline_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline {baseline_name}")
    baseline = BASELINE_REGISTRY[baseline_name]
    features: List[np.ndarray] = []
    labels: List[int] = []
    processed = 0
    for row in rows:
        label = int(row.get("label", 1))
        if target_label is not None and label != target_label:
            continue
        pool = _get_candidate_pool_for_row(row, graph, split_tag, budgets, candidate_cache)
        subgraph = baseline(row["drug_id"], row["disease_id"], pool, budgets.max_edges_total)
        feat = build_feature_vector(split_tag, f"{row['drug_id']}__{row['disease_id']}", row["drug_id"], row["disease_id"], pool, subgraph)
        features.append(feat)
        labels.append(label)
        processed += 1
        if limit is not None and processed >= limit:
            break
    return features, labels


def _mix_positive_examples(
    baseline_feats: List[np.ndarray],
    evolved_feats: List[np.ndarray],
    mix_frac: float,
    rng: random.Random,
) -> List[np.ndarray]:
    if not evolved_feats:
        return baseline_feats
    if mix_frac <= 0:
        return evolved_feats
    if mix_frac >= 1:
        return baseline_feats + evolved_feats
    expected_baseline = int(len(evolved_feats) * (mix_frac / max(1.0 - mix_frac, 1e-6)))
    if expected_baseline < len(baseline_feats):
        expected_baseline = max(1, expected_baseline)
        baseline_feats = rng.sample(baseline_feats, expected_baseline)
    return baseline_feats + evolved_feats


def _serialize_training_subgraphs(
    candidate_pool: CandidatePool,
    subgraphs: List["SubgraphRepresentation"],
) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for rank, subgraph in enumerate(subgraphs, start=1):
        edges = []
        for idx in sorted(subgraph.edge_idx_set):
            if idx >= len(candidate_pool.edges):
                continue
            src, rel_id, dst = candidate_pool.edges[idx]
            edges.append(
                {
                    "src": candidate_pool.nodes[src],
                    "dst": candidate_pool.nodes[dst],
                    "rel_id": rel_id,
                    "edge_idx": idx,
                }
            )
        records.append({"rank": rank, "edges": edges, "size": len(subgraph.edge_idx_set)})
    return records


def _persist_training_topk(
    split_tag: str,
    seed: int,
    round_idx: int,
    query_id: str,
    candidate_pool: CandidatePool,
    subgraphs: List["SubgraphRepresentation"],
) -> None:
    dest = Path("artifacts") / "evo_train" / split_tag / f"seed{seed}" / f"round_{round_idx}" / query_id
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "topk.json", "w", encoding="utf-8") as handle:
        json.dump(_serialize_training_subgraphs(candidate_pool, subgraphs), handle, indent=2)


def _gather_evolved_examples(
    args: argparse.Namespace,
    graph: TypedGraph,
    split_tag: str,
    seed: int,
    round_idx: int,
    classifier: Classifier,
    budgets: CandidateBudget,
    positive_rows: Sequence[Dict[str, Any]],
    candidate_cache: Dict[str, CandidatePool],
) -> List[np.ndarray]:
    config = _build_evolution_config(args, args.budget)
    config.topk = args.evo_train_topk
    evolved: List[np.ndarray] = []
    limit = args.evo_train_max_cases
    for row in positive_rows[:limit]:
        cp = _get_candidate_pool_for_row(row, graph, split_tag, budgets, candidate_cache)
        rng = random.Random(hash((row["drug_id"], row["disease_id"], split_tag, seed, round_idx)))
        subgraphs = evolve_query(
            row["drug_id"],
            row["disease_id"],
            split_tag,
            cp,
            classifier,
            config,
            rng,
            persist_topk_file=False,
            persist_stats_file=False,
        )
        if not subgraphs:
            continue
        query_id = f"{row['drug_id']}__{row['disease_id']}"
        if not args.no_evo_artifacts:
            _persist_training_topk(split_tag, seed, round_idx, query_id, cp, subgraphs)
        for subgraph in subgraphs:
            evolved.append(
                build_feature_vector(split_tag, query_id, row["drug_id"], row["disease_id"], cp, subgraph)
            )
    return evolved


def _fit_cotrain_classifier(
    args: argparse.Namespace,
    split_tag: str,
    seed: int,
    round_idx: int,
    positive_baseline_feats: List[np.ndarray],
    evolved_feats: List[np.ndarray],
    negative_feats: List[np.ndarray],
) -> Classifier:
    mix_frac = min(max(args.mix_khop_frac, 0.0), 1.0)
    mix_rng = random.Random(hash((split_tag, seed, round_idx, "mix")))
    positives = _mix_positive_examples(positive_baseline_feats, evolved_feats, mix_frac, mix_rng)
    if not positives:
        raise RuntimeError("No positive examples available for co-training round")
    if not negative_feats:
        raise RuntimeError("No negative examples available for co-training round")
    labels = [1] * len(positives) + [0] * len(negative_feats)
    features = positives + negative_feats
    X = np.vstack(features)
    y = np.array(labels)
    classifier = Classifier(mode=args.classifier, device=args.device)
    classifier.fit(X, y)
    artifact_path = Path("artifacts") / "classifiers" / split_tag / f"cotrain_round{round_idx}_{args.train_subgraph_method}_{classifier.mode}"
    classifier.save(artifact_path, input_dim=X.shape[1])
    return classifier


def _record_metric(
    metrics: List[Dict[str, Any]],
    cotrain_metrics: List[Dict[str, Any]],
    split: str,
    seed: int,
    round_idx: int,
    method: str,
    mode: str,
    roc: float,
    pr: float,
) -> None:
    entry = {
        "split": split,
        "seed": seed,
        "round": round_idx,
        "method": method,
        "mode": mode,
        "roc_auc": roc,
        "pr_auc": pr,
    }
    metrics.append(entry)
    cotrain_metrics.append(entry.copy())


def _evaluate_round(
    args: argparse.Namespace,
    split_tag: str,
    seed: int,
    round_idx: int,
    classifier: Classifier,
    graph: TypedGraph,
    test_rows: Sequence[Dict[str, Any]],
    pool_budget: int,
    metrics_rows: List[Dict[str, Any]],
    cotrain_records: List[Dict[str, Any]],
    diversity_records: List[Dict[str, Any]],
    diversity_values: List[float],
) -> tuple[CandidatePool | None, bool]:
    reference_pool: CandidatePool | None = None
    ran_evolution = False
    if args.method in ("baseline", "all"):
        baseline_records = score_baseline_explanations(
            test_rows,
            graph,
            args.split,
            split_tag,
            classifier,
            args.baseline_method,
            args.budget,
            pool_budget,
        )
        roc, pr, _ = summarize_records(baseline_records)
        _record_metric(metrics_rows, cotrain_records, args.split, seed, round_idx, args.baseline_method, "baseline", roc, pr)
    if args.method in ("evolution", "all"):
        diversity_batch, reference_pool, evolution_records = run_evolution_phase(
            args,
            split_tag,
            classifier,
            seed,
            graph,
            test_rows,
            persist_artifacts=not args.no_evo_artifacts,
        )
        ran_evolution = bool(evolution_records)
        if ran_evolution:
            roc, pr, _ = summarize_records(evolution_records)
            _record_metric(metrics_rows, cotrain_records, args.split, seed, round_idx, "evolution", "evolution", roc, pr)
        if diversity_batch:
            avg_jaccard = sum(diversity_batch) / len(diversity_batch)
            diversity_records.append({"split": args.split, "seed": seed, "round": round_idx, "avg_jaccard": avg_jaccard})
            diversity_values.extend(diversity_batch)
    return reference_pool, ran_evolution


def classifier_artifact_path(split_tag: str, method: str, classifier_mode: str) -> Path:
    return Path("artifacts/classifiers") / split_tag / f"{method}_{classifier_mode}"


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def load_split_filtered_graph(split: str, excluded_subsets: Sequence[str] = ("val", "test")) -> TypedGraph:
    triples_path = Path("data/processed/triples.tsv")
    df = load_filtered_triples(triples_path, split, excluded_subsets)
    return TypedGraph.from_triples(df)


def load_filtered_triples(triples_path: Path, split: str, excluded_subsets: Sequence[str]) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(triples_path, sep="\t", dtype=str)
    split_dir = Path("data/splits") / split
    excluded_pairs = set()
    for subset in excluded_subsets:
        path = split_dir / f"{subset}.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if int(row.get("label", 0)) == 1:
                excluded_pairs.add((row["drug_id"], row["disease_id"]))
    if not excluded_pairs:
        return df
    mask = ~(
        df["relation"].str.upper() == "TREATS"
    ) | ~df.apply(lambda r: (r["head"], r["tail"]) in excluded_pairs, axis=1)
    return df[mask].reset_index(drop=True)


def run_evolution_phase(
    args: argparse.Namespace,
    split_tag: str,
    classifier: Any,
    seed: int,
    graph: TypedGraph | None = None,
    test_rows: Sequence[Dict[str, Any]] | None = None,
    *,
    persist_artifacts: bool = True,
) -> tuple[List[float], CandidatePool | None, List[Dict[str, Any]]]:
    graph = graph or load_split_filtered_graph(args.split)
    budgets = CandidateBudget(max_edges_total=args.budget)
    query_rows = test_rows if test_rows is not None else load_jsonl(Path("data/splits") / args.split / "test.jsonl")[: args.max_cases]
    diversity_scores: List[float] = []
    reference_pool: CandidatePool | None = None
    labeled_scores: List[Dict[str, Any]] = []
    config = EvolutionConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        budget=args.budget,
        topk=args.topk,
    )
    for row in query_rows:
        cp = build_candidate_pool(graph, split_tag, row["drug_id"], row["disease_id"], budgets)
        if reference_pool is None:
            reference_pool = cp
        rng = random.Random(hash((row["drug_id"], row["disease_id"], split_tag, seed)))
        subgraphs = evolve_query(
            row["drug_id"],
            row["disease_id"],
            split_tag,
            cp,
            classifier,
            config,
            rng,
            persist_topk_file=persist_artifacts,
            persist_stats_file=persist_artifacts,
        )
        diversity_scores.extend(
            jaccard_distance(a.edge_idx_set, b.edge_idx_set)
            for i, a in enumerate(subgraphs)
            for b in subgraphs[i + 1 :]
        )
        if not subgraphs:
            continue
        top = subgraphs[0]
        query_id = f"{row['drug_id']}__{row['disease_id']}"
        features = build_feature_vector(split_tag, query_id, row["drug_id"], row["disease_id"], cp, top)
        score = classifier.predict_proba(features.reshape(1, -1))[0]
        labeled_scores.append(
            {"label": int(row.get("label", 1)), "score": float(score), "size": len(top.edge_idx_set)}
        )
    return diversity_scores, reference_pool, labeled_scores


def write_metrics_table(rows: List[Dict[str, Any]], results_root: Path | None = None) -> None:
    root = Path(results_root) if results_root else Path("results")
    path = root / "tables" / "main_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("split,seed,round,method,mode,roc_auc,pr_auc\n")
        for row in rows:
            handle.write(
                f"{row['split']},{row['seed']},{row['round']},{row['method']},{row['mode']},{row['roc_auc']},{row['pr_auc']}\n"
            )


def write_cotrain_metrics(rows: List[Dict[str, Any]], results_root: Path | None = None) -> None:
    root = Path(results_root) if results_root else Path("results")
    path = root / "tables" / "cotrain_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("split,seed,round,method,mode,roc_auc,pr_auc\n")
        for row in rows:
            handle.write(
                f"{row['split']},{row['seed']},{row['round']},{row['method']},{row['mode']},{row['roc_auc']},{row['pr_auc']}\n"
            )


def write_size_tradeoff(records: Sequence[Dict[str, Any]], results_root: Path | None = None) -> None:
    root = Path(results_root) if results_root else Path("results")
    path = root / "tables" / "size_tradeoff.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("split,seed,method,budget,roc_auc,pr_auc,median_edges\n")
        for record in records:
            handle.write(
                f"{record['split']},{record['seed']},{record['method']},{record['budget']},"
                f"{record['roc_auc']},{record['pr_auc']},{record['median_edges']}\n"
            )


def write_diversity_table(records: Sequence[Dict[str, Any]], results_root: Path | None = None) -> None:
    root = Path(results_root) if results_root else Path("results")
    path = root / "tables" / "diversity.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("split,seed,round,avg_jaccard\n")
        for record in records:
            handle.write(
                f"{record['split']},{record['seed']},{record['round']},{record['avg_jaccard']}\n"
            )


def write_robustness_table(
    method: str, base_pool: CandidatePool | None, corruptions: Sequence[float], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("method,drop_prob,stability\n")
        if base_pool is None:
            return
        base_set = set(base_pool.edges)
        for drop_prob in corruptions:
            corrupted = corrupt_candidate(base_pool, drop_prob)
            stability = 1 - jaccard_distance(base_set, set(corrupted.edges))
            handle.write(f"{method},{drop_prob},{stability}\n")


def _load_relation_lookup() -> Dict[int, str]:
    rel_path = Path("data/processed/relation2id.json")
    if not rel_path.exists():
        return {}
    with open(rel_path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    lookup: Dict[int, str] = {}
    for name, idx in mapping.items():
        try:
            lookup[int(idx)] = name
        except (TypeError, ValueError):
            continue
    return lookup


def _load_node_types() -> Dict[str, str]:
    types_path = Path("data/processed/node_types.json")
    if not types_path.exists():
        return {}
    with open(types_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_shortest_path(
    edges: Sequence[Dict[str, Any]],
    start: str,
    end: str,
    max_depth: int = 6,
) -> Sequence[Dict[str, Any]]:
    adjacency: Dict[str, List[Tuple[str, int, bool]]] = defaultdict(list)
    for edge in edges:
        src = edge["src"]
        dst = edge["dst"]
        rel_id = edge["rel_id"]
        adjacency[src].append((dst, rel_id, True))
        adjacency[dst].append((src, rel_id, False))
    queue: deque[tuple[str, List[Dict[str, Any]]]] = deque([(start, [])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        if len(path) >= max_depth:
            continue
        for neighbor, rel_id, forward in adjacency.get(node, []):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            segment = {"src": node, "dst": neighbor, "rel_id": rel_id, "forward": forward}
            queue.append((neighbor, path + [segment]))
    return []


def _format_path_string(path_segments: Sequence[Dict[str, Any]], relation_lookup: Dict[int, str]) -> str:
    if not path_segments:
        return ""
    tokens = []
    for seg in path_segments:
        rel_name = relation_lookup.get(seg["rel_id"], f"REL{seg['rel_id']}")
        if seg["forward"]:
            tokens.append(f"{seg['src']} --[{rel_name}]--> {seg['dst']}")
        else:
            tokens.append(f"{seg['dst']} --[{rel_name}]--> {seg['src']}")
    return " ⟶ ".join(tokens)


def _type_summary(type_counts: Counter[str]) -> str:
    if not type_counts:
        return "No typed nodes"
    parts = [f"{typ.title()}: {count}" for typ, count in type_counts.most_common(4)]
    return ", ".join(parts)


def write_case_studies(split_tag: str, limit: int, results_root: Path | None = None) -> None:
    source_dir = Path("artifacts/evo") / split_tag
    if not source_dir.exists():
        return
    relation_lookup = _load_relation_lookup()
    node_types = _load_node_types()
    query_dirs = [entry for entry in source_dir.iterdir() if entry.is_dir()]
    for query_dir in sorted(query_dirs)[:limit]:
        topk_path = query_dir / "topk.json"
        if not topk_path.exists():
            continue
        query_id = query_dir.name
        if "__" in query_id:
            drug, disease = query_id.split("__", 1)
        else:
            drug = query_id
            disease = "Unknown"
        with open(topk_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        root = Path(results_root) if results_root else Path("results")
        case_path = root / "case_studies" / f"{query_id}.md"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        with open(case_path, "w", encoding="utf-8") as handle:
            handle.write(f"# Case study {query_id}\n\n")
            handle.write(f"- **Compound**: {drug}\n")
            handle.write(f"- **Disease**: {disease}\n\n")
            handle.write("## Top hypotheses\n")
            for entry in data[:3]:
                size = entry.get("size", len(entry.get("edges", [])))
                handle.write(f"### Hypothesis {entry.get('rank', '?')} ({size} edges)\n")
                nodes: Set[str] = set()
                for edge in entry.get("edges", []):
                    nodes.add(edge["src"])
                    nodes.add(edge["dst"])
                type_counts = Counter(
                    node_types.get(node, "UNKNOWN").upper()
                    for node in nodes
                )
                genes = [node for node in nodes if node_types.get(node, "").upper() == "GENE"]
                pathways = [node for node in nodes if node_types.get(node, "").upper() == "PATHWAY"]
                side_effects = [
                    node for node in nodes if node_types.get(node, "").upper() == "SIDE EFFECT"
                ]
                handle.write(f"- **Type distribution**: {_type_summary(type_counts)}\n")
                if genes:
                    top_genes = ", ".join(genes[:3])
                    handle.write(f"- **Gene hits**: {len(genes)} (e.g., {top_genes})\n")
                if pathways:
                    handle.write(f"- **Pathways**: {len(pathways)} detected\n")
                if side_effects:
                    handle.write(f"- **Side effects**: {len(side_effects)} contextual nodes\n")
                path_segments = _find_shortest_path(entry.get("edges", []), drug, disease)
                if path_segments:
                    path_str = _format_path_string(path_segments, relation_lookup)
                    handle.write(f"- **Representative path**: `{path_str}`\n")
                else:
                    handle.write("- **Representative path**: Not found inside this subgraph\n")
                commentary: List[str] = []
                if genes:
                    commentary.append(
                        f"gene-level focus ({len(genes)} targets, ex. {', '.join(genes[:3])})"
                    )
                if pathways:
                    commentary.append(f"{len(pathways)} pathway nodes anchor biological processes")
                if side_effects:
                    commentary.append(f"{len(side_effects)} side-effect records provide safety context")
                if not commentary:
                    commentary.append("heterogeneous node mix emphasizes the KG neighborhood")
                handle.write(f"- **Biology commentary**: {'. '.join(commentary)}.\n")
                handle.write("- **Edges**:\n")
                for edge in entry.get("edges", []):
                    rel_name = relation_lookup.get(edge["rel_id"], f"REL{edge['rel_id']}")
                    handle.write(
                        f"  - `{edge['src']} --[{rel_name}]--> {edge['dst']}`\n"
                    )
                handle.write("\n")


def load_test_queries(split: str, limit: int) -> List[Dict[str, Any]]:
    rows = load_jsonl(Path("data/splits") / split / "test.jsonl")
    return rows[:limit]


def summarize_records(records: Sequence[Dict[str, Any]]) -> tuple[float, float, int]:
    if not records:
        return float("nan"), float("nan"), 0
    labels = [record["label"] for record in records]
    scores = [record["score"] for record in records]
    sizes = [record["size"] for record in records]
    roc = roc_auc(labels, scores)
    pr = pr_auc(labels, scores)
    median_size = int(statistics.median(sizes)) if sizes else 0
    return roc, pr, median_size


def score_baseline_explanations(
    rows: Sequence[Dict[str, Any]],
    graph: TypedGraph,
    split: str,
    split_tag: str,
    classifier: Classifier,
    method: str,
    budget: int,
    pool_budget: int,
) -> List[Dict[str, Any]]:
    baseline = BASELINE_REGISTRY.get(method)
    if baseline is None:
        raise ValueError(f"Unknown baseline {method}")
    budgets = CandidateBudget(max_edges_total=pool_budget)
    records: List[Dict[str, Any]] = []
    for row in rows:
        drug = row["drug_id"]
        disease = row["disease_id"]
        query_id = f"{drug}__{disease}"
        cp = build_candidate_pool(graph, split_tag, drug, disease, budgets)
        subgraph = baseline(drug, disease, cp, budget)
        features = build_feature_vector(split_tag, query_id, drug, disease, cp, subgraph)
        score = classifier.predict_proba(features.reshape(1, -1))[0]
        records.append({"label": int(row.get("label", 1)), "score": float(score), "size": len(subgraph.edge_idx_set)})
    return records


def _build_evolution_config(args: argparse.Namespace, budget: int) -> EvolutionConfig:
    return EvolutionConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        budget=budget,
        topk=args.topk,
    )


def score_evolution_explanations(
    rows: Sequence[Dict[str, Any]],
    graph: TypedGraph,
    split: str,
    split_tag: str,
    classifier: Classifier,
    args: argparse.Namespace,
    seed: int,
    budget: int,
    pool_budget: int,
    persist_artifacts: bool,
) -> List[Dict[str, Any]]:
    budgets = CandidateBudget(max_edges_total=pool_budget)
    config = _build_evolution_config(args, budget)
    records: List[Dict[str, Any]] = []
    for row in rows:
        drug = row["drug_id"]
        disease = row["disease_id"]
        query_id = f"{drug}__{disease}"
        cp = build_candidate_pool(graph, split_tag, drug, disease, budgets)
        rng = random.Random(hash((drug, disease, split_tag, seed, budget)))
        subgraphs = evolve_query(
            drug,
            disease,
            split_tag,
            cp,
            classifier,
            config,
            rng,
            persist_topk_file=persist_artifacts,
            persist_stats_file=persist_artifacts,
        )
        if not subgraphs:
            continue
        top = subgraphs[0]
        features = build_feature_vector(split_tag, query_id, drug, disease, cp, top)
        score = classifier.predict_proba(features.reshape(1, -1))[0]
        records.append({"label": int(row.get("label", 1)), "score": float(score), "size": len(top.edge_idx_set)})
    return records


def build_size_tradeoff_curves(records: Sequence[Dict[str, Any]]) -> Dict[str, List[tuple[int, float]]]:
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if math.isnan(record["roc_auc"]):
            continue
        grouped[record["method"]][record["budget"]].append(record["roc_auc"])
    curves: Dict[str, List[tuple[int, float]]] = {}
    for method, budgets in grouped.items():
        points = []
        for budget in sorted(budgets):
            scores = budgets[budget]
            if not scores:
                continue
            points.append((budget, statistics.mean(scores)))
        if points:
            curves[method] = points
    return curves


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_dir)
    set_feature_cache_persistence(not args.no_feature_cache)
    set_candidate_cache_persistence(not args.no_candidate_cache)
    ensure_directories(results_root)
    mpl_cache_dir = results_root / ".matplotlib_cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    if args.skip_prep:
        logging.info("Skipping download/triples/split prep (assumes files exist)")
        _ensure_prep(args.split)
    else:
        download_hetionet(Path("data/raw/hetionet"))
        os.environ.setdefault("USE_PYKEEN_DATASET", "1")
        try:
            build_triples(
                Path("data/raw/hetionet"),
                Path("data/processed"),
                use_pykeen=True,
                cache_root=Path("data/raw/hetionet"),
            )
        except RuntimeError:
            logging.warning("Falling back to raw TSV parsing because PyKEEN download failed")
            build_triples(Path("data/raw/hetionet"), Path("data/processed"), use_pykeen=False)
        build_splits(Path("data/processed"))
    metrics_rows: List[Dict[str, Any]] = []
    cotrain_records: List[Dict[str, Any]] = []
    diversity_records: List[Dict[str, Any]] = []
    diversity_values: List[float] = []
    robustness_pool: CandidatePool | None = None
    size_tradeoff_records: List[Dict[str, Any]] = []
    test_rows = load_test_queries(args.split, args.max_cases)
    pool_budget = max(args.budget, max(SIZE_TRADEOFF_BUDGETS))
    size_tradeoff_budgets = SIZE_TRADEOFF_BUDGETS if not args.skip_size_tradeoff else ()
    for seed in range(args.seeds):
        split_tag = f"{args.split}_seed{seed}"
        if args.reuse_artifacts:
            logging.info("Reusing cached embeddings/classifier artifacts for %s", split_tag)
        else:
            train_pykeen(
                split_tag,
                args.model,
                args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_split=args.split,
                random_seed=args.random_seed,
            )
        export_report(split_tag)
        if args.reuse_artifacts:
            classifier = Classifier.load(
                classifier_artifact_path(split_tag, args.train_subgraph_method, args.classifier),
                args.device,
            )
        else:
            classifier = train_classifier(
                args.split,
                args.train_subgraph_method,
                args.device,
                args.classifier,
                args.budget,
                artifact_tag=split_tag,
                limit=args.max_rows,
            )
        graph = load_split_filtered_graph(args.split)
        train_rows = load_jsonl(Path("data/splits") / args.split / "train.jsonl")
        if args.max_rows and args.max_rows > 0:
            train_rows = train_rows[: args.max_rows]
        budgets = CandidateBudget(max_edges_total=args.budget)
        candidate_cache: Dict[str, CandidatePool] = {}
        positive_rows = [row for row in train_rows if int(row.get("label", 1)) == 1]
        negative_rows = [row for row in train_rows if int(row.get("label", 1)) == 0]
        positive_baseline_feats, _ = _build_examples_for_rows(
            positive_rows,
            args.train_subgraph_method,
            budgets,
            graph,
            split_tag,
            candidate_cache,
            target_label=1,
        )
        neg_method = args.evo_negatives if args.evo_negatives != "none" else args.train_subgraph_method
        negative_feats, _ = _build_examples_for_rows(
            negative_rows,
            neg_method,
            budgets,
            graph,
            split_tag,
            candidate_cache,
            target_label=0,
        )
        reference_pool, ran_evo = _evaluate_round(
            args,
            split_tag,
            seed,
            0,
            classifier,
            graph,
            test_rows,
            pool_budget,
            metrics_rows,
            cotrain_records,
            diversity_records,
            diversity_values,
        )
        if robustness_pool is None and reference_pool is not None:
            robustness_pool = reference_pool
        ran_evolution_any = ran_evo
        final_classifier = classifier
        if args.cotrain_rounds > 0 and positive_baseline_feats and negative_feats and positive_rows:
            current_classifier = classifier
            for round_idx in range(1, args.cotrain_rounds + 1):
                evolved_feats = _gather_evolved_examples(
                    args,
                    graph,
                    split_tag,
                    seed,
                    round_idx,
                    current_classifier,
                    budgets,
                    positive_rows,
                    candidate_cache,
                )
                if not evolved_feats:
                    logging.warning(
                        "Skipping co-training round %s for %s because no evolved subgraphs were found",
                        round_idx,
                        split_tag,
                    )
                    continue
                current_classifier = _fit_cotrain_classifier(
                    args,
                    split_tag,
                    seed,
                    round_idx,
                    positive_baseline_feats,
                    evolved_feats,
                    negative_feats,
                )
                reference_pool, ran_evo = _evaluate_round(
                    args,
                    split_tag,
                    seed,
                    round_idx,
                    current_classifier,
                    graph,
                    test_rows,
                    pool_budget,
                    metrics_rows,
                    cotrain_records,
                    diversity_records,
                    diversity_values,
                )
                if ran_evo and robustness_pool is None and reference_pool is not None:
                    robustness_pool = reference_pool
                if ran_evo:
                    ran_evolution_any = True
                final_classifier = current_classifier
        classifier = final_classifier
        if args.method in ("evolution", "all") and ran_evolution_any:
            write_case_studies(split_tag, args.max_cases, results_root)
        if test_rows and size_tradeoff_budgets:
            for budget in size_tradeoff_budgets:
                baseline_records = score_baseline_explanations(
                    test_rows,
                    graph,
                    args.split,
                    split_tag,
                    classifier,
                    args.baseline_method,
                    budget,
                    pool_budget,
                )
                roc_b, pr_b, median_edges = summarize_records(baseline_records)
                size_tradeoff_records.append(
                    {
                        "split": args.split,
                        "seed": seed,
                        "method": args.baseline_method,
                        "budget": budget,
                        "roc_auc": roc_b,
                        "pr_auc": pr_b,
                        "median_edges": median_edges,
                    }
                )
                if args.method in ("evolution", "all") and budget != args.budget:
                    evo_budget_records = score_evolution_explanations(
                        test_rows,
                        graph,
                        args.split,
                        split_tag,
                        classifier,
                        args,
                        seed,
                        budget,
                        pool_budget,
                        persist_artifacts=False,
                    )
                    roc_e, pr_e, median_edges = summarize_records(evo_budget_records)
                    size_tradeoff_records.append(
                        {
                            "split": args.split,
                            "seed": seed,
                            "method": "evolution",
                            "budget": budget,
                            "roc_auc": roc_e,
                            "pr_auc": pr_e,
                            "median_edges": median_edges,
                        }
                    )
    write_metrics_table(metrics_rows, results_root)
    write_cotrain_metrics(cotrain_records, results_root)
    if size_tradeoff_records:
        write_size_tradeoff(size_tradeoff_records, results_root)
        curves = build_size_tradeoff_curves(size_tradeoff_records)
        if curves:
            plot_size_tradeoff(curves, results_root / "plots" / "size_tradeoff.png")
    if diversity_records:
        write_diversity_table(diversity_records, results_root)
    if diversity_values:
        plot_diversity_boxplot(diversity_values, results_root / "plots" / "diversity_boxplot.png")
    corruptions = [0.05, 0.1, 0.2]
    write_robustness_table("evolution", robustness_pool, corruptions, results_root / "tables" / "robustness.csv")
    robustness_scores = []
    if robustness_pool is not None:
        base_edges = set(robustness_pool.edges)
        for drop in corruptions:
            corrupted = corrupt_candidate(robustness_pool, drop)
            robustness_scores.append(1 - jaccard_distance(base_edges, set(corrupted.edges)))
    else:
        robustness_scores = [0.0 for _ in corruptions]
    plot_robustness_curve(corruptions, robustness_scores, results_root / "plots" / "robustness_curves.png")


def _ensure_prep(split: str) -> None:
    triples_path = Path("data/processed/triples.tsv")
    if not triples_path.exists():
        raise RuntimeError(
            "Processed triples not found under data/processed/triples.tsv; remove --skip-prep or rebuild."
        )
    split_dir = Path("data/splits") / split
    missing = []
    for subset in ("train", "val", "test"):
        if not (split_dir / f"{subset}.jsonl").exists():
            missing.append(subset)
    if missing:
        raise RuntimeError(
            f"Missing split files for '{split}' in data/splits/{split}: {', '.join(missing)}; remove --skip-prep to rebuild."
        )


if __name__ == "__main__":
    main()
