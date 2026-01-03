"""Evolutionary search driver that outputs top-K explanations per query."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

from evolution.operators import add_frontier, crossover, delete_edge, swap_edges, type_aware_add
from evolution.representation import SubgraphRepresentation
from features.featurize import build_feature_vector
from models.classifier import Classifier
from retrieval.candidate_pool import CandidatePool


@dataclass
class EvolutionConfig:
    pop_size: int = 80
    generations: int = 25
    budget: int = 100
    lambda_reg: float = 0.001
    mutation_rate: float = 0.6
    crossover_rate: float = 0.3
    immigrant_rate: float = 0.1
    topk: int = 5
    add_rate: float = 0.5
    swap_rate: float = 0.25
    type_rate: float = 0.25


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    median_fitness: float
    avg_size: float


def random_individual(candidate_pool: CandidatePool, budget: int, rng: random.Random) -> SubgraphRepresentation:
    max_edges = min(budget, len(candidate_pool.edges))
    indices = list(range(len(candidate_pool.edges)))
    rng.shuffle(indices)
    selection = frozenset(indices[:max_edges])
    node_indices = set()
    for idx in selection:
        src, _, dst = candidate_pool.edges[idx]
        node_indices.update({src, dst})
    return SubgraphRepresentation(edge_idx_set=selection, node_idx_set=frozenset(node_indices))


def evaluate_population(
    population: List[SubgraphRepresentation],
    drug: str,
    disease: str,
    candidate_pool: CandidatePool,
    split: str,
    classifier: Classifier,
    config: EvolutionConfig,
) -> List[Tuple[float, SubgraphRepresentation]]:
    scored: List[Tuple[float, SubgraphRepresentation]] = []
    for indiv in population:
        features = build_feature_vector(split, f"{drug}__{disease}", drug, disease, candidate_pool, indiv)
        prob = classifier.predict_proba(features.reshape(1, -1))[0]
        fitness = math.log(prob + 1e-9) - len(indiv.edge_idx_set) * config.lambda_reg
        scored.append((fitness, indiv))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def persist_topk(
    split: str,
    query_id: str,
    candidate_pool: CandidatePool,
    scored: List[Tuple[float, SubgraphRepresentation]],
    config: EvolutionConfig,
) -> None:
    dest_dir = Path("artifacts/evo") / split / query_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, (_, indiv) in enumerate(scored[: config.topk], start=1):
        edges = []
        for idx in sorted(indiv.edge_idx_set):
            if idx >= len(candidate_pool.edges):
                continue
            src, rel, dst = candidate_pool.edges[idx]
            edges.append(
                {
                    "src": candidate_pool.nodes[src],
                    "dst": candidate_pool.nodes[dst],
                    "rel_id": rel,
                    "edge_idx": idx,
                }
            )
        records.append({"rank": rank, "edges": edges, "size": len(indiv.edge_idx_set)})
    with open(dest_dir / "topk.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def persist_stats(split: str, query_id: str, stats: List[GenerationStats]) -> None:
    dest_dir = Path("artifacts/evo") / split / query_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / "stats.json", "w", encoding="utf-8") as handle:
        json.dump([stat.__dict__ for stat in stats], handle, indent=2)


def evolve_query(
    drug: str,
    disease: str,
    split: str,
    candidate_pool: CandidatePool,
    classifier: Classifier,
    config: EvolutionConfig,
    rng: random.Random,
) -> List[SubgraphRepresentation]:
    population = [random_individual(candidate_pool, config.budget, rng) for _ in range(config.pop_size)]
    stats: List[GenerationStats] = []
    for generation in range(config.generations):
        scored = evaluate_population(population, drug, disease, candidate_pool, split, classifier, config)
        if not scored:
            break
        sizes = [len(indiv.edge_idx_set) for _, indiv in scored]
        fitness_vals = [score for score, _ in scored]
        stats.append(
            GenerationStats(
                generation=generation,
                best_fitness=fitness_vals[0],
                median_fitness=float(np_median(fitness_vals)),
                avg_size=float(sum(sizes) / len(sizes)) if sizes else 0.0,
            )
        )
        elites = [indiv for _, indiv in scored[: max(1, int(len(scored) * 0.15))]]
        new_population = elites.copy()
        candidate_indices = list(range(len(candidate_pool.edges)))
        while len(new_population) < config.pop_size:
            if random.random() < config.crossover_rate and len(elites) >= 2:
                parent_a, parent_b = random.sample(elites, 2)
                child = crossover(parent_a, parent_b, config.budget, rng)
            else:
                parent = random.choice(elites)
                if rng.random() < config.mutation_rate:
                    mutation_roll = rng.random()
                    total_rate = config.add_rate + config.swap_rate + config.type_rate
                    if total_rate <= 0:
                        child = add_frontier(parent, candidate_indices, config.budget, rng)
                    else:
                        threshold_add = config.add_rate / total_rate
                        threshold_swap = (config.add_rate + config.swap_rate) / total_rate
                        if mutation_roll < threshold_add:
                            child = add_frontier(parent, candidate_indices, config.budget, rng)
                        elif mutation_roll < threshold_swap:
                            child = swap_edges(parent, candidate_indices, rng, config.budget)
                        else:
                            anchor_nodes: set[int] = set()
                            for edge_idx in parent.edge_idx_set:
                                if 0 <= edge_idx < len(candidate_pool.edges):
                                    src, _, dst = candidate_pool.edges[edge_idx]
                                    anchor_nodes.update({src, dst})
                            preferred = {
                                idx
                                for idx, (src, _, dst) in enumerate(candidate_pool.edges)
                                if idx not in parent.edge_idx_set and (src in anchor_nodes or dst in anchor_nodes)
                            }
                            child = type_aware_add(parent, candidate_indices, rng, preferred, config.budget)
                else:
                    child = delete_edge(parent, rng)
            if len(child.edge_idx_set) < config.budget and random.random() < config.immigrant_rate:
                child = random_individual(candidate_pool, config.budget, rng)
            new_population.append(child)
        population = new_population
    final_scores = evaluate_population(population, drug, disease, candidate_pool, split, classifier, config)
    query_id = f"{drug}__{disease}"
    persist_topk(split, query_id, candidate_pool, final_scores, config)
    persist_stats(split, query_id, stats)
    return [indiv for _, indiv in final_scores[: config.topk]]


def np_median(values: Iterable[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
