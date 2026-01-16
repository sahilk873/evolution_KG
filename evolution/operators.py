"""Mutation and crossover primitives for evolutionary search."""
from __future__ import annotations

import random
from typing import Iterable, List, Set, Tuple

from evolution.representation import SubgraphRepresentation

Edge = Tuple[int, int, int]


def node_set_from_edges(edge_indices: Iterable[int], candidate_edges: List[Edge]) -> frozenset[int]:
    nodes: Set[int] = set()
    for idx in edge_indices:
        if 0 <= idx < len(candidate_edges):
            src, _, dst = candidate_edges[idx]
            nodes.update({src, dst})
    return frozenset(nodes)


def add_frontier(
    indiv: SubgraphRepresentation,
    candidate_edges: List[Edge],
    candidate_indices: List[int],
    budget: int,
    rng: random.Random,
    forbidden: Set[int] | None = None,
) -> SubgraphRepresentation:
    current = set(indiv.edge_idx_set)
    if indiv.node_idx_set:
        available = [
            idx
            for idx in candidate_indices
            if idx not in current
            and 0 <= idx < len(candidate_edges)
            and (
                candidate_edges[idx][0] in indiv.node_idx_set
                or candidate_edges[idx][2] in indiv.node_idx_set
            )
        ]
    else:
        available = [idx for idx in candidate_indices if idx not in current]
    forbidden = forbidden or set()
    available = [idx for idx in available if idx not in forbidden]
    if not available:
        fallback = [idx for idx in candidate_indices if idx not in current and idx not in (forbidden or set())]
        if not fallback:
            return indiv
        available = fallback
    add_idx = rng.choice(available)
    current.add(add_idx)
    if len(current) > budget:
        removal_pool = list(current - {add_idx}) or [add_idx]
        current.remove(rng.choice(removal_pool))
    node_set = node_set_from_edges(current, candidate_edges)
    return SubgraphRepresentation(frozenset(current), node_set)


def delete_edge(
    indiv: SubgraphRepresentation, candidate_edges: List[Edge], rng: random.Random
) -> SubgraphRepresentation:
    if not indiv.edge_idx_set:
        return indiv
    removed = rng.choice(list(indiv.edge_idx_set))
    remaining = set(indiv.edge_idx_set)
    remaining.remove(removed)
    node_set = node_set_from_edges(remaining, candidate_edges)
    return SubgraphRepresentation(frozenset(remaining), node_set)


def swap_edges(
    indiv: SubgraphRepresentation,
    candidate_edges: List[Edge],
    candidate_indices: List[int],
    rng: random.Random,
    budget: int,
) -> SubgraphRepresentation:
    if not indiv.edge_idx_set:
        return indiv
    removed = rng.choice(list(indiv.edge_idx_set))
    remaining = set(indiv.edge_idx_set)
    remaining.remove(removed)
    node_set = node_set_from_edges(remaining, candidate_edges)
    mutated = SubgraphRepresentation(frozenset(remaining), node_set)
    return add_frontier(
        mutated,
        candidate_edges,
        candidate_indices,
        budget,
        rng,
        forbidden={removed},
    )


def type_aware_add(
    indiv: SubgraphRepresentation,
    candidate_edges: List[Edge],
    candidate_indices: List[int],
    rng: random.Random,
    preferred: Set[int],
    budget: int,
) -> SubgraphRepresentation:
    pool = [idx for idx in candidate_indices if idx in preferred and idx not in indiv.edge_idx_set]
    if pool:
        choice = rng.choice(pool)
        return add_frontier(indiv, candidate_edges, [choice], budget, rng)
    return add_frontier(indiv, candidate_edges, candidate_indices, budget, rng)


def crossover(
    parent_a: SubgraphRepresentation,
    parent_b: SubgraphRepresentation,
    budget: int,
    rng: random.Random,
    candidate_edges: List[Edge],
    path_edge_indices: Set[int],
    relation_counts: dict[int, int],
) -> SubgraphRepresentation:
    union = parent_a.edge_idx_set | parent_b.edge_idx_set
    selected: List[int] = []
    for idx in sorted(path_edge_indices):
        if idx in union:
            selected.append(idx)
            if len(selected) >= budget:
                break
    if len(selected) < budget:
        remaining = [idx for idx in union if idx not in selected]
        remaining.sort(
            key=lambda idx: (
                relation_counts.get(candidate_edges[idx][1], 0),
                rng.random(),
            )
        )
        selected.extend(remaining)
    trimmed = set(selected[:budget])
    node_set = node_set_from_edges(trimmed, candidate_edges)
    return SubgraphRepresentation(frozenset(trimmed), node_set)
