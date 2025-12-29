"""Mutation and crossover primitives for evolutionary search."""
from __future__ import annotations

import random
from typing import Iterable, List, Set, Tuple

from evolution.representation import SubgraphRepresentation

Edge = Tuple[int, int, int]


def add_frontier(
    indiv: SubgraphRepresentation, candidate_indices: List[int], budget: int, rng: random.Random
) -> SubgraphRepresentation:
    current = set(indiv.edge_idx_set)
    available = [idx for idx in candidate_indices if idx not in current]
    if not available:
        return indiv
    add_idx = rng.choice(available)
    current.add(add_idx)
    if len(current) > budget:
        current.pop()
    return SubgraphRepresentation(frozenset(current), indiv.node_idx_set)


def delete_edge(indiv: SubgraphRepresentation, rng: random.Random) -> SubgraphRepresentation:
    if not indiv.edge_idx_set:
        return indiv
    removed = rng.choice(list(indiv.edge_idx_set))
    remaining = set(indiv.edge_idx_set)
    remaining.remove(removed)
    return SubgraphRepresentation(frozenset(remaining), indiv.node_idx_set)


def swap_edges(
    indiv: SubgraphRepresentation, candidate_indices: List[int], rng: random.Random, budget: int
) -> SubgraphRepresentation:
    mutated = delete_edge(indiv, rng)
    return add_frontier(mutated, candidate_indices, budget, rng)


def type_aware_add(
    indiv: SubgraphRepresentation, candidate_indices: List[int], rng: random.Random, preferred: Set[int], budget: int
) -> SubgraphRepresentation:
    pool = [idx for idx in candidate_indices if idx in preferred and idx not in indiv.edge_idx_set]
    if pool:
        choice = rng.choice(pool)
        return add_frontier(indiv, [choice], budget, rng)
    return add_frontier(indiv, candidate_indices, budget, rng)


def crossover(
    parent_a: SubgraphRepresentation,
    parent_b: SubgraphRepresentation,
    budget: int,
    rng: random.Random,
) -> SubgraphRepresentation:
    union = parent_a.edge_idx_set | parent_b.edge_idx_set
    trimmed = set(sorted(union)[:budget])
    return SubgraphRepresentation(frozenset(trimmed), parent_a.node_idx_set | parent_b.node_idx_set)
