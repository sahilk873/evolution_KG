import random

from evolution.operators import (
    add_frontier,
    crossover,
    delete_edge,
    swap_edges,
    type_aware_add,
)
from evolution.representation import SubgraphRepresentation


def test_add_frontier_appends_edge_when_budget_allows() -> None:
    rng = random.Random(0)
    parent = SubgraphRepresentation(edge_idx_set=frozenset(), node_idx_set=frozenset())
    child = add_frontier(parent, [0, 1], 2, rng)
    assert len(child.edge_idx_set) == 1
    assert child.edge_idx_set != parent.edge_idx_set


def test_add_frontier_respects_budget() -> None:
    rng = random.Random(1)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0, 1}), node_idx_set=frozenset())
    child = add_frontier(parent, [0, 1, 2], 2, rng)
    assert len(child.edge_idx_set) <= 2


def test_delete_edge_reduces_size() -> None:
    rng = random.Random(0)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0, 1}), node_idx_set=frozenset())
    child = delete_edge(parent, rng)
    assert len(child.edge_idx_set) == 1
    assert child.edge_idx_set != parent.edge_idx_set


def test_swap_edges_replaces_one_edge() -> None:
    rng = random.Random(0)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0, 1}), node_idx_set=frozenset())
    child = swap_edges(parent, [0, 1, 2], rng, 2)
    assert len(child.edge_idx_set) == len(parent.edge_idx_set)
    assert child.edge_idx_set != parent.edge_idx_set


def test_type_aware_add_prefers_preferred_indices() -> None:
    rng = random.Random(0)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0}), node_idx_set=frozenset())
    preferred = {1}
    child = type_aware_add(parent, [0, 1, 2], rng, preferred, 2)
    assert 1 in child.edge_idx_set


def test_type_aware_add_adds_when_no_preferred() -> None:
    rng = random.Random(1)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0}), node_idx_set=frozenset())
    child = type_aware_add(parent, [0, 1], rng, set(), 2)
    assert len(child.edge_idx_set) == 2


def test_crossover_trims_union_to_budget() -> None:
    parent_a = SubgraphRepresentation(edge_idx_set=frozenset({0, 2}), node_idx_set=frozenset())
    parent_b = SubgraphRepresentation(edge_idx_set=frozenset({1, 3}), node_idx_set=frozenset())
    child = crossover(parent_a, parent_b, 2, random.Random(0))
    assert len(child.edge_idx_set) == 2
    assert child.edge_idx_set == frozenset({0, 1})
