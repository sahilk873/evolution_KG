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
    candidate_indices = [0, 1]
    candidate_edges = [(idx, 0, idx + 1) for idx in candidate_indices]
    child = add_frontier(parent, candidate_edges, candidate_indices, 2, rng)
    assert len(child.edge_idx_set) == 1
    assert child.edge_idx_set != parent.edge_idx_set


def test_add_frontier_respects_budget() -> None:
    rng = random.Random(1)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0, 1}), node_idx_set=frozenset())
    candidate_indices = [0, 1, 2]
    candidate_edges = [(idx, 0, idx + 1) for idx in candidate_indices]
    child = add_frontier(parent, candidate_edges, candidate_indices, 2, rng)
    assert len(child.edge_idx_set) <= 2


def test_delete_edge_reduces_size() -> None:
    rng = random.Random(0)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0, 1}), node_idx_set=frozenset())
    candidate_edges = [(idx, 0, idx + 1) for idx in sorted(parent.edge_idx_set)]
    child = delete_edge(parent, candidate_edges, rng)
    assert len(child.edge_idx_set) == 1
    assert child.edge_idx_set != parent.edge_idx_set


def test_swap_edges_replaces_one_edge() -> None:
    rng = random.Random(0)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0, 1}), node_idx_set=frozenset())
    candidate_indices = [0, 1, 2]
    candidate_edges = [(idx, 0, idx + 1) for idx in candidate_indices]
    child = swap_edges(parent, candidate_edges, candidate_indices, rng, 2)
    assert len(child.edge_idx_set) == len(parent.edge_idx_set)
    assert child.edge_idx_set != parent.edge_idx_set


def test_type_aware_add_prefers_preferred_indices() -> None:
    rng = random.Random(0)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0}), node_idx_set=frozenset())
    preferred = {1}
    candidate_indices = [0, 1, 2]
    candidate_edges = [(idx, 0, idx + 1) for idx in candidate_indices]
    child = type_aware_add(parent, candidate_edges, candidate_indices, rng, preferred, 2)
    assert 1 in child.edge_idx_set


def test_type_aware_add_adds_when_no_preferred() -> None:
    rng = random.Random(1)
    parent = SubgraphRepresentation(edge_idx_set=frozenset({0}), node_idx_set=frozenset())
    candidate_indices = [0, 1]
    candidate_edges = [(idx, 0, idx + 1) for idx in candidate_indices]
    child = type_aware_add(parent, candidate_edges, candidate_indices, rng, set(), 2)
    assert len(child.edge_idx_set) == 2


def test_crossover_trims_union_to_budget() -> None:
    parent_a = SubgraphRepresentation(edge_idx_set=frozenset({0, 2}), node_idx_set=frozenset())
    parent_b = SubgraphRepresentation(edge_idx_set=frozenset({1, 3}), node_idx_set=frozenset())
    candidate_edges = [(idx, 0, idx + 1) for idx in range(4)]
    path_edge_indices = {0, 1}
    relation_counts = {0: 1}
    child = crossover(
        parent_a,
        parent_b,
        2,
        random.Random(0),
        candidate_edges,
        path_edge_indices,
        relation_counts,
    )
    assert len(child.edge_idx_set) == 2
    assert child.edge_idx_set == frozenset({0, 1})
