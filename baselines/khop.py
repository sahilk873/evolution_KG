"""2-hop baseline that expands from drug and disease anchors."""
from __future__ import annotations

from typing import Set

from evolution.representation import SubgraphRepresentation
from retrieval.candidate_pool import CandidatePool


def explain(drug: str, disease: str, candidate_pool: CandidatePool, budget: int) -> SubgraphRepresentation:
    def node_index(value: str) -> int | None:
        try:
            return candidate_pool.nodes.index(value)
        except ValueError:
            return None

    anchors = {idx for idx in (node_index(drug), node_index(disease)) if idx is not None}
    selected_edges: Set[int] = set()
    frontier: Set[int] = set(anchors)
    for _ in range(2):
        new_frontier: Set[int] = set()
        for idx, (src, _, dst) in enumerate(candidate_pool.edges):
            if idx in selected_edges:
                continue
            if len(selected_edges) >= budget:
                break
            if src in frontier or dst in frontier:
                selected_edges.add(idx)
                new_frontier.update({src, dst})
        frontier = new_frontier
        if not frontier:
            break
    nodes = set(frontier) | anchors
    return SubgraphRepresentation(edge_idx_set=frozenset(selected_edges), node_idx_set=frozenset(nodes))
