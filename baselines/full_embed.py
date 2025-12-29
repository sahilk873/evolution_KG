"""Embedding-only baseline that uses pair-level features."""
from __future__ import annotations

from evolution.representation import SubgraphRepresentation
from retrieval.candidate_pool import CandidatePool


def explain(drug: str, disease: str, candidate_pool: CandidatePool, budget: int) -> SubgraphRepresentation:
    nodes = set()
    for anchor in (drug, disease):
        try:
            nodes.add(candidate_pool.nodes.index(anchor))
        except ValueError:
            continue
    return SubgraphRepresentation(edge_idx_set=frozenset(), node_idx_set=frozenset(nodes))
