"""Random walk baseline over candidate edges."""
from __future__ import annotations

import random
from typing import Set

from evolution.representation import SubgraphRepresentation
from retrieval.candidate_pool import CandidatePool


def explain(drug: str, disease: str, candidate_pool: CandidatePool, budget: int, seed: int = 42) -> SubgraphRepresentation:
    rng = random.Random(seed)
    edge_indices = list(range(len(candidate_pool.edges)))
    rng.shuffle(edge_indices)
    selected = edge_indices[:budget]
    nodes: Set[int] = set()
    for idx in selected:
        src, _, dst = candidate_pool.edges[idx]
        nodes.update({src, dst})
    return SubgraphRepresentation(edge_idx_set=frozenset(selected), node_idx_set=frozenset(nodes))
