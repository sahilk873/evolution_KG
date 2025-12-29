"""Helpers to compute robustness signals from KG noise."""
from __future__ import annotations

import random
from typing import Iterable, List, Tuple

from retrieval.candidate_pool import CandidatePool

Edge = Tuple[int, int, int]


def corrupt_candidate(candidate_pool: CandidatePool, drop_prob: float, seed: int = 42) -> CandidatePool:
    rng = random.Random(seed)
    kept_edges: List[Edge] = [edge for edge in candidate_pool.edges if rng.random() > drop_prob]
    return CandidatePool(nodes=candidate_pool.nodes, edges=kept_edges)


def compute_stability(clean_edges: Iterable[int], corrupt_edges: Iterable[int]) -> float:
    clean_set = set(clean_edges)
    corrupt_set = set(corrupt_edges)
    if not clean_set and not corrupt_set:
        return 0.0
    return 1 - len(clean_set & corrupt_set) / len(clean_set | corrupt_set)
