"""Budget constants for candidate pools."""
from dataclasses import dataclass


@dataclass
class CandidateBudget:
    max_neighbors_per_node: int = 200
    max_edges_total: int = 10_000
    max_nodes_total: int = 10_000
    hops: int = 2
