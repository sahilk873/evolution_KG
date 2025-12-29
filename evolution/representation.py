"""Representation utilities for subgraph individuals."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
from typing import FrozenSet, Iterable, Set


@dataclass(frozen=True)
class SubgraphRepresentation:
    edge_idx_set: FrozenSet[int]
    node_idx_set: FrozenSet[int]

    @classmethod
    def from_edges(cls, edge_indices: Iterable[int], node_indices: Iterable[int]) -> "SubgraphRepresentation":
        return cls(edge_idx_set=frozenset(edge_indices), node_idx_set=frozenset(node_indices))

    def __len__(self) -> int:
        return len(self.edge_idx_set)

    def hash(self) -> str:
        pairs = sorted(self.edge_idx_set)
        digest = md5(" ".join(map(str, pairs)).encode("utf-8")).hexdigest()
        return digest
