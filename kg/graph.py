"""Adjacency helpers for typed Hetionet graphs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from kg import schema

Edge = Tuple[str, str]


@dataclass
class TypedGraph:
    out_adj: Dict[str, List[Tuple[str, str]]]
    in_adj: Dict[str, List[Tuple[str, str]]]
    nodes_by_type: Dict[str, List[str]]

    @classmethod
    def from_triples(cls, triples_path: Path, budget_per_node: int = 200) -> "TypedGraph":
        df = pd.read_csv(triples_path, sep="\t", dtype=str)
        out_adj = defaultdict(list)
        in_adj = defaultdict(list)
        nodes_by_type = defaultdict(list)
        for _, row in df.iterrows():
            head = row["head"]
            rel = row["relation"]
            tail = row["tail"]
            out_adj[head].append((rel, tail))
            in_adj[tail].append((rel, head))
            nodes_by_type[row.get("head_type", "UNKNOWN")].append(head)
            nodes_by_type[row.get("tail_type", "UNKNOWN")].append(tail)
        for key in nodes_by_type:
            nodes_by_type[key] = list(dict.fromkeys(nodes_by_type[key]))
        schema.apply_reserved_types(nodes_by_type)
        return cls(out_adj=out_adj, in_adj=in_adj, nodes_by_type=nodes_by_type)

    def neighbors(self, node_id: str, direction: str = "out", cap: int = 200) -> List[Edge]:
        adj = self.out_adj if direction == "out" else self.in_adj
        return adj.get(node_id, [])[:cap]

    def sample_bounded_neighbors(
        self, node_id: str, direction: str = "out", cap: int = 200, max_total: int = 10000
    ) -> List[Edge]:
        result = []
        neighbors = self.neighbors(node_id, direction, cap)
        for rel, nbr in neighbors:
            if len(result) >= max_total:
                break
            result.append((rel, nbr))
        return result


def load_graph(budget_per_node: int = 200) -> TypedGraph:
    triples = Path("data/processed/triples.tsv")
    return TypedGraph.from_triples(triples, budget_per_node)
