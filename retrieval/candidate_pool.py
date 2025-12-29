"""Build and cache candidate pools for drug–disease queries."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from kg.graph import TypedGraph
from retrieval.constraints import CandidateBudget

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EdgeTuple = Tuple[int, int, int]
NODE_TYPE_PREFERENCE = {"GENE", "PATHWAY", "ANATOMY"}


@dataclass
class CandidatePool:
    nodes: List[str]
    edges: List[EdgeTuple]

    @property
    def node_index(self) -> Dict[str, int]:
        return {node: idx for idx, node in enumerate(self.nodes)}

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}


def cache_path(split: str, query_id: str) -> Path:
    dest = Path("artifacts/cand") / split / f"{query_id}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def load_relation_map() -> Dict[str, int]:
    rel_path = Path("data/processed/relation2id.json")
    if not rel_path.exists():
        return {}
    with open(rel_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_candidate_pool(
    graph: TypedGraph,
    split: str,
    drug: str,
    disease: str,
    budgets: CandidateBudget,
    seed: int = 42,
    relation_map: Dict[str, int] | None = None,
) -> CandidatePool:
    query_id = f"{drug}__{disease}"
    cache = cache_path(split, query_id)
    if cache.exists():
        arr = np.load(cache, allow_pickle=True)
        nodes = list(arr["nodes"].tolist())
        edges = list(zip(arr["src_idx"].tolist(), arr["rel_id"].tolist(), arr["dst_idx"].tolist()))
        return CandidatePool(nodes=nodes, edges=edges)
    relation_map = relation_map or load_relation_map()
    frontier: Set[str] = {drug, disease}
    collected: Set[str] = set(frontier)
    edges: List[Tuple[str, str, str]] = []
    rng = np.random.RandomState(seed)
    for _ in range(budgets.hops):
        next_frontier: Set[str] = set()
        for node in sorted(frontier):
            for direction in ("out", "in"):
                neighbors = graph.neighbors(node, direction, budgets.max_neighbors_per_node)
                rng.shuffle(neighbors)
                for rel, nbr in neighbors:
                    if len(collected) >= budgets.max_nodes_total:
                        break
                    collected.add(nbr)
                    next_frontier.add(nbr)
                    ordered = (node, rel, nbr) if direction == "out" else (nbr, rel, node)
                    edges.append(ordered)
                    if len(edges) >= budgets.max_edges_total:
                        break
                if len(edges) >= budgets.max_edges_total or len(collected) >= budgets.max_nodes_total:
                    break
        frontier = next_frontier
        if not frontier:
            break
    nodes = list(collected)[: budgets.max_nodes_total]
    node_index = {node: idx for idx, node in enumerate(nodes)}
    trimmed_edges: List[EdgeTuple] = []
    seen = set()
    for src, rel, dst in edges:
        if len(trimmed_edges) >= budgets.max_edges_total:
            break
        if src not in node_index or dst not in node_index:
            continue
        edge_id = (node_index[src], rel, node_index[dst])
        if edge_id in seen:
            continue
        seen.add(edge_id)
        rel_id = relation_map.get(rel, abs(hash(rel)) % 10_000)
        trimmed_edges.append((node_index[src], rel_id, node_index[dst]))
    pool = CandidatePool(nodes=nodes, edges=trimmed_edges)
    np.savez_compressed(
        cache,
        nodes=np.array(nodes, dtype=object),
        src_idx=np.array([src for src, *_ in trimmed_edges], dtype=np.int32),
        rel_id=np.array([rel for _, rel, _ in trimmed_edges], dtype=np.int32),
        dst_idx=np.array([dst for *_, dst in trimmed_edges], dtype=np.int32),
    )
    logging.info("Cached candidate pool for %s", query_id)
    return pool
