"""Utilities building path-based descriptors inside candidate pools."""
from __future__ import annotations

from typing import List, Tuple

import networkx as nx
from retrieval.candidate_pool import CandidatePool


def build_graph(candidate_pool: CandidatePool) -> nx.DiGraph:
    graph = nx.DiGraph()
    for idx, node in enumerate(candidate_pool.nodes):
        graph.add_node(node)
    for src_idx, rel_id, dst_idx in candidate_pool.edges:
        graph.add_edge(candidate_pool.nodes[src_idx], candidate_pool.nodes[dst_idx], relation=rel_id)
    return graph


def shortest_paths(candidate_pool: CandidatePool, source: str, target: str, k: int = 5) -> List[List[str]]:
    graph = build_graph(candidate_pool)
    if source not in graph or target not in graph:
        return []
    paths: List[List[str]] = []
    try:
        for path in nx.all_shortest_paths(graph, source, target):
            paths.append(path)
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return paths
