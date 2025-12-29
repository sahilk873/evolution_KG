"""Shortest path baseline inside the candidate pool."""
from __future__ import annotations

from evolution.representation import SubgraphRepresentation
from features.path_features import shortest_paths
from retrieval.candidate_pool import CandidatePool


def explain(drug: str, disease: str, candidate_pool: CandidatePool, budget: int, max_paths: int = 5) -> SubgraphRepresentation:
    path_list = shortest_paths(candidate_pool, drug, disease, k=max_paths)
    edge_lookup = {
        (candidate_pool.nodes[src_idx], candidate_pool.nodes[dst_idx]): idx
        for idx, (src_idx, _, dst_idx) in enumerate(candidate_pool.edges)
    }
    selected_edges = set()
    node_indices = set()
    for path in path_list:
        for src_node, dst_node in zip(path, path[1:]):
            if len(selected_edges) >= budget:
                break
            edge_key = (src_node, dst_node)
            if edge_key not in edge_lookup:
                continue
            idx = edge_lookup[edge_key]
            selected_edges.add(idx)
            try:
                node_indices.add(candidate_pool.nodes.index(src_node))
                node_indices.add(candidate_pool.nodes.index(dst_node))
            except ValueError:
                continue
        if len(selected_edges) >= budget:
            break
    return SubgraphRepresentation(edge_idx_set=frozenset(selected_edges), node_idx_set=frozenset(node_indices))
