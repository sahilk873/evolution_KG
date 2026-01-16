"""Fast cached featurization of subgraphs leveraging pretrained embeddings."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from evolution.representation import SubgraphRepresentation
from retrieval.candidate_pool import CandidatePool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NODE_TYPES = ["COMPOUND", "DISEASE", "GENE", "PATHWAY", "ANATOMY", "UNKNOWN"]

FEATURE_CACHE: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

def get_query_cache(split: str, query_id: str) -> Dict[str, np.ndarray]:
    key = (split, query_id)
    if key not in FEATURE_CACHE:
        FEATURE_CACHE[key] = {}
    return FEATURE_CACHE[key]

def clear_feature_cache(split: str | None = None, query_id: str | None = None) -> None:
    if split and query_id:
        FEATURE_CACHE.pop((split, query_id), None)
        return
    if split:
        for key in [k for k in FEATURE_CACHE if k[0] == split]:
            FEATURE_CACHE.pop(key, None)
        return
    FEATURE_CACHE.clear()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        logging.error("Could not parse JSON %s (%s); treating as empty mapping", path, exc)
        return {}


def load_entity_embeddings(split: str) -> Dict[str, np.ndarray]:
    emb_path = Path("artifacts/pykeen") / split / "entity_embeddings.npy"
    mapping = load_json(Path("data/processed/entity2id.json"))
    if not emb_path.exists() or not mapping:
        raise FileNotFoundError("Embeddings or entity map missing")
    embeddings = np.load(emb_path)
    return {
        entity: embeddings[idx]
        for entity, idx in mapping.items()
        if isinstance(idx, int) and idx < len(embeddings)
    }


def cache_path(split: str, query_id: str, subgraph_hash: str) -> Path:
    dest = Path("artifacts/features") / split / query_id / f"{subgraph_hash}.npy"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def aggregate_node_embeddings(nodes: Sequence[str], embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    vectors = [embeddings[node] for node in nodes if node in embeddings]
    if not vectors:
        sample = next(iter(embeddings.values()))
        return np.concatenate([np.zeros_like(sample), np.zeros_like(sample)])
    stacked = np.stack(vectors)
    return np.concatenate([stacked.mean(axis=0), stacked.max(axis=0)])


def resolve_nodes(candidate_pool: CandidatePool, edge_index_set: Iterable[int]) -> Sequence[str]:
    nodes = set()
    for idx in edge_index_set:
        if 0 <= idx < len(candidate_pool.edges):
            src, _, dst = candidate_pool.edges[idx]
            nodes.add(candidate_pool.nodes[src])
            nodes.add(candidate_pool.nodes[dst])
    return list(nodes)


def build_feature_vector(
    split: str,
    query_id: str,
    drug: str,
    disease: str,
    candidate_pool: CandidatePool,
    subgraph: SubgraphRepresentation,
) -> np.ndarray:
    subgraph_hash = subgraph.hash()
    query_cache = get_query_cache(split, query_id)
    if subgraph_hash in query_cache:
        return query_cache[subgraph_hash]
    dest = cache_path(split, query_id, subgraph_hash)
    cached = load_cached_features(dest)
    if cached is not None:
        logging.debug("Loaded cached features from %s", dest)
        query_cache[subgraph_hash] = cached
        return cached

    entity_embeddings = load_entity_embeddings(split)
    node_types = load_json(Path("data/processed/node_types.json"))
    relation_map = load_json(Path("data/processed/relation2id.json"))
    drug_vec = entity_embeddings.get(drug, np.zeros_like(next(iter(entity_embeddings.values()))))
    disease_vec = entity_embeddings.get(disease, np.zeros_like(drug_vec))
    prod_vec = drug_vec * disease_vec
    diff_vec = np.abs(drug_vec - disease_vec)
    subgraph_nodes = resolve_nodes(candidate_pool, subgraph.edge_idx_set)
    pooled_vec = aggregate_node_embeddings(subgraph_nodes, entity_embeddings)
    rel_hist = np.zeros(max(len(relation_map), 1))
    for idx in subgraph.edge_idx_set:
        if 0 <= idx < len(candidate_pool.edges):
            _, rel_id, _ = candidate_pool.edges[idx]
            if rel_id < rel_hist.shape[0]:
                rel_hist[rel_id] += 1
    type_hist = np.zeros(len(NODE_TYPES))
    for node in subgraph_nodes:
        typ = node_types.get(node, "UNKNOWN").upper()
        if typ in NODE_TYPES:
            type_hist[NODE_TYPES.index(typ)] += 1
    feature_vector = np.concatenate([drug_vec, disease_vec, prod_vec, diff_vec, pooled_vec, rel_hist, type_hist])
    np.save(dest, feature_vector)
    query_cache[subgraph_hash] = feature_vector
    logging.debug("Cached features to %s", dest)
    return feature_vector


def load_cached_features(artifacts_path: Path) -> np.ndarray | None:
    if artifacts_path.exists():
        return np.load(artifacts_path)
    return None
