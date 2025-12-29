"""Utilities to verify exported embedding arrays."""
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def export_report(split_name: str) -> None:
    artifact_dir = Path("artifacts/pykeen") / split_name
    emb_path = artifact_dir / "entity_embeddings.npy"
    rel_path = artifact_dir / "relation_embeddings.npy"
    entity_map = Path("data/processed/entity2id.json")
    if not emb_path.exists() or not rel_path.exists() or not entity_map.exists():
        logging.warning("Missing embeddings or id maps for %s", split_name)
        return
    entity_embeddings = np.load(emb_path)
    relation_embeddings = np.load(rel_path)
    with open(entity_map, "r", encoding="utf-8") as handle:
        entity2id = json.load(handle)
    if entity_embeddings.shape[0] != len(entity2id):
        raise ValueError("Entity embedding count does not match entity2id")
    report = {
        "entity_shape": entity_embeddings.shape,
        "relation_shape": relation_embeddings.shape,
        "entity_norm_mean": float(np.linalg.norm(entity_embeddings, axis=1).mean()),
        "relation_norm_mean": float(np.linalg.norm(relation_embeddings, axis=1).mean()),
    }
    with open(artifact_dir / "embedding_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logging.info("Wrote embedding report to %s", artifact_dir / "embedding_report.json")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate exported embeddings")
    parser.add_argument("--split", type=str, default="random")
    args = parser.parse_args()
    export_report(args.split)


if __name__ == "__main__":
    main()
