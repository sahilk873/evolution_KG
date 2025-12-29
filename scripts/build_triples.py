"""Parse Hetionet triples, preferring PyKEEN's dataset helper."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NODE_TYPES = ["COMPOUND", "DISEASE", "GENE", "PATHWAY", "ANATOMY", "UNKNOWN"]
USE_PYKEEN_DATASET = os.getenv("USE_PYKEEN_DATASET", "1") == "1"


def _infer_type(label: str) -> str:
    text = label.upper()
    for node_type in NODE_TYPES:
        if text.startswith(node_type):
            return node_type
    if ":" in text:
        return text.split(":")[0]
    if "::" in text:
        return text.split("::")[0]
    return "UNKNOWN"


def _raw_triple_paths(raw_dir: Path) -> Iterable[Path]:
    if not raw_dir.exists():
        return ()
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".tsv", ".txt", ".csv"}:
            yield path


def _read_raw_triples(raw_dir: Path) -> Tuple[Sequence[Tuple[str, str, str]], Dict[str, str]]:
    triples = []
    node_types: Dict[str, str] = {}
    for path in _raw_triple_paths(raw_dir):
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if not {"subject", "predicate", "object"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            head = str(row["subject"]).strip()
            rel = str(row["predicate"]).strip()
            tail = str(row["object"]).strip()
            if not head or not tail or not rel:
                continue
            triples.append((head, rel, tail))
            node_types.setdefault(head, _infer_type(head))
            node_types.setdefault(tail, _infer_type(tail))
    return triples, node_types


def _write_processed(
    df: pd.DataFrame,
    processed_dir: Path,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    node_types: Dict[str, str],
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "triples.tsv", sep="\t", index=False)
    for name, payload in (
        ("entity2id.json", entity2id),
        ("relation2id.json", relation2id),
        ("node_types.json", node_types),
    ):
        with open(processed_dir / name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _from_pykeen(processed_dir: Path, cache_root: Path) -> None:
    from pykeen.datasets import Hetionet

    dataset = Hetionet(cache_root=str(cache_root))
    mapped = dataset.training.mapped_triples.cpu().numpy()
    entity_to_id = dataset.training.entity_to_id
    relation_to_id = dataset.training.relation_to_id
    id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}
    id_to_relation = {idx: rel for rel, idx in relation_to_id.items()}
    rows = []
    node_types: Dict[str, str] = {}
    for head_id, rel_id, tail_id in mapped:
        head = id_to_entity[int(head_id)]
        tail = id_to_entity[int(tail_id)]
        relation = id_to_relation[int(rel_id)]
        node_types.setdefault(head, _infer_type(head))
        node_types.setdefault(tail, _infer_type(tail))
        rows.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "head_type": node_types[head],
                "tail_type": node_types[tail],
            }
        )
    df = pd.DataFrame(rows)
    _write_processed(df, processed_dir, entity_to_id, relation_to_id, node_types)
    logging.info("Built %d triples from PyKEEN Hetionet", len(df))


def build_triples(
    raw_dir: Path,
    processed_dir: Path,
    use_pykeen: bool | None = None,
    cache_root: Path | None = None,
) -> None:
    use_pykeen = USE_PYKEEN_DATASET if use_pykeen is None else use_pykeen
    if use_pykeen:
        cache_root = cache_root or raw_dir
        _from_pykeen(processed_dir, cache_root)
        return
    triples, node_types = _read_raw_triples(raw_dir)
    if not triples:
        raise RuntimeError("No triples found in raw data")
    entity2id = {entity: idx for idx, entity in enumerate(sorted({node for triple in triples for node in (triple[0], triple[2])}))}
    relation2id = {rel: idx for idx, rel in enumerate(sorted({rel for _, rel, _ in triples}))}
    rows = [
        {
            "head": head,
            "relation": rel,
            "tail": tail,
            "head_type": node_types.get(head, "UNKNOWN"),
            "tail_type": node_types.get(tail, "UNKNOWN"),
        }
        for head, rel, tail in triples
    ]
    df = pd.DataFrame(rows)
    _write_processed(df, processed_dir, entity2id, relation2id, node_types)
    logging.info("Built %d triples from raw assets", len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Hetionet triples")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hetionet"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--no-pykeen", dest="use_pykeen", action="store_false")
    args = parser.parse_args()
    build_triples(args.raw_dir, args.processed_dir, use_pykeen=args.use_pykeen)


if __name__ == "__main__":
    main()
