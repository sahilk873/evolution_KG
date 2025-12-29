"""Simple schema helpers for typed Hetionet data."""
from typing import Dict, List

REQUIRED_NODE_TYPES = ["COMPOUND", "DISEASE", "GENE", "PATHWAY", "ANATOMY"]


def normalize_type(name: str) -> str:
    return name.upper()


def apply_reserved_types(nodes_by_type: Dict[str, List[str]]) -> None:
    for required in REQUIRED_NODE_TYPES:
        nodes_by_type.setdefault(required, [])
