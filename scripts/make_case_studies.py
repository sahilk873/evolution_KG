"""Generate case-study markdown from saved top-K subgraphs."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def gather_case_studies(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for topk_file in source.rglob("topk.json"):
        with open(topk_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        case_path = target / f"{topk_file.parent.name}.md"
        with open(case_path, "w", encoding="utf-8") as handle:
            handle.write(f"# Case Study {topk_file.parent.name}\n\n")
            for idx, entry in enumerate(data[:3], start=1):
                handle.write(f"## Hypothesis {idx}\n")
                for edge in entry.get("edges", []):
                    handle.write(f"- {edge['src']} -[{edge['rel_id']}]-> {edge['dst']}\n")
        logging.info("Wrote case study %s", case_path)


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="Compile case study markdown")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/evo"))
    parser.add_argument("--output", type=Path, default=Path("results/case_studies"))
    args = parser.parse_args()
    gather_case_studies(args.artifacts, args.output)


if __name__ == "__main__":
    main()
