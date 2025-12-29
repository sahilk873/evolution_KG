"""Wrapper to ensure PyKEEN's Hetionet dataset is cached locally."""
from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def download_hetionet(output: Path) -> None:
    try:
        from pykeen.datasets import Hetionet
    except ImportError:
        logging.warning("PyKEEN not available; please install pykeen to download Hetionet")
        return
    Hetionet(cache_root=str(output))
    logging.info("PyKEEN Hetionet cached under %s", output)


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="Download Hetionet via PyKEEN")
    parser.add_argument("--output", type=Path, default=Path("data/raw/hetionet"))
    args = parser.parse_args()
    download_hetionet(args.output)


if __name__ == "__main__":
    main()
