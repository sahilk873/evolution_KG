"""Matplotlib helpers for the required plots."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_size_tradeoff(edges: Sequence[int], scores: Sequence[float], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(edges, scores, marker="o")
    plt.xlabel("Edge budget")
    plt.ylabel("Metric")
    plt.title("Size vs performance")
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_diversity_boxplot(distances: Sequence[float], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.boxplot(distances)
    plt.title("Mechanistic diversity")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.ylabel("Jaccard distance")
    plt.savefig(path)
    plt.close()


def plot_robustness_curve(budget: Sequence[float], drops: Sequence[float], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(budget, drops, marker="s")
    plt.title("Robustness under KG corruption")
    plt.xlabel("Drop probability")
    plt.ylabel("Δ Metric")
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
