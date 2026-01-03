"""Baseline logistic regression and optional GPU MLP classifiers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Classifier:
    def __init__(self, mode: Literal["logistic", "mlp"] = "logistic", device: str = "cpu"):
        self.mode = mode
        self.device = device
        self.model = None

    def _mlp_factory(self, input_dim: int) -> Any:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - only when torch unavailable
            raise RuntimeError("Torch is required for MLP classifiers") from exc

        class TorchMLP(nn.Module):  # type: ignore[valid-type]
            def __init__(self, hidden_dim: int = 256) -> None:
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return self.backbone(x).squeeze(-1)

        return TorchMLP(), torch, nn

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.mode == "logistic":
            self.model = LogisticRegression(max_iter=1000, solver="liblinear")
            self.model.fit(X, y)
            return
        mlp, torch, nn = self._mlp_factory(X.shape[1])
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        target = torch.tensor(y, dtype=torch.float32).to(self.device)
        mlp = mlp.to(self.device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(20):
            optimizer.zero_grad()
            logits = mlp(tensor)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
        self.model = mlp

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "logistic" and self.model is not None:
            proba = self.model.predict_proba(X)
            return proba[:, 1]
        if self.mode == "mlp" and self.model is not None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - only when torch unavailable
                raise RuntimeError("Torch is required for MLP classifiers") from exc
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            prob = torch.sigmoid(logits).cpu().numpy()
            return prob
        raise RuntimeError("Classifier has not been trained")

    def save(self, path: Path, input_dim: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"mode": self.mode}
        if input_dim is not None:
            meta["input_dim"] = input_dim
        if self.mode == "logistic" and hasattr(self.model, "coef_"):
            joblib.dump(self.model, path.with_suffix(".joblib"))
        elif self.mode == "mlp" and self.model is not None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Torch is required for MLP classifiers") from exc
            torch.save(self.model.state_dict(), path.with_suffix(".pt"))
        else:
            raise RuntimeError("Unknown model format")
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

    @classmethod
    def load(cls, path: Path, device: str) -> "Classifier":
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Classifier metadata not found at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        mode = meta.get("mode")
        if mode not in ("logistic", "mlp"):
            raise RuntimeError(f"Unsupported classifier mode {mode}")
        classifier = cls(mode=mode, device=device)
        if mode == "logistic":
            model_path = path.with_suffix(".joblib")
            if not model_path.exists():
                raise FileNotFoundError(f"Classifier parameters missing at {model_path}")
            classifier.model = joblib.load(model_path)
            return classifier
        input_dim = meta.get("input_dim")
        if input_dim is None:
            raise RuntimeError("Missing input_dim metadata for MLP classifier")
        mlp, torch, _ = classifier._mlp_factory(int(input_dim))
        weight_path = path.with_suffix(".pt")
        if not weight_path.exists():
            raise FileNotFoundError(f"Classifier weights not found at {weight_path}")
        mlp.load_state_dict(torch.load(weight_path, map_location=device))
        classifier.model = mlp.to(device)
        return classifier
