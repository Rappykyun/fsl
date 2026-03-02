"""RandomForest training and evaluation utilities."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV


def _count_grid_combinations(grid: dict[str, list[Any]]) -> int:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return len(list(product(*values)))


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    random_state: int,
) -> tuple[RandomForestClassifier, RandomizedSearchCV]:
    rf_cfg = config["train"]["random_forest"]
    cv = int(rf_cfg.get("cv", 3))

    param_distributions: dict[str, list[Any]] = {
        "n_estimators": rf_cfg.get("n_estimators", [300]),
        "max_depth": rf_cfg.get("max_depth", [None]),
        "min_samples_split": rf_cfg.get("min_samples_split", [2]),
        "min_samples_leaf": rf_cfg.get("min_samples_leaf", [1]),
        "max_features": rf_cfg.get("max_features", ["sqrt"]),
        "class_weight": rf_cfg.get("class_weight", ["balanced"]),
    }

    base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    n_iter = min(12, _count_grid_combinations(param_distributions))

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    return best_model, search


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    label_names: list[str],
    out_path: str | Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def merge_metrics_file(metrics_path: str | Path, key: str, payload: dict[str, Any]) -> None:
    path = Path(metrics_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data[key] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
