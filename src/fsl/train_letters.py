"""Train RandomForest model for A-Z letters."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fsl.config import load_yaml
from fsl.data.io import list_letter_samples
from fsl.features.landmarks import HandLandmarkExtractor
from fsl.models.random_forest import compute_metrics, merge_metrics_file, save_confusion_matrix, train_random_forest


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    dataset_cfg = cfg["dataset"]
    feature_cfg = cfg["feature"]
    train_cfg = cfg["train"]
    artifacts_cfg = cfg["artifacts"]

    samples = list_letter_samples(dataset_cfg["letters_dir"])
    if not samples:
        raise RuntimeError("No letter images found.")

    extractor = HandLandmarkExtractor(
        use_two_hands=feature_cfg["use_two_hands"],
        normalize=feature_cfg["normalize"],
        include_handedness=feature_cfg["include_handedness"],
        static_image_mode=True,
    )

    features: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []

    try:
        for idx, sample in enumerate(samples, start=1):
            image = cv2.imread(str(sample.path))
            if image is None:
                continue
            vector = extractor.extract(image)
            if vector is None:
                continue
            features.append(vector)
            labels.append(sample.label)
            paths.append(str(sample.path))
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(samples)} images")
    finally:
        extractor.close()

    if not features:
        raise RuntimeError("No usable features extracted from letters dataset.")

    X = np.asarray(features, dtype=np.float32)
    y_text = np.asarray(labels)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(train_cfg["test_size"]),
        random_state=int(train_cfg["random_state"]),
        stratify=y,
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    model, search = train_random_forest(X_train, y_train, cfg, random_state=int(train_cfg["random_state"]))

    y_pred = model.predict(X_test)
    class_names = encoder.inverse_transform(np.arange(len(encoder.classes_))).tolist()

    metrics = compute_metrics(y_test, y_pred, class_names)
    metrics.update(
        {
            "num_samples_total": int(len(samples)),
            "num_samples_used": int(len(features)),
            "num_samples_skipped": int(len(samples) - len(features)),
            "best_params": search.best_params_,
            "split": "holdout",
        }
    )

    model_path = Path(artifacts_cfg["model_path"])
    encoder_path = Path(artifacts_cfg["encoder_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    save_confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        label_names=class_names,
        out_path="reports/letters_confusion_matrix.png",
        title="Letters Confusion Matrix (Holdout)",
    )

    holdout_manifest = pd.DataFrame(
        {
            "path": np.asarray(paths)[test_idx],
            "label": y_text[test_idx],
            "split": "test",
        }
    )
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    holdout_manifest.to_csv("artifacts/letters_holdout_manifest.csv", index=False)

    merge_metrics_file("reports/metrics.json", "letters_train", metrics)

    print(f"Saved model: {model_path}")
    print(f"Saved encoder: {encoder_path}")
    print("Saved report: reports/letters_confusion_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
