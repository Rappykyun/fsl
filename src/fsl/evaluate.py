"""Evaluate trained models on deterministic test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fsl.config import load_yaml
from fsl.data.io import list_letter_samples
from fsl.features.landmarks import HandLandmarkExtractor
from fsl.features.nms import FaceNMSExtractor
from fsl.models.random_forest import compute_metrics, merge_metrics_file, save_confusion_matrix


def _extract_from_paths(paths: list[str], extractor) -> tuple[np.ndarray, list[int]]:
    vectors: list[np.ndarray] = []
    kept_indices: list[int] = []

    for idx, path in enumerate(paths):
        image = cv2.imread(path)
        if image is None:
            continue
        vector = extractor.extract(image)
        if vector is None:
            continue
        vectors.append(vector)
        kept_indices.append(idx)

    return np.asarray(vectors, dtype=np.float32), kept_indices


def _evaluate_letters(model_path: Path, split: str) -> dict:
    train_cfg = load_yaml("configs/train_letters.yaml")
    data_cfg = load_yaml("configs/data.yaml")

    model = joblib.load(model_path)
    encoder_path = model_path.with_name("letters_label_encoder.joblib")
    encoder = joblib.load(encoder_path)

    samples = list_letter_samples(data_cfg["dataset"]["letters_dir"])

    extractor = HandLandmarkExtractor(
        use_two_hands=train_cfg["feature"]["use_two_hands"],
        normalize=train_cfg["feature"]["normalize"],
        include_handedness=train_cfg["feature"]["include_handedness"],
        static_image_mode=True,
    )

    try:
        labels = [sample.label for sample in samples]
        paths = [str(sample.path) for sample in samples]
        y = encoder.transform(labels)

        indices = np.arange(len(samples))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(train_cfg["train"]["test_size"]),
            random_state=int(train_cfg["train"]["random_state"]),
            stratify=y,
        )

        use_idx = train_idx if split == "train" else test_idx
        selected_paths = [paths[i] for i in use_idx]
        selected_y = y[use_idx]

        X, kept = _extract_from_paths(selected_paths, extractor)
        y_true = selected_y[kept]

        y_pred = model.predict(X)
        class_names = encoder.inverse_transform(np.arange(len(encoder.classes_))).tolist()

        metrics = compute_metrics(y_true, y_pred, class_names)
        save_confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            label_names=class_names,
            out_path="reports/letters_confusion_matrix.png",
            title=f"Letters Confusion Matrix ({split})",
        )
        merge_metrics_file("reports/metrics.json", f"letters_eval_{split}", metrics)
        return metrics
    finally:
        extractor.close()


def _evaluate_words(model_path: Path, split: str) -> dict:
    train_cfg = load_yaml("configs/train_words.yaml")

    model = joblib.load(model_path)
    encoder_path = model_path.with_name("words_label_encoder.joblib")
    encoder = joblib.load(encoder_path)

    metadata_path = Path(train_cfg["dataset"]["metadata_dir"]) / f"word_frames_{split}.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run frame extraction first.")

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {metadata_path}")

    extractor = HandLandmarkExtractor(
        use_two_hands=train_cfg["feature"]["use_two_hands"],
        normalize=train_cfg["feature"]["normalize"],
        include_handedness=train_cfg["feature"]["include_handedness"],
        static_image_mode=True,
    )

    try:
        paths = df["frame_path"].astype(str).tolist()
        y = encoder.transform(df["label"].astype(str).tolist())
        X, kept = _extract_from_paths(paths, extractor)
        y_true = y[kept]

        y_pred = model.predict(X)
        class_names = encoder.inverse_transform(np.arange(len(encoder.classes_))).tolist()

        metrics = compute_metrics(y_true, y_pred, class_names)
        save_confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            label_names=class_names,
            out_path="reports/words_confusion_matrix.png",
            title=f"Words Confusion Matrix ({split})",
        )
        merge_metrics_file("reports/metrics.json", f"words_eval_{split}", metrics)
        return metrics
    finally:
        extractor.close()


def _evaluate_nms(model_path: Path, split: str) -> dict:
    train_cfg = load_yaml("configs/train_nms.yaml")

    model = joblib.load(model_path)
    encoder_path = model_path.with_name("nms_label_encoder.joblib")
    encoder = joblib.load(encoder_path)

    metadata_path = Path(train_cfg["dataset"]["metadata_dir"]) / f"nms_frames_{split}.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run NMS frame extraction first.")

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {metadata_path}")

    extractor = FaceNMSExtractor(
        static_image_mode=True,
        max_num_faces=int(train_cfg.get("nms", {}).get("max_num_faces", 1)),
        min_detection_confidence=float(train_cfg.get("nms", {}).get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(train_cfg.get("nms", {}).get("min_tracking_confidence", 0.5)),
    )

    try:
        paths = df["frame_path"].astype(str).tolist()
        y = encoder.transform(df["label"].astype(str).tolist())
        X, kept = _extract_from_paths(paths, extractor)
        y_true = y[kept]

        y_pred = model.predict(X)
        class_names = encoder.inverse_transform(np.arange(len(encoder.classes_))).tolist()

        metrics = compute_metrics(y_true, y_pred, class_names)
        save_confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            label_names=class_names,
            out_path="reports/nms_confusion_matrix.png",
            title=f"NMS Confusion Matrix ({split})",
        )
        merge_metrics_file("reports/metrics.json", f"nms_eval_{split}", metrics)
        return metrics
    finally:
        extractor.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    model_path = Path(args.model)
    model_name = model_path.name.lower()

    if "letters" in model_name:
        metrics = _evaluate_letters(model_path, args.split)
    elif "words" in model_name:
        metrics = _evaluate_words(model_path, args.split)
    elif "nms" in model_name:
        metrics = _evaluate_nms(model_path, args.split)
    else:
        raise ValueError("Model name must include 'letters', 'words', or 'nms' for routing.")

    print("Evaluation complete")
    print({k: metrics[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]})


if __name__ == "__main__":
    main()
