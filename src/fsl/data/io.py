"""Dataset readers and path resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str


def list_letter_samples(letters_dir: str | Path) -> list[ImageSample]:
    root = Path(letters_dir)
    samples: list[ImageSample] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file():
                samples.append(ImageSample(path=image_path, label=class_dir.name.upper()))
    return samples


def _normalize_csv_path(raw_path: str) -> str:
    return raw_path.replace("\\", "/").strip()


def resolve_video_path(fsl105_root: str | Path, vid_path: str) -> Path:
    root = Path(fsl105_root)
    normalized = _normalize_csv_path(vid_path)
    trimmed = normalized.removeprefix("./")

    candidates = [
        root / trimmed,
        root / "clips" / trimmed,
        root / "clips" / trimmed.removeprefix("clips/"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_labels_df(fsl105_root: str | Path) -> pd.DataFrame:
    labels_path = Path(fsl105_root) / "labels.csv"
    return pd.read_csv(labels_path, encoding="utf-8-sig")


def load_split_df(fsl105_root: str | Path, split: str) -> pd.DataFrame:
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    csv_path = Path(fsl105_root) / f"{split}.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["id_label"] = df["id_label"].astype(int)
    df["resolved_path"] = df["vid_path"].apply(lambda value: str(resolve_video_path(fsl105_root, value)))
    return df


def filter_word_rows(df: pd.DataFrame, word_ids: Iterable[int]) -> pd.DataFrame:
    allowed = {int(item) for item in word_ids}
    return df[df["id_label"].isin(allowed)].copy()
