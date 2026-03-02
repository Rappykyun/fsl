"""Extract static-looking frames from selected FSL-105 word videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from fsl.config import load_yaml
from fsl.constants import WORD_ID_TO_LABEL
from fsl.data.io import filter_word_rows, load_split_df
from fsl.features.landmarks import HandLandmarkExtractor


def _extract_clip_frames(
    clip_path: Path,
    out_dir: Path,
    extractor: HandLandmarkExtractor,
    sample_fps: float,
    static_motion_threshold: float,
    min_frame_width: int,
    min_frame_height: int,
    tail_start_ratio: float,
    max_frames_per_clip: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        return []

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    step = max(1, int(round(fps / sample_fps)))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tail_start_idx = 0
    if total_frames > 0:
        tail_start_idx = int(total_frames * tail_start_ratio)

    prev_gray = None
    frame_idx = 0
    saved: list[dict[str, object]] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue
        if frame_idx < tail_start_idx:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        if w < min_frame_width or h < min_frame_height:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        if prev_gray is not None:
            motion_score = float(cv2.absdiff(gray, prev_gray).mean())
        prev_gray = gray

        if motion_score > static_motion_threshold:
            frame_idx += 1
            continue

        vector = extractor.extract(frame)
        if vector is None:
            frame_idx += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{clip_path.stem}_f{frame_idx:05d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame)

        saved.append(
            {
                "frame_path": str(out_path),
                "frame_idx": frame_idx,
                "motion_score": motion_score,
                "clip_path": str(clip_path),
            }
        )
        frame_idx += 1

    capture.release()

    if max_frames_per_clip > 0 and len(saved) > max_frames_per_clip:
        # Keep the most stable tail frames to represent the held sign.
        saved = sorted(saved, key=lambda row: float(row["motion_score"]))[:max_frames_per_clip]
        saved = sorted(saved, key=lambda row: int(row["frame_idx"]))

    return saved


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    dataset_cfg = cfg["dataset"]
    feature_cfg = cfg["feature"]
    extraction_cfg = cfg["extraction"]

    fsl105_root = dataset_cfg["fsl105_root"]
    word_ids = dataset_cfg["word_ids"]
    out_root = Path(dataset_cfg["word_frames_dir"])
    metadata_dir = Path(dataset_cfg["extracted_metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)

    extractor = HandLandmarkExtractor(
        use_two_hands=feature_cfg["use_two_hands"],
        normalize=feature_cfg["normalize"],
        include_handedness=feature_cfg["include_handedness"],
        static_image_mode=False,
    )

    try:
        for split in ("train", "test"):
            split_df = load_split_df(fsl105_root, split)
            rows = filter_word_rows(split_df, word_ids)

            collected: list[dict[str, object]] = []
            for _, row in rows.iterrows():
                label = WORD_ID_TO_LABEL.get(int(row["id_label"]), str(row["label"]))
                clip_path = Path(row["resolved_path"])
                class_dir = out_root / split / label

                records = _extract_clip_frames(
                    clip_path=clip_path,
                    out_dir=class_dir,
                    extractor=extractor,
                    sample_fps=float(extraction_cfg["sample_fps"]),
                    static_motion_threshold=float(extraction_cfg["static_motion_threshold"]),
                    min_frame_width=int(extraction_cfg["min_frame_width"]),
                    min_frame_height=int(extraction_cfg["min_frame_height"]),
                    tail_start_ratio=float(extraction_cfg.get("tail_start_ratio", 0.60)),
                    max_frames_per_clip=int(extraction_cfg.get("max_frames_per_clip", 8)),
                )

                for record in records:
                    record["id_label"] = int(row["id_label"])
                    record["label"] = label
                    record["split"] = split
                collected.extend(records)

            out_csv = metadata_dir / f"word_frames_{split}.csv"
            pd.DataFrame(collected).to_csv(out_csv, index=False)
            print(f"[{split}] clips={len(rows)} extracted_frames={len(collected)} csv={out_csv}")
    finally:
        extractor.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
