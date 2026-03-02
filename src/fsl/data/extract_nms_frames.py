"""Extract face frames from NMS videos and create train/test metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from fsl.config import load_yaml
from fsl.features.nms import FaceNMSExtractor


def _list_video_rows(nms_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for class_dir in sorted(p for p in nms_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for video_path in sorted(class_dir.glob("*")):
            if video_path.is_file():
                rows.append({"video_path": str(video_path), "label": label})
    return pd.DataFrame(rows)


def _extract_video_frames(
    video_path: Path,
    out_dir: Path,
    extractor: FaceNMSExtractor,
    sample_fps: float,
    static_motion_threshold: float,
    min_frame_width: int,
    min_frame_height: int,
    tail_start_ratio: float,
    max_frames_per_video: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    step = max(1, int(round(fps / sample_fps)))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tail_start_idx = int(total_frames * tail_start_ratio) if total_frames > 0 else 0

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

        vec = extractor.extract(frame)
        if vec is None:
            frame_idx += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{video_path.stem}_f{frame_idx:05d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame)

        saved.append(
            {
                "frame_path": str(out_path),
                "frame_idx": frame_idx,
                "motion_score": motion_score,
                "video_path": str(video_path),
            }
        )
        frame_idx += 1

    capture.release()

    if max_frames_per_video > 0 and len(saved) > max_frames_per_video:
        saved = sorted(saved, key=lambda row: float(row["motion_score"]))[:max_frames_per_video]
        saved = sorted(saved, key=lambda row: int(row["frame_idx"]))

    return saved


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    dataset_cfg = cfg["dataset"]
    extraction_cfg = cfg["extraction"]
    nms_cfg = cfg.get("nms", {})

    nms_root = Path(dataset_cfg["nms_root"])
    out_root = Path(dataset_cfg["nms_frames_dir"])
    metadata_dir = Path(dataset_cfg["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_df = _list_video_rows(nms_root)
    if video_df.empty:
        raise RuntimeError(f"No NMS videos found in {nms_root}")

    train_df, test_df = train_test_split(
        video_df,
        test_size=float(dataset_cfg.get("test_size", 0.2)),
        random_state=int(dataset_cfg.get("random_state", 42)),
        stratify=video_df["label"],
    )

    extractor = FaceNMSExtractor(
        static_image_mode=False,
        max_num_faces=int(nms_cfg.get("max_num_faces", 1)),
        min_detection_confidence=float(nms_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(nms_cfg.get("min_tracking_confidence", 0.5)),
    )

    try:
        for split, split_df in (("train", train_df), ("test", test_df)):
            collected: list[dict[str, object]] = []
            for _, row in split_df.iterrows():
                video_path = Path(str(row["video_path"]))
                label = str(row["label"])
                class_dir = out_root / split / label

                records = _extract_video_frames(
                    video_path=video_path,
                    out_dir=class_dir,
                    extractor=extractor,
                    sample_fps=float(extraction_cfg.get("sample_fps", 2.0)),
                    static_motion_threshold=float(extraction_cfg.get("static_motion_threshold", 10.0)),
                    min_frame_width=int(extraction_cfg.get("min_frame_width", 64)),
                    min_frame_height=int(extraction_cfg.get("min_frame_height", 64)),
                    tail_start_ratio=float(extraction_cfg.get("tail_start_ratio", 0.20)),
                    max_frames_per_video=int(extraction_cfg.get("max_frames_per_video", 10)),
                )

                for record in records:
                    record["label"] = label
                    record["split"] = split
                collected.extend(records)

            frame_csv = metadata_dir / f"nms_frames_{split}.csv"
            pd.DataFrame(collected).to_csv(frame_csv, index=False)

            video_csv = metadata_dir / f"nms_videos_{split}.csv"
            split_df.to_csv(video_csv, index=False)

            print(f"[{split}] videos={len(split_df)} extracted_frames={len(collected)} csv={frame_csv}")
    finally:
        extractor.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
