"""Face-based NMS feature extraction (neutral vs eyebrow raise)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


def _resolve_face_mesh_class():
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_mesh"):
        return solutions.face_mesh.FaceMesh

    try:
        from mediapipe.python.solutions.face_mesh import FaceMesh as LegacyFaceMesh

        return LegacyFaceMesh
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "MediaPipe FaceMesh API not found. Install a compatible mediapipe build "
            "(for example, mediapipe==0.10.14) and retry."
        ) from exc


def _dist(points: np.ndarray, a: int, b: int) -> float:
    pa = points[a]
    pb = points[b]
    return float(np.linalg.norm(pa - pb))


def extract_nms_features(face_xyz: np.ndarray) -> np.ndarray:
    """Build compact geometric features for eyebrow state classification."""
    if face_xyz.ndim != 2 or face_xyz.shape[1] != 3 or face_xyz.shape[0] < 387:
        raise ValueError(f"Expected shape [N,3] with N>=387, got {face_xyz.shape}")

    scale = _dist(face_xyz, 33, 263)
    if scale <= 1e-8:
        scale = _dist(face_xyz, 1, 152)
    if scale <= 1e-8:
        scale = 1.0

    left_eye_center_y = float((face_xyz[159, 1] + face_xyz[145, 1]) / 2.0)
    right_eye_center_y = float((face_xyz[386, 1] + face_xyz[374, 1]) / 2.0)

    left_inner_raise = (left_eye_center_y - float(face_xyz[105, 1])) / scale
    left_outer_raise = (left_eye_center_y - float(face_xyz[70, 1])) / scale
    right_inner_raise = (right_eye_center_y - float(face_xyz[334, 1])) / scale
    right_outer_raise = (right_eye_center_y - float(face_xyz[300, 1])) / scale

    avg_raise = (left_inner_raise + left_outer_raise + right_inner_raise + right_outer_raise) / 4.0
    asymmetry = abs((left_inner_raise + left_outer_raise) - (right_inner_raise + right_outer_raise)) / 2.0

    left_eye_open = _dist(face_xyz, 159, 145) / scale
    right_eye_open = _dist(face_xyz, 386, 374) / scale
    mouth_open = _dist(face_xyz, 13, 14) / scale

    brow_mid_y = float((face_xyz[105, 1] + face_xyz[334, 1]) / 2.0)
    nose_y = float(face_xyz[1, 1])
    brow_to_nose = (nose_y - brow_mid_y) / scale

    features = np.array(
        [
            left_inner_raise,
            left_outer_raise,
            right_inner_raise,
            right_outer_raise,
            avg_raise,
            asymmetry,
            left_eye_open,
            right_eye_open,
            mouth_open,
            brow_to_nose,
        ],
        dtype=np.float32,
    )
    return features


@dataclass
class NMSFeatureSettings:
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class FaceNMSExtractor:
    def __init__(
        self,
        *,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self.settings = NMSFeatureSettings(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        mesh_cls = _resolve_face_mesh_class()
        self._mesh = mesh_cls(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        face_xyz = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark], dtype=np.float32)
        return extract_nms_features(face_xyz)

    def close(self) -> None:
        self._mesh.close()
