"""MediaPipe hand-landmark extraction and vectorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

HAND_POINTS = 21
HAND_VECTOR_SIZE = HAND_POINTS * 3


def _resolve_hands_class():
    """Support both legacy and newer MediaPipe package layouts."""
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "hands"):
        return solutions.hands.Hands

    try:
        from mediapipe.python.solutions.hands import Hands as LegacyHands

        return LegacyHands
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        raise RuntimeError(
            "MediaPipe Hands API not found. Install a compatible mediapipe build "
            "(for example, mediapipe==0.10.14) and retry."
        ) from exc


@dataclass
class FeatureSettings:
    use_two_hands: bool = True
    normalize: bool = True
    include_handedness: bool = True


def feature_vector_length(use_two_hands: bool = True, include_handedness: bool = True) -> int:
    hands = 2 if use_two_hands else 1
    per_hand = HAND_VECTOR_SIZE + (1 if include_handedness else 0)
    return hands * per_hand


def normalize_hand_landmarks(hand_xyz: np.ndarray) -> np.ndarray:
    if hand_xyz.shape != (HAND_POINTS, 3):
        raise ValueError(f"Expected shape {(HAND_POINTS, 3)}, got {hand_xyz.shape}")

    shifted = hand_xyz - hand_xyz[0]
    scale = np.linalg.norm(shifted[9])
    if scale <= 1e-8:
        scale = np.max(np.linalg.norm(shifted, axis=1))
    if scale <= 1e-8:
        scale = 1.0
    return shifted / scale


def _serialize_hand(hand_xyz: np.ndarray, handedness: str | None, include_handedness: bool) -> np.ndarray:
    flat = hand_xyz.reshape(-1)
    if include_handedness:
        handedness_value = 1.0 if (handedness or "").upper() == "RIGHT" else 0.0
        flat = np.concatenate([flat, np.array([handedness_value], dtype=np.float32)])
    return flat.astype(np.float32)


def build_feature_vector(
    hands_xyz: Sequence[np.ndarray],
    handedness_labels: Sequence[str] | None = None,
    *,
    use_two_hands: bool = True,
    normalize: bool = True,
    include_handedness: bool = True,
) -> np.ndarray:
    max_hands = 2 if use_two_hands else 1
    per_hand_size = HAND_VECTOR_SIZE + (1 if include_handedness else 0)

    hand_data: dict[str, np.ndarray] = {}
    for idx, hand in enumerate(hands_xyz[:max_hands]):
        hand_arr = np.asarray(hand, dtype=np.float32)
        if hand_arr.shape != (HAND_POINTS, 3):
            continue
        if normalize:
            hand_arr = normalize_hand_landmarks(hand_arr)
        label = None
        if handedness_labels and idx < len(handedness_labels):
            label = handedness_labels[idx].upper()

        if label in {"LEFT", "RIGHT"}:
            hand_data[label] = _serialize_hand(hand_arr, label, include_handedness)
        else:
            key = "LEFT" if "LEFT" not in hand_data else "RIGHT"
            hand_data[key] = _serialize_hand(hand_arr, key, include_handedness)

    ordered = [hand_data.get("LEFT"), hand_data.get("RIGHT")]
    if not use_two_hands:
        ordered = ordered[:1]

    vectors: list[np.ndarray] = []
    for item in ordered:
        if item is None:
            vectors.append(np.zeros(per_hand_size, dtype=np.float32))
        else:
            vectors.append(item)

    return np.concatenate(vectors, axis=0)


class HandLandmarkExtractor:
    def __init__(
        self,
        *,
        use_two_hands: bool = True,
        normalize: bool = True,
        include_handedness: bool = True,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        hands_cls = _resolve_hands_class()
        self.settings = FeatureSettings(
            use_two_hands=use_two_hands,
            normalize=normalize,
            include_handedness=include_handedness,
        )
        self._hands = hands_cls(
            static_image_mode=static_image_mode,
            max_num_hands=2 if use_two_hands else 1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        hands_xyz: list[np.ndarray] = []
        handedness_labels: list[str] = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
            hands_xyz.append(arr)
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness_labels.append(results.multi_handedness[idx].classification[0].label)
            else:
                handedness_labels.append("LEFT")

        return build_feature_vector(
            hands_xyz,
            handedness_labels,
            use_two_hands=self.settings.use_two_hands,
            normalize=self.settings.normalize,
            include_handedness=self.settings.include_handedness,
        )

    def close(self) -> None:
        self._hands.close()
