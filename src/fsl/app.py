"""Realtime webcam app for static FSL recognition."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from fsl.config import load_yaml
from fsl.constants import WORD_LABEL_TO_FILIPINO
from fsl.features.landmarks import HandLandmarkExtractor
from fsl.features.nms import FaceNMSExtractor
from fsl.inference.predictor import PredictionStabilizer


def _map_display_label(mode: str, label: str) -> str:
    if mode == "WORD":
        return WORD_LABEL_TO_FILIPINO.get(label, label)
    return label


def _predict_label(model, encoder, vector: np.ndarray) -> tuple[str, float]:
    proba = model.predict_proba(vector.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    label = str(encoder.inverse_transform([pred_idx])[0])
    return label, confidence


def _candidate_vectors(vector: np.ndarray, feature_cfg: dict) -> list[np.ndarray]:
    """Generate fallback candidates to reduce handedness-mismatch errors in webcam demos."""
    candidates: list[np.ndarray] = [vector]
    include_handedness = bool(feature_cfg.get("include_handedness", False))
    if not include_handedness:
        return candidates

    use_two_hands = bool(feature_cfg.get("use_two_hands", False))
    num_hands = 2 if use_two_hands else 1
    if num_hands <= 0 or vector.size % num_hands != 0:
        return candidates

    per_hand = vector.size // num_hands
    if per_hand < 2:
        return candidates

    flipped_bits = vector.copy()
    for hand_idx in range(num_hands):
        bit_idx = (hand_idx + 1) * per_hand - 1
        flipped_bits[bit_idx] = 1.0 - flipped_bits[bit_idx]
    candidates.append(flipped_bits)

    if num_hands == 2:
        swapped = vector.copy()
        left = vector[:per_hand].copy()
        right = vector[per_hand : 2 * per_hand].copy()
        swapped[:per_hand] = right
        swapped[per_hand : 2 * per_hand] = left
        swapped[per_hand - 1] = 1.0 - swapped[per_hand - 1]
        swapped[2 * per_hand - 1] = 1.0 - swapped[2 * per_hand - 1]
        candidates.append(swapped)

    unique_candidates: list[np.ndarray] = []
    for item in candidates:
        if not any(np.array_equal(item, existing) for existing in unique_candidates):
            unique_candidates.append(item)
    return unique_candidates


def _predict_with_fallback(model, encoder, vector: np.ndarray, feature_cfg: dict) -> tuple[str, float]:
    best_label = ""
    best_conf = -1.0
    for candidate in _candidate_vectors(vector, feature_cfg):
        label, conf = _predict_label(model, encoder, candidate)
        if conf > best_conf:
            best_label = label
            best_conf = conf
    return best_label, best_conf


def _load_logo(path_str: str, max_width: int) -> tuple[np.ndarray, np.ndarray] | None:
    logo_path = Path(path_str)
    if not logo_path.exists():
        return None

    logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if logo is None:
        return None

    if logo.ndim == 2:
        logo_bgr = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGR)
        alpha = np.ones((logo.shape[0], logo.shape[1]), dtype=np.float32)
    elif logo.shape[2] == 4:
        logo_bgr = logo[:, :, :3]
        alpha = logo[:, :, 3].astype(np.float32) / 255.0
    else:
        logo_bgr = logo
        alpha = np.ones((logo.shape[0], logo.shape[1]), dtype=np.float32)

    if max_width > 0 and logo_bgr.shape[1] > max_width:
        ratio = max_width / float(logo_bgr.shape[1])
        new_w = max(1, int(round(logo_bgr.shape[1] * ratio)))
        new_h = max(1, int(round(logo_bgr.shape[0] * ratio)))
        logo_bgr = cv2.resize(logo_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return logo_bgr, alpha


def _overlay_logo_top_right(
    frame: np.ndarray,
    logo_bgr: np.ndarray,
    logo_alpha: np.ndarray,
    *,
    margin_top: int,
    margin_right: int,
    opacity: float,
) -> None:
    h, w = frame.shape[:2]
    lh, lw = logo_bgr.shape[:2]
    if lh <= 0 or lw <= 0 or lh > h or lw > w:
        return

    x0 = max(0, w - lw - max(0, margin_right))
    y0 = max(0, max(0, margin_top))
    x1 = min(w, x0 + lw)
    y1 = min(h, y0 + lh)
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    logo_crop = logo_bgr[: y1 - y0, : x1 - x0]
    alpha_crop = logo_alpha[: y1 - y0, : x1 - x0]
    blend = np.clip(alpha_crop * float(opacity), 0.0, 1.0)[:, :, None]
    roi[:] = (blend * logo_crop + (1.0 - blend) * roi).astype(np.uint8)


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    inf_cfg = cfg["inference"]
    feature_cfg = cfg["feature"]
    model_cfg = cfg["models"]
    cam_cfg = cfg["camera"]
    nms_cfg = cfg.get("nms", {})

    letters_model = joblib.load(Path(model_cfg["letters_model"]))
    letters_encoder = joblib.load(Path(model_cfg["letters_encoder"]))
    words_model = joblib.load(Path(model_cfg["words_model"]))
    words_encoder = joblib.load(Path(model_cfg["words_encoder"]))
    nms_enabled = bool(nms_cfg.get("enabled", False))
    nms_model = None
    nms_encoder = None
    nms_extractor = None
    nms_stabilizer = None
    nms_state = None

    if nms_enabled:
        nms_model_path = Path(nms_cfg.get("model", "models/nms_model.joblib"))
        nms_encoder_path = Path(nms_cfg.get("encoder", "models/nms_label_encoder.joblib"))
        if nms_model_path.exists() and nms_encoder_path.exists():
            nms_model = joblib.load(nms_model_path)
            nms_encoder = joblib.load(nms_encoder_path)
            nms_extractor = FaceNMSExtractor(
                static_image_mode=False,
                max_num_faces=int(nms_cfg.get("max_num_faces", 1)),
                min_detection_confidence=float(nms_cfg.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(nms_cfg.get("min_tracking_confidence", 0.5)),
            )
            nms_window = max(2, int(nms_cfg.get("stability_frames", 5)))
            nms_votes = max(2, int(round(nms_window * 0.7)))
            nms_stabilizer = PredictionStabilizer(window_size=nms_window, min_count=nms_votes)
        else:
            print(f"NMS disabled: missing {nms_model_path} or {nms_encoder_path}")
            nms_enabled = False

    mode = str(inf_cfg.get("mode", "LETTER")).upper()
    shared_conf = float(inf_cfg.get("conf_threshold", 0.45))
    shared_stable = int(inf_cfg.get("stability_frames", 4))
    mode_settings = {
        "LETTER": {
            "conf_threshold": float(inf_cfg.get("letter_conf_threshold", shared_conf)),
            "stability_frames": int(inf_cfg.get("letter_stability_frames", shared_stable)),
        },
        "WORD": {
            "conf_threshold": float(inf_cfg.get("word_conf_threshold", max(0.25, shared_conf - 0.10))),
            "stability_frames": int(inf_cfg.get("word_stability_frames", shared_stable + 1)),
        },
    }
    if mode not in mode_settings:
        mode = "LETTER"

    def _make_stabilizer(active_mode: str) -> PredictionStabilizer:
        window = max(2, int(mode_settings[active_mode]["stability_frames"]))
        min_votes = max(2, int(round(window * 0.7)))
        return PredictionStabilizer(window_size=window, min_count=min_votes)

    stabilizer = _make_stabilizer(mode)

    extractor = HandLandmarkExtractor(
        use_two_hands=feature_cfg["use_two_hands"],
        normalize=feature_cfg["normalize"],
        include_handedness=feature_cfg["include_handedness"],
        static_image_mode=False,
    )

    cap = cv2.VideoCapture(int(cam_cfg["index"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))
    flip_horizontal = bool(cam_cfg.get("flip_horizontal", True))
    window_name = str(cam_cfg.get("window_name", "FSL Realtime Recognizer"))
    fullscreen = bool(cam_cfg.get("fullscreen", True))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, int(cam_cfg["width"]), int(cam_cfg["height"]))

    if not cap.isOpened():
        extractor.close()
        raise RuntimeError("Unable to open webcam.")

    logo_cfg = cfg.get("logo", {})
    logo_enabled = bool(logo_cfg.get("enabled", True))
    logo_pack = None
    if logo_enabled:
        logo_pack = _load_logo(
            path_str=str(logo_cfg.get("path", "assets/project_kumpas_logo.jpg")),
            max_width=int(logo_cfg.get("max_width", 180)),
        )
        if logo_pack is None:
            print("Logo disabled: logo file not found or unreadable.")
            logo_enabled = False
    logo_margin_top = int(logo_cfg.get("margin_top", 12))
    logo_margin_right = int(logo_cfg.get("margin_right", 16))
    logo_opacity = float(logo_cfg.get("opacity", 0.95))

    committed_tokens: list[str] = []
    current_display = ""
    nms_display = "neutral"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            vector = extractor.extract(frame)
            current_pred_label = None
            confidence = 0.0
            hand_detected = vector is not None
            nms_conf = 0.0

            if nms_enabled and nms_model is not None and nms_encoder is not None and nms_extractor is not None:
                nms_vec = nms_extractor.extract(frame)
                nms_state = None
                if nms_vec is not None:
                    nms_label, nms_conf = _predict_label(nms_model, nms_encoder, nms_vec)
                    if nms_conf >= float(nms_cfg.get("conf_threshold", 0.55)):
                        nms_state = nms_label
                if nms_stabilizer is not None:
                    nms_stable = nms_stabilizer.update(nms_state)
                    if nms_stable is not None:
                        nms_display = nms_stable

            if vector is not None:
                if mode == "LETTER":
                    raw_label, confidence = _predict_with_fallback(letters_model, letters_encoder, vector, feature_cfg)
                else:
                    raw_label, confidence = _predict_with_fallback(words_model, words_encoder, vector, feature_cfg)
                mapped = _map_display_label(mode, raw_label)
                conf_threshold = float(mode_settings[mode]["conf_threshold"])
                current_pred_label = mapped if confidence >= conf_threshold else None

            stable = stabilizer.update(current_pred_label)
            if stable is not None:
                current_display = stable

            output_text = " ".join(committed_tokens)
            nms_is_raised = str(nms_display).lower() in {
                value.lower()
                for value in nms_cfg.get(
                    "raised_labels",
                    ["eyebrow_raise", "raised", "eyebrows_raised"],
                )
            }
            decorated_current = f"{current_display}?" if (current_display and nms_is_raised) else (current_display or "-")

            cv2.putText(frame, f"Mode: {mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Current: {decorated_current}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            current_conf_threshold = float(mode_settings[mode]["conf_threshold"])
            current_stability = int(mode_settings[mode]["stability_frames"])
            cv2.putText(frame, f"Thresh: {current_conf_threshold:.2f}  Stable: {current_stability}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)
            cv2.putText(frame, f"Hand: {'YES' if hand_detected else 'NO'}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            if nms_enabled:
                cv2.putText(frame, f"NMS: {nms_display} ({nms_conf:.2f})", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 120), 2)
                cv2.putText(frame, f"Output: {output_text}", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"Output: {output_text}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(
                frame,
                "f=fullscreen m=mode [/] thresh -=stable +=stable n=nms on/off space=commit backspace=delete c=clear q=quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (200, 200, 200),
                1,
            )

            if logo_enabled and logo_pack is not None:
                _overlay_logo_top_right(
                    frame,
                    logo_pack[0],
                    logo_pack[1],
                    margin_top=logo_margin_top,
                    margin_right=logo_margin_right,
                    opacity=logo_opacity,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, int(cam_cfg["width"]), int(cam_cfg["height"]))
            if key == ord("m"):
                mode = "WORD" if mode == "LETTER" else "LETTER"
                stabilizer = _make_stabilizer(mode)
                current_display = ""
            elif key == ord("n"):
                nms_enabled = not nms_enabled if (nms_model is not None and nms_encoder is not None and nms_extractor is not None) else nms_enabled
            elif key == ord("["):
                mode_settings[mode]["conf_threshold"] = max(0.10, float(mode_settings[mode]["conf_threshold"]) - 0.05)
            elif key == ord("]"):
                mode_settings[mode]["conf_threshold"] = min(0.95, float(mode_settings[mode]["conf_threshold"]) + 0.05)
            elif key == ord("-"):
                mode_settings[mode]["stability_frames"] = max(2, int(mode_settings[mode]["stability_frames"]) - 1)
                stabilizer = _make_stabilizer(mode)
                current_display = ""
            elif key in (ord("="), ord("+")):
                mode_settings[mode]["stability_frames"] = min(20, int(mode_settings[mode]["stability_frames"]) + 1)
                stabilizer = _make_stabilizer(mode)
                current_display = ""
            elif key == ord("c"):
                committed_tokens.clear()
            elif key == ord(" "):
                if current_display:
                    token = f"{current_display}?" if nms_is_raised else current_display
                    committed_tokens.append(token)
                    stabilizer.reset()
                    current_display = ""
            elif key in (8, 127):
                if committed_tokens:
                    committed_tokens.pop()
    finally:
        cap.release()
        extractor.close()
        if nms_extractor is not None:
            nms_extractor.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
