import numpy as np

from fsl.features.nms import extract_nms_features


def _mock_face_landmarks() -> np.ndarray:
    points = np.zeros((468, 3), dtype=np.float32)

    points[33] = [0.30, 0.45, 0.0]
    points[263] = [0.70, 0.45, 0.0]
    points[1] = [0.50, 0.55, 0.0]
    points[152] = [0.50, 0.90, 0.0]

    points[159] = [0.37, 0.46, 0.0]
    points[145] = [0.37, 0.50, 0.0]
    points[386] = [0.63, 0.46, 0.0]
    points[374] = [0.63, 0.50, 0.0]

    points[105] = [0.40, 0.38, 0.0]
    points[70] = [0.32, 0.39, 0.0]
    points[334] = [0.60, 0.38, 0.0]
    points[300] = [0.68, 0.39, 0.0]

    points[13] = [0.50, 0.62, 0.0]
    points[14] = [0.50, 0.66, 0.0]

    return points


def test_extract_nms_features_shape_and_finite() -> None:
    features = extract_nms_features(_mock_face_landmarks())
    assert features.shape == (10,)
    assert np.all(np.isfinite(features))


def test_extract_nms_features_raises_on_bad_shape() -> None:
    bad = np.zeros((20, 3), dtype=np.float32)
    try:
        extract_nms_features(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass
