import numpy as np

from fsl.features.landmarks import build_feature_vector, feature_vector_length, normalize_hand_landmarks


def test_feature_vector_zero_padding_is_deterministic() -> None:
    vec1 = build_feature_vector([], [], use_two_hands=True, normalize=True, include_handedness=True)
    vec2 = build_feature_vector([], [], use_two_hands=True, normalize=True, include_handedness=True)

    assert vec1.shape[0] == feature_vector_length(use_two_hands=True, include_handedness=True)
    assert np.allclose(vec1, 0.0)
    assert np.array_equal(vec1, vec2)


def test_normalize_hand_landmarks_keeps_fixed_shape() -> None:
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    hand[9] = np.array([2.0, 1.0, 1.0], dtype=np.float32)
    hand[5] = np.array([1.5, 1.5, 1.0], dtype=np.float32)

    out = normalize_hand_landmarks(hand)
    assert out.shape == (21, 3)
    assert np.allclose(out[0], np.array([0.0, 0.0, 0.0]))
    assert np.isclose(out[9][0], 1.0)
