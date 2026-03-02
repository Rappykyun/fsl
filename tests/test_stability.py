from fsl.inference.predictor import PredictionStabilizer


def test_prediction_stabilizer_majority_threshold() -> None:
    stabilizer = PredictionStabilizer(window_size=4, min_count=3)

    assert stabilizer.update("A") is None
    assert stabilizer.update("A") is None
    assert stabilizer.update("B") is None
    assert stabilizer.update("A") == "A"


def test_prediction_stabilizer_reset() -> None:
    stabilizer = PredictionStabilizer(window_size=3, min_count=3)
    stabilizer.update("A")
    stabilizer.update("A")
    stabilizer.reset()
    assert stabilizer.update("A") is None
