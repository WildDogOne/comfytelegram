import pytest


@pytest.fixture(autouse=True)
def _no_staged_sam_encoder(monkeypatch, tmp_path):
    """`Settings.sam_encoder_path` defaults to a real file under `models/`,
    which turns "✨ Smart" select on for any mask-editor test once that file
    is staged locally — running the encoder and a background upload to a
    fake relay. Point it somewhere empty; tests that want it set it."""
    monkeypatch.setenv("SAM_ENCODER_PATH", str(tmp_path / "no-encoder.onnx"))


@pytest.fixture(autouse=True)
def _no_staged_pose_detector(monkeypatch, tmp_path):
    """Same for `/pose`'s DWPose detector (`Settings.pose_detector_path`/
    `pose_estimator_path`): a locally staged model would make pose tests
    run real detection on their dummy images."""
    monkeypatch.setenv("POSE_DETECTOR_PATH", str(tmp_path / "no-detector.onnx"))
    monkeypatch.setenv("POSE_ESTIMATOR_PATH", str(tmp_path / "no-estimator.onnx"))
