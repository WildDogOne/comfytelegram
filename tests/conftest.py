import pytest


@pytest.fixture(autouse=True)
def _no_staged_sam_encoder(monkeypatch, tmp_path):
    """`Settings.sam_encoder_path` defaults to a real file under `models/`,
    which turns "✨ Smart" select on for any mask-editor test once that file
    is staged locally — running the encoder and a background upload to a
    fake relay. Point it somewhere empty; tests that want it set it."""
    monkeypatch.setenv("SAM_ENCODER_PATH", str(tmp_path / "no-encoder.onnx"))
