import json
import pytest
from hardsub_ocr.config import Crop, UserSettings


def test_crop_parse(): assert Crop.parse("400, 700,1120,180") == Crop(400, 700, 1120, 180)


def test_crop_invalid():
    with pytest.raises(ValueError): Crop.parse("1,2,3")


def test_old_settings_file_uses_new_quality_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"interval": 1.0, "recent_video": "old.mp4"}), encoding="utf-8")
    settings = UserSettings.load(path)
    assert settings.interval == 1.0
    assert settings.transition_settle_seconds == .15
    assert settings.candidate_consensus_enabled is True
    assert settings.processing_mode == "fast"
    assert settings.auxiliary_fallback_enabled is False
