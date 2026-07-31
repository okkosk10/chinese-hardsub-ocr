import json
from hardsub_ocr.utils.file_utils import atomic_write_json


def test_atomic_json(tmp_path):
    path = tmp_path / "상태.json"
    atomic_write_json(path, {"interrupted": True, "text": "中文"})
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == "中文"
    assert not path.with_suffix(".json.tmp").exists()

