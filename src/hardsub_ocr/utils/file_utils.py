from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any


def output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    return tuple(output_dir / f"{stem}.zh-ocr.{ext}" for ext in ("srt", "json", "log"))  # type: ignore[return-value]


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def safe_filename(text: str, limit: int = 50) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).strip(" .")
    return (cleaned or "empty")[:limit]

