from __future__ import annotations

import time
from typing import Any
import numpy as np

from hardsub_ocr.models import OcrResult
from hardsub_ocr.subtitle.text_normalizer import merge_ocr_lines, normalize_text


class RapidOcrEngine:
    name = "rapidocr-onnxruntime"

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("RapidOCR이 설치되지 않았습니다. setup.ps1을 실행하세요.") from exc
        self._engine = RapidOCR()

    def recognize(self, image: np.ndarray) -> OcrResult:
        started = time.perf_counter()
        raw, _ = self._engine(image)
        lines: list[tuple[float, float, str, float, Any]] = []
        for item in raw or []:
            if len(item) < 3:
                continue
            box, text, confidence = item[0], str(item[1]), float(item[2])
            x = min(float(point[0]) for point in box)
            y = min(float(point[1]) for point in box)
            lines.append((y, x, text, confidence, box))
        lines.sort(key=lambda value: (round(value[0] / 20), value[1]))
        raw_lines = [value[2].strip() for value in lines if value[2].strip()]
        before, text, overlaps = merge_ocr_lines(raw_lines)
        confidence = sum(x[3] for x in lines) / len(lines) if lines else 0.0
        return OcrResult(text, normalize_text(text), confidence, raw, [x[4] for x in lines],
                         (time.perf_counter() - started) * 1000, raw_lines, before, text, overlaps)
