from __future__ import annotations

from typing import Protocol
import numpy as np

from hardsub_ocr.models import OcrResult


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OcrResult: ...

