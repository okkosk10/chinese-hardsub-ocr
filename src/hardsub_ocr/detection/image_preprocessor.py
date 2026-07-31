from __future__ import annotations

import cv2
import numpy as np


MODES = ("original", "gray2x", "gray3x", "contrast", "sharpen", "threshold", "adaptive", "denoise", "morph_close")


def preprocess(image: np.ndarray, mode: str = "gray2x") -> np.ndarray:
    if mode == "original":
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode in ("gray2x", "gray3x"):
        factor = 2 if mode == "gray2x" else 3
        return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    if mode == "contrast":
        return cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    if mode == "sharpen":
        return cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    if mode == "threshold":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if mode == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    if mode == "denoise":
        return cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    if mode == "morph_close":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    raise ValueError(f"지원하지 않는 전처리 모드: {mode}")

