import cv2
import numpy as np


def change_score(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 1.0
    a = cv2.resize(previous, (160, 48), interpolation=cv2.INTER_AREA)
    b = cv2.resize(current, (160, 48), interpolation=cv2.INTER_AREA)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(a, b)
    mean = float(np.mean(diff)) / 255.0
    active = float(np.count_nonzero(diff > 22)) / diff.size
    return min(1.0, mean * 0.45 + active * 0.55)

