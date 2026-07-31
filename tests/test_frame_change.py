import numpy as np
from hardsub_ocr.detection.frame_change import change_score


def test_same_frame_is_zero():
    frame = np.zeros((60, 200, 3), np.uint8)
    assert change_score(frame, frame) == 0


def test_different_frame_changes():
    assert change_score(np.zeros((60, 200, 3), np.uint8), np.full((60, 200, 3), 255, np.uint8)) > .9

