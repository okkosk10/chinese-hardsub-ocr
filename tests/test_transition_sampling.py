from pathlib import Path

import numpy as np

from hardsub_ocr.config import Crop, OcrConfig
from hardsub_ocr.models import OcrResult
from hardsub_ocr.pipeline import OcrPipeline
from hardsub_ocr.video.ffmpeg_reader import FFmpegFrameReader


class FakeEngine:
    name = "fake"

    def recognize(self, image):
        return OcrResult("稳定字幕", "稳定字幕", .9, raw_lines=["稳定字幕"])


class FakeAuxReader:
    runs = 0

    def __init__(self, path, start, end, crop, interval, threads):
        self.start, self.interval = start, interval

    def frames(self):
        type(self).runs += 1
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        for index in range(3):
            yield index, self.start + index * self.interval, frame

    def stop(self):
        pass


def test_continuous_change_debounces_auxiliary_ffmpeg(monkeypatch):
    FakeAuxReader.runs = 0
    monkeypatch.setattr("hardsub_ocr.pipeline.FFmpegFrameReader", FakeAuxReader)
    config = OcrConfig(Path("not-opened.mp4"), 600, 610, Crop(0, 0, 40, 20))
    pipeline = OcrPipeline(config, FakeEngine())
    frame = np.zeros((20, 40, 3), dtype=np.uint8)

    first = pipeline._collect_candidates(600.0, frame)
    second = pipeline._collect_candidates(600.2, frame)

    assert FakeAuxReader.runs == 1
    assert pipeline.auxiliary_ffmpeg_runs == 1
    assert pipeline.transition_debounce_skips == 1
    assert first[0].initial_transition_frame is True
    assert first[0].timestamp == 600.0
    assert len(second) == 1


def test_reader_timestamp_is_absolute_for_late_start():
    reader = FFmpegFrameReader(Path("not-opened.mp4"), 600, 610, Crop(0, 0, 40, 20), .5)
    assert reader.timestamp_for_index(0) == 600.0
    assert reader.timestamp_for_index(3) == 601.5
