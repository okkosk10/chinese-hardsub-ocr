from pathlib import Path
import json

import numpy as np

from hardsub_ocr.config import Crop, OcrConfig
from hardsub_ocr.models import OcrResult, VideoInfo
from hardsub_ocr.pipeline import OcrPipeline
from hardsub_ocr.video.ffmpeg_reader import FFmpegFrameReader


class FakeEngine:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        text = "第二句" if float(np.mean(image)) > 100 else "第一句"
        return OcrResult(text, text, .9, raw_lines=[text])


class StreamingReader:
    constructions = 0
    runs = 0

    def __init__(self, path, start, end, crop, interval, threads):
        type(self).constructions += 1
        self.start, self.end, self.interval = start, end, interval

    def frames(self):
        type(self).runs += 1
        index, timestamp = 0, self.start
        while timestamp < self.end - 1e-9:
            value = 0 if timestamp < self.start + (self.end - self.start) / 2 else 255
            yield index, timestamp, np.full((20, 40, 3), value, dtype=np.uint8)
            index += 1
            timestamp = self.start + index * self.interval

    def stop(self):
        pass


def configure_stream_mocks(monkeypatch, duration: float):
    StreamingReader.constructions = StreamingReader.runs = 0
    monkeypatch.setattr("hardsub_ocr.pipeline.FFmpegFrameReader", StreamingReader)
    monkeypatch.setattr("hardsub_ocr.pipeline.probe_video", lambda _: VideoInfo(40, 20, duration, 30))


def test_continuous_changes_use_one_main_ffmpeg_process(monkeypatch, tmp_path):
    configure_stream_mocks(monkeypatch, 601.0)
    def alternating_frames(self):
        type(self).runs += 1
        index, timestamp = 0, self.start
        while timestamp < self.end - 1e-9:
            yield index, timestamp, np.full((20, 40, 3), 255 * (index % 2), dtype=np.uint8)
            index += 1
            timestamp = self.start + index * self.interval
    monkeypatch.setattr(StreamingReader, "frames", alternating_frames)
    source = tmp_path / "input.mp4"; source.touch()
    config = OcrConfig(source, 600, 601, Crop(0, 0, 40, 20), tmp_path / "out",
                       interval=.2, change_threshold=.001)
    pipeline = OcrPipeline(config, FakeEngine())
    pipeline.run()

    assert StreamingReader.constructions == 1
    assert StreamingReader.runs == 1
    assert pipeline.ffmpeg_process_count == 1
    assert pipeline.auxiliary_ffmpeg_runs == 0
    assert pipeline.transition_debounce_skips >= 1
    metadata = json.loads((tmp_path / "out" / "input.zh-ocr.json").read_text(encoding="utf-8"))["metadata"]
    assert metadata["ffmpeg_process_count"] == 1
    assert metadata["auxiliary_ffmpeg_runs"] == 0
    assert metadata["change_detections"] >= 1
    assert metadata["candidate_collections"] >= 1
    assert metadata["total_processing_seconds"] >= 0
    assert metadata["processing_ratio"] >= 0


def test_timestamp_ocr_cache_prevents_duplicate_recognition(tmp_path):
    engine = FakeEngine()
    config = OcrConfig(tmp_path / "unused.mp4", 0, 1, Crop(0, 0, 40, 20))
    pipeline = OcrPipeline(config, engine)
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    first = pipeline._recognize_frame(10.0, frame, 0, initial_transition=True)
    second = pipeline._recognize_frame(10.0, frame, 1)

    assert first.normalized_text == second.normalized_text
    assert engine.calls == 1
    assert pipeline.ocr_cache_hits == 1


def test_thirty_second_content_regression_uses_single_stream(monkeypatch, tmp_path):
    configure_stream_mocks(monkeypatch, 30.0)
    source = tmp_path / "sample.mp4"; source.touch()
    pipeline = OcrPipeline(OcrConfig(source, 0, 30, Crop(0, 0, 40, 20), tmp_path / "out"), FakeEngine())
    pipeline.run()

    texts = [segment.text for segment in pipeline.segments]
    assert texts == ["第一句", "第二句"]
    assert pipeline.ffmpeg_process_count == 1
    assert pipeline.auxiliary_ffmpeg_runs == 0


def test_reader_timestamp_is_absolute_for_late_start():
    reader = FFmpegFrameReader(Path("not-opened.mp4"), 600, 610, Crop(0, 0, 40, 20), .5)
    assert reader.timestamp_for_index(0) == 600.0
    assert reader.timestamp_for_index(3) == 601.5


def test_ring_buffer_keeps_only_recent_second(tmp_path):
    pipeline = OcrPipeline(OcrConfig(tmp_path / "unused.mp4", 0, 3, Crop(0, 0, 40, 20)), FakeEngine())
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    for timestamp in (0.0, .4, .8, 1.2, 1.6):
        pipeline._remember_frame(timestamp, frame.copy())
    assert [timestamp for timestamp, _ in pipeline.frame_ring] == [.8, 1.2, 1.6]
