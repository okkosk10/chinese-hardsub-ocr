from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Callable

import cv2

from hardsub_ocr.config import OcrConfig
from hardsub_ocr.detection.frame_change import change_score
from hardsub_ocr.detection.image_preprocessor import preprocess
from hardsub_ocr.models import OcrEvent, Progress, SubtitleSegment
from hardsub_ocr.ocr.base import OcrEngine
from hardsub_ocr.ocr.rapidocr_engine import RapidOcrEngine
from hardsub_ocr.subtitle.segment_builder import SegmentBuilder
from hardsub_ocr.subtitle.srt_writer import write_srt
from hardsub_ocr.utils.file_utils import atomic_write_json, output_paths, safe_filename
from hardsub_ocr.video.ffmpeg_reader import FFmpegFrameReader
from hardsub_ocr.video.video_probe import probe_video

ProgressCallback = Callable[[Progress, OcrEvent | None], None]


class OcrPipeline:
    def __init__(self, config: OcrConfig, engine: OcrEngine | None = None,
                 callback: ProgressCallback | None = None) -> None:
        self.config = config
        self.engine = engine
        self.callback = callback
        self.cancel_event = threading.Event()
        self.reader: FFmpegFrameReader | None = None
        self.events: list[OcrEvent] = []
        self.segments: list[SubtitleSegment] = []
        self.progress = Progress(total_duration=config.end_time - config.start_time)
        self._started_iso = ""

    def cancel(self) -> None:
        self.cancel_event.set()
        if self.reader:
            self.reader.stop()

    def run(self) -> tuple[Path, Path, Path]:
        video = probe_video(self.config.input_path)
        self.config.end_time = min(self.config.end_time, video.duration)
        self.config.validate(video.width, video.height)
        srt_path, json_path, log_path = output_paths(self.config.input_path, self.config.output_dir)
        self._started_iso = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        builder = SegmentBuilder(self.config.similarity_threshold, self.config.short_similarity_threshold,
                                 self.config.min_duration, self.config.max_duration,
                                 self.config.blank_tolerance, self.config.end_grace)
        previous = None
        self.reader = FFmpegFrameReader(self.config.input_path, self.config.start_time, self.config.end_time,
                                        self.config.crop, self.config.interval, self.config.ffmpeg_threads)
        try:
            if self.engine is None:
                self.engine = RapidOcrEngine()
            for frame_index, timestamp, frame in self.reader.frames():
                if self.cancel_event.is_set():
                    break
                score = change_score(previous, frame)
                previous = frame.copy()
                event = OcrEvent(frame_index, timestamp, frame_change_score=score, crop=str(self.config.crop),
                                 preprocessing_mode=self.config.preprocess_mode)
                if frame_index and score < self.config.change_threshold:
                    self.progress.ocr_skips += 1
                    event.segment_action = "skip_unchanged"
                else:
                    try:
                        result = self.engine.recognize(preprocess(frame, self.config.preprocess_mode))
                        event.detected_text, event.normalized_text = result.text, result.normalized_text
                        event.confidence, event.processing_time_ms = result.confidence, result.processing_time_ms
                        event.segment_action, event.previous_similarity_score = builder.add(
                            timestamp, result.normalized_text, result.confidence, frame_index)
                        self.progress.ocr_runs += 1
                        if self.config.save_debug_images and self._should_debug(event):
                            debug_dir = self.config.output_dir / "debug"
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            name = f"{timestamp:012.3f}_{event.confidence:.2f}_{safe_filename(event.normalized_text)}.jpg"
                            target = debug_dir / name
                            cv2.imencode(".jpg", frame)[1].tofile(str(target))
                            event.debug_image_path = str(target)
                    except Exception as exc:
                        event.error = str(exc)
                        event.segment_action = "error"
                        logging.getLogger("hardsub_ocr").exception("OCR frame %s failed", frame_index)
                self.events.append(event)
                self.progress.frames += 1
                self.progress.current_time = timestamp
                self.progress.segments = len(builder.segments) + int(builder.current is not None)
                self.progress.elapsed = time.monotonic() - started
                done = max(self.config.interval, timestamp - self.config.start_time + self.config.interval)
                self.progress.eta = self.progress.elapsed * max(0, self.progress.total_duration - done) / done
                if self.callback:
                    self.callback(self.progress, event)
                if frame_index % 20 == 0:
                    self._save(json_path, srt_path, builder, video, interrupted=False, finished=False)
            self.segments = builder.finish(min(self.config.end_time, self.progress.current_time + self.config.interval))
            self._save(json_path, srt_path, builder, video, self.cancel_event.is_set(), finished=True)
            return srt_path, json_path, log_path
        except BaseException:
            self.segments = builder.finish(max(self.config.start_time, self.progress.current_time))
            self._save(json_path, srt_path, builder, video, True, finished=True)
            raise
        finally:
            if self.reader:
                self.reader.stop()

    def _save(self, json_path: Path, srt_path: Path, builder: SegmentBuilder, video: object,
              interrupted: bool, finished: bool) -> None:
        segments = list(builder.segments) + ([builder.current] if builder.current else [])
        write_srt(srt_path, segments)
        payload = {
            "metadata": {
                "input_file": str(self.config.input_path), "resolution": [video.width, video.height],
                "video_duration": video.duration, "start_time": self.config.start_time,
                "end_time": self.config.end_time, "crop": str(self.config.crop),
                "interval": self.config.interval, "change_threshold": self.config.change_threshold,
                "similarity_threshold": self.config.similarity_threshold,
                "ocr_engine": getattr(self.engine, "name", type(self.engine).__name__),
                "run_started_at": self._started_iso,
                "run_finished_at": datetime.now(timezone.utc).isoformat() if finished else None,
                "interrupted": interrupted,
            },
            "progress": asdict(self.progress),
            "segments": [segment.to_dict() for segment in segments],
            "events": [event.to_dict() for event in self.events],
        }
        atomic_write_json(json_path, payload)

    @staticmethod
    def _should_debug(event: OcrEvent) -> bool:
        return bool(event.error or event.confidence < 0.5 or event.segment_action in {"start", "replace"})

