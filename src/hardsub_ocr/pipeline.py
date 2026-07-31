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
from hardsub_ocr.models import OcrCandidate, OcrEvent, Progress, SubtitleSegment
from hardsub_ocr.ocr.base import OcrEngine
from hardsub_ocr.ocr.rapidocr_engine import RapidOcrEngine
from hardsub_ocr.subtitle.segment_builder import SegmentBuilder
from hardsub_ocr.subtitle.candidate_selector import CandidateSelection, select_candidate
from hardsub_ocr.subtitle.srt_writer import write_srt
from hardsub_ocr.subtitle.text_normalizer import merge_ocr_lines, normalize_text
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
        self.aux_reader: FFmpegFrameReader | None = None
        self.events: list[OcrEvent] = []
        self.segments: list[SubtitleSegment] = []
        self.progress = Progress(total_duration=config.end_time - config.start_time)
        self._started_iso = ""

    def cancel(self) -> None:
        self.cancel_event.set()
        if self.reader:
            self.reader.stop()
        if self.aux_reader:
            self.aux_reader.stop()

    def run(self) -> tuple[Path, Path, Path]:
        video = probe_video(self.config.input_path)
        self.config.end_time = min(self.config.end_time, video.duration)
        self.config.validate(video.width, video.height)
        srt_path, json_path, log_path = output_paths(self.config.input_path, self.config.output_dir)
        self._started_iso = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        builder = SegmentBuilder(self.config.similarity_threshold, self.config.short_similarity_threshold,
                                 self.config.min_duration, self.config.max_duration,
                                 self.config.blank_tolerance, self.config.end_grace,
                                 self.config.empty_confirmation_count, self.config.empty_confirmation_seconds)
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
                        event.transition_start = timestamp
                        previous_text = builder.current.text if builder.current else (
                            builder.segments[-1].text if builder.segments else "")
                        candidates = self._collect_candidates(timestamp, frame)
                        selection = self._select(candidates, previous_text)
                        self._record_selection(event, candidates, selection)
                        event.processing_time_ms = sum(candidate.processing_time_ms for candidate in candidates)
                        if selection.selected is None:
                            empty_candidates = candidates or [OcrCandidate(timestamp, "", "", 0.0, self.config.preprocess_mode)]
                            for candidate in empty_candidates:
                                event.segment_action, event.previous_similarity_score = builder.add(
                                    candidate.timestamp, "", 0.0, frame_index)
                        elif not selection.confirmed:
                            event.segment_action = "hold_unconfirmed"
                        else:
                            chosen = selection.selected
                            event.detected_text = chosen.text
                            event.normalized_text = chosen.normalized_text
                            event.confidence = chosen.confidence
                            event.segment_action, event.previous_similarity_score = builder.add(
                                timestamp, chosen.normalized_text, chosen.confidence, frame_index)
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
            if self.aux_reader:
                self.aux_reader.stop()

    def _collect_candidates(self, transition_timestamp: float, fallback_frame) -> list[OcrCandidate]:
        read_count = self.config.candidate_frame_count
        recognition_count = read_count
        if self.config.processing_mode == "fast":
            recognition_count = min(2, read_count)
        start = transition_timestamp + self.config.transition_settle_seconds
        end = min(self.config.end_time, start + self.config.candidate_window_seconds)
        frames: list[tuple[float, object]] = []
        if end > start + 0.01:
            auxiliary_interval = min(0.2, max(0.1, self.config.candidate_window_seconds / max(read_count, 1)))
            self.aux_reader = FFmpegFrameReader(self.config.input_path, start, end, self.config.crop,
                                                auxiliary_interval, self.config.ffmpeg_threads)
            for _, timestamp, frame in self.aux_reader.frames():
                if self.cancel_event.is_set():
                    break
                frames.append((timestamp, frame))
                if len(frames) >= read_count:
                    # Continue consuming the very small window so FFmpeg exits cleanly,
                    # but do not retain or OCR additional frames.
                    continue
            frames = frames[:read_count]
            self.aux_reader = None
        if not frames and not self.cancel_event.is_set():
            frames = [(transition_timestamp, fallback_frame)]
        candidates = [self._recognize_frame(timestamp, frame, index)
                      for index, (timestamp, frame) in enumerate(frames[:recognition_count])]
        if candidates and all(not candidate.normalized_text for candidate in candidates):
            for index, (timestamp, frame) in enumerate(frames[recognition_count:], recognition_count):
                candidates.append(self._recognize_frame(timestamp, frame, index))
        return candidates

    def _recognize_frame(self, timestamp: float, frame, candidate_index: int) -> OcrCandidate:
        modes = [self.config.preprocess_mode]
        if self.config.processing_mode == "precise":
            modes = ["original", "gray2x"]
        recognized: list[tuple[object, str, object]] = []
        for mode in dict.fromkeys(modes):
            prepared = preprocess(frame, mode)
            result = self.engine.recognize(prepared)  # type: ignore[union-attr]
            self.progress.ocr_runs += 1
            recognized.append((result, mode, prepared))
        result, mode, prepared = max(recognized, key=lambda item: item[0].confidence)
        raw_lines = list(result.raw_lines) if result.raw_lines else [line for line in result.text.splitlines() if line]
        if raw_lines:
            before, after, overlaps = merge_ocr_lines(raw_lines, self.config.line_overlap_max_chars)
            if not self.config.line_overlap_dedup_enabled:
                after, overlaps = "\n".join(raw_lines), []
        else:
            before, after, overlaps = result.text, result.text, []
        normalized = normalize_text(after, deduplicate_lines=False)
        candidate = OcrCandidate(timestamp, after, normalized, result.confidence, mode, result.processing_time_ms,
                                 raw_lines=raw_lines, joined_text_before_dedup=before,
                                 joined_text_after_dedup=after, deduplicated_overlap=overlaps)
        if self.config.save_debug_images:
            debug_dir = self.config.output_dir / "debug" / "candidates"
            debug_dir.mkdir(parents=True, exist_ok=True)
            base = f"{timestamp:012.3f}_{candidate_index}_{mode}"
            original_path, processed_path = debug_dir / f"{base}_crop.jpg", debug_dir / f"{base}_processed.jpg"
            cv2.imencode(".jpg", frame)[1].tofile(str(original_path))
            cv2.imencode(".jpg", prepared)[1].tofile(str(processed_path))
            candidate.original_image_path, candidate.preprocessed_image_path = str(original_path), str(processed_path)
        return candidate

    def _select(self, candidates: list[OcrCandidate], previous_text: str) -> CandidateSelection:
        if self.config.candidate_consensus_enabled:
            return select_candidate(candidates, previous_text, self.config.candidate_consensus_threshold,
                                    self.config.suspicious_suffix_removal_enabled,
                                    self.config.unstable_suffix_max_chars)
        nonempty = [candidate for candidate in candidates if candidate.normalized_text]
        if not nonempty:
            return CandidateSelection(None, candidates, "all_candidates_empty", 0.0, confirmed=False)
        selected = max(nonempty, key=lambda candidate: candidate.confidence)
        return CandidateSelection(selected, [item for item in candidates if item is not selected],
                                  "highest_confidence_consensus_disabled", 0.0,
                                  confirmed=selected.confidence >= 0.75)

    @staticmethod
    def _record_selection(event: OcrEvent, candidates: list[OcrCandidate], selection: CandidateSelection) -> None:
        event.candidates = [candidate.to_dict() for candidate in candidates]
        event.selected_candidate = selection.selected.to_dict() if selection.selected else None
        event.rejected_candidates = [candidate.to_dict() for candidate in selection.rejected]
        event.selection_reason = selection.reason
        event.removed_unstable_suffix = selection.removed_unstable_suffix
        event.consensus_score = selection.consensus_score
        if selection.selected:
            event.transition_mix_detected = selection.selected.transition_mix_detected
            event.matched_previous_fragment = selection.selected.matched_previous_fragment
            event.remaining_new_fragment = selection.selected.remaining_new_fragment

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
                "transition_settle_seconds": self.config.transition_settle_seconds,
                "candidate_window_seconds": self.config.candidate_window_seconds,
                "candidate_frame_count": self.config.candidate_frame_count,
                "candidate_consensus_enabled": self.config.candidate_consensus_enabled,
                "line_overlap_dedup_enabled": self.config.line_overlap_dedup_enabled,
                "suspicious_suffix_removal_enabled": self.config.suspicious_suffix_removal_enabled,
                "line_overlap_max_chars": self.config.line_overlap_max_chars,
                "candidate_consensus_threshold": self.config.candidate_consensus_threshold,
                "unstable_suffix_max_chars": self.config.unstable_suffix_max_chars,
                "empty_confirmation_count": self.config.empty_confirmation_count,
                "empty_confirmation_seconds": self.config.empty_confirmation_seconds,
                "processing_mode": self.config.processing_mode,
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
