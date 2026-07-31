from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
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
from hardsub_ocr.ocr.line_grouper import lines_are_visually_separate
from hardsub_ocr.subtitle.segment_builder import SegmentBuilder
from hardsub_ocr.subtitle.candidate_selector import (
    CandidateSelection, resolve_with_single_candidate_fallback, select_candidate,
)
from hardsub_ocr.subtitle.srt_writer import write_srt
from hardsub_ocr.subtitle.srt_cleaner import clean_segments, cleaned_srt_path
from hardsub_ocr.subtitle.text_normalizer import merge_ocr_lines, normalize_text
from hardsub_ocr.utils.file_utils import atomic_write_json, output_paths, safe_filename
from hardsub_ocr.video.ffmpeg_reader import FFmpegFrameReader
from hardsub_ocr.video.video_probe import probe_video

ProgressCallback = Callable[[Progress, OcrEvent | None], None]


@dataclass(slots=True)
class PendingTransition:
    frame_index: int
    timestamp: float
    frame: object
    event: OcrEvent
    candidates: list[OcrCandidate]


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
        self.auxiliary_ffmpeg_runs = 0
        self.auxiliary_decode_ms = 0.0
        self.fallback_used = 0
        self.transition_debounce_skips = 0
        self._last_aux_transition_at: float | None = None
        self.ocr_cache_hits = 0
        self.change_detections = 0
        self.candidate_collections = 0
        self.ffmpeg_process_count = 0
        self.frame_ring: deque[tuple[float, object]] = deque()
        self._ocr_cache: dict[tuple[float, str, str, bool], OcrCandidate] = {}
        self._run_started_monotonic = 0.0

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
        self._run_started_monotonic = time.monotonic()
        builder = SegmentBuilder(self.config.similarity_threshold, self.config.short_similarity_threshold,
                                 self.config.min_duration, self.config.max_duration,
                                 self.config.blank_tolerance, self.config.end_grace,
                                 self.config.empty_confirmation_count, self.config.empty_confirmation_seconds)
        previous_detection_frame = None
        next_detection_timestamp = self.config.start_time
        pending: PendingTransition | None = None
        stream_interval = min(self.config.interval, self.config.stream_frame_interval)
        self.reader = FFmpegFrameReader(self.config.input_path, self.config.start_time, self.config.end_time,
                                        self.config.crop, stream_interval, self.config.ffmpeg_threads)
        self.ffmpeg_process_count = 1
        try:
            if self.engine is None:
                self.engine = RapidOcrEngine()
            for frame_index, timestamp, frame in self.reader.frames():
                if self.cancel_event.is_set():
                    break
                self._remember_frame(timestamp, frame)

                completed_event: OcrEvent | None = None
                if pending is not None:
                    self._feed_pending(pending, timestamp, frame)
                    if self._pending_complete(pending, timestamp):
                        completed_event = self._finalize_pending(pending, builder, timestamp)
                        pending = None

                if timestamp + 1e-9 >= next_detection_timestamp:
                    score = change_score(previous_detection_frame, frame)
                    previous_detection_frame = frame.copy()
                    while next_detection_timestamp <= timestamp + 1e-9:
                        next_detection_timestamp += self.config.interval
                    if frame_index == 0 or score >= self.config.change_threshold:
                        self.change_detections += 1
                        if (self._last_aux_transition_at is not None
                                and timestamp - self._last_aux_transition_at < self.config.candidate_window_seconds):
                            self.transition_debounce_skips += 1
                            if pending is not None:
                                pending.event.transition_debounce_skips += 1
                        elif pending is None:
                            pending = self._start_pending(frame_index, timestamp, frame, score)
                            self._last_aux_transition_at = timestamp
                    else:
                        self.progress.ocr_skips += 1

                self.progress.frames += 1
                self.progress.current_time = timestamp
                self.progress.segments = len(builder.segments) + int(builder.current is not None)
                self.progress.elapsed = time.monotonic() - self._run_started_monotonic
                done = max(stream_interval, timestamp - self.config.start_time + stream_interval)
                self.progress.eta = self.progress.elapsed * max(0, self.progress.total_duration - done) / done
                if self.callback:
                    self.callback(self.progress, completed_event)
                if frame_index % 100 == 0:
                    self._save(json_path, srt_path, builder, video, interrupted=False, finished=False)
            if pending is not None:
                completed_event = self._finalize_pending(pending, builder, self.config.end_time)
                if self.callback:
                    self.callback(self.progress, completed_event)
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

    def _target_candidate_count(self) -> int:
        return min(2, self.config.candidate_frame_count) if self.config.processing_mode == "fast" else self.config.candidate_frame_count

    def _remember_frame(self, timestamp: float, frame) -> None:
        self.frame_ring.append((timestamp, frame))
        cutoff = timestamp - self.config.ring_buffer_seconds
        while self.frame_ring and self.frame_ring[0][0] < cutoff:
            self.frame_ring.popleft()
        for key in [key for key in self._ocr_cache if key[0] < cutoff]:
            self._ocr_cache.pop(key, None)

    def _start_pending(self, frame_index: int, timestamp: float, frame, score: float) -> PendingTransition:
        event = OcrEvent(frame_index, timestamp, frame_change_score=score, crop=str(self.config.crop),
                         preprocessing_mode=self.config.preprocess_mode, transition_start=timestamp)
        candidate = self._recognize_frame(timestamp, frame, 0, initial_transition=True)
        self.candidate_collections += 1
        return PendingTransition(frame_index, timestamp, frame.copy(), event, [candidate])

    def _feed_pending(self, pending: PendingTransition, timestamp: float, frame) -> None:
        if len(pending.candidates) >= self._target_candidate_count():
            return
        if timestamp + 1e-9 < pending.timestamp + self.config.transition_settle_seconds:
            return
        if any(abs(candidate.timestamp - timestamp) < 1e-6 for candidate in pending.candidates):
            return
        pending.candidates.append(self._recognize_frame(timestamp, frame, len(pending.candidates)))

    def _pending_complete(self, pending: PendingTransition, timestamp: float) -> bool:
        return (len(pending.candidates) >= self._target_candidate_count()
                or timestamp + 1e-9 >= pending.timestamp + self.config.candidate_window_seconds
                or self.cancel_event.is_set())

    def _finalize_pending(self, pending: PendingTransition, builder: SegmentBuilder,
                          finalized_at: float) -> OcrEvent:
        event = pending.event
        if (len(pending.candidates) < self._target_candidate_count()
                and self.config.auxiliary_fallback_enabled and not self.cancel_event.is_set()):
            runs_before, decode_before = self.auxiliary_ffmpeg_runs, self.auxiliary_decode_ms
            pending.candidates.extend(self._collect_auxiliary_fallback(pending))
            event.auxiliary_ffmpeg_runs = self.auxiliary_ffmpeg_runs - runs_before
            event.auxiliary_decode_ms = self.auxiliary_decode_ms - decode_before
        previous_text = builder.current.text if builder.current else (
            builder.segments[-1].text if builder.segments else "")
        selection = self._select(pending.candidates, previous_text)
        self._record_selection(event, pending.candidates, selection)
        event.processing_time_ms = sum(candidate.processing_time_ms for candidate in pending.candidates)
        chosen, fallback = resolve_with_single_candidate_fallback(selection)
        event.fallback_used = fallback
        if fallback:
            self.fallback_used += 1
        if chosen is None:
            for candidate in pending.candidates:
                event.segment_action, event.previous_similarity_score = builder.add(
                    candidate.timestamp, "", 0.0, pending.frame_index)
            if (pending.candidates
                    and finalized_at - pending.candidates[0].timestamp + 1e-9 >= self.config.empty_confirmation_seconds):
                event.segment_action, event.previous_similarity_score = builder.add(
                    finalized_at, "", 0.0, pending.frame_index)
        else:
            event.detected_text, event.normalized_text = chosen.text, chosen.normalized_text
            event.confidence = chosen.confidence
            event.segment_action, event.previous_similarity_score = builder.add(
                pending.timestamp, chosen.normalized_text, chosen.confidence, pending.frame_index)
        if self.config.save_debug_images and self._should_debug(event):
            debug_dir = self.config.output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            name = f"{pending.timestamp:012.3f}_{event.confidence:.2f}_{safe_filename(event.normalized_text)}.jpg"
            target = debug_dir / name
            cv2.imencode(".jpg", pending.frame)[1].tofile(str(target))
            event.debug_image_path = str(target)
        self.events.append(event)
        return event

    def _collect_auxiliary_fallback(self, pending: PendingTransition) -> list[OcrCandidate]:
        needed = self._target_candidate_count() - len(pending.candidates)
        if needed <= 0:
            return []
        start = max(pending.timestamp + self.config.transition_settle_seconds,
                    pending.candidates[-1].timestamp + self.config.stream_frame_interval)
        end = min(self.config.end_time, pending.timestamp + self.config.candidate_window_seconds)
        if end <= start:
            return []
        self.aux_reader = FFmpegFrameReader(self.config.input_path, start, end, self.config.crop,
                                            self.config.stream_frame_interval, self.config.ffmpeg_threads)
        self.auxiliary_ffmpeg_runs += 1
        self.ffmpeg_process_count += 1
        started = time.perf_counter()
        candidates: list[OcrCandidate] = []
        try:
            for _, timestamp, frame in self.aux_reader.frames():
                candidates.append(self._recognize_frame(timestamp, frame, len(pending.candidates) + len(candidates)))
                if len(candidates) >= needed or self.cancel_event.is_set():
                    break
        finally:
            self.auxiliary_decode_ms += (time.perf_counter() - started) * 1000
            self.aux_reader.stop()
            self.aux_reader = None
        return candidates

    def _recognize_frame(self, timestamp: float, frame, candidate_index: int,
                         initial_transition: bool = False) -> OcrCandidate:
        cache_key = (round(timestamp, 6), self.config.processing_mode,
                     self.config.preprocess_mode, self.config.line_overlap_dedup_enabled)
        cached = self._ocr_cache.get(cache_key)
        if cached is not None:
            self.ocr_cache_hits += 1
            return replace(cached, initial_transition_frame=initial_transition)
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
            visually_separate = (self.config.preserve_separated_lines
                                 and lines_are_visually_separate(result.line_boxes,
                                                                 self.config.line_separation_gap_ratio))
            if visually_separate or not self.config.line_overlap_dedup_enabled:
                before, after, overlaps = "\n".join(raw_lines), "\n".join(raw_lines), []
            else:
                before, after, overlaps = merge_ocr_lines(raw_lines, self.config.line_overlap_max_chars)
            if not self.config.line_overlap_dedup_enabled:
                after, overlaps = "\n".join(raw_lines), []
        else:
            before, after, overlaps = result.text, result.text, []
        normalized = normalize_text(after, deduplicate_lines=False)
        candidate = OcrCandidate(timestamp, after, normalized, result.confidence, mode, result.processing_time_ms,
                                 initial_transition,
                                 raw_lines=raw_lines, joined_text_before_dedup=before,
                                 joined_text_after_dedup=after, deduplicated_overlap=overlaps,
                                 line_boxes=list(result.line_boxes))
        if self.config.save_debug_images:
            debug_dir = self.config.output_dir / "debug" / "candidates"
            debug_dir.mkdir(parents=True, exist_ok=True)
            base = f"{timestamp:012.3f}_{candidate_index}_{mode}"
            original_path, processed_path = debug_dir / f"{base}_crop.jpg", debug_dir / f"{base}_processed.jpg"
            cv2.imencode(".jpg", frame)[1].tofile(str(original_path))
            cv2.imencode(".jpg", prepared)[1].tofile(str(processed_path))
            candidate.original_image_path, candidate.preprocessed_image_path = str(original_path), str(processed_path)
        self._ocr_cache[cache_key] = replace(candidate, initial_transition_frame=False)
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
        cleaned_segments, removed_segments = clean_segments(segments)
        cleaned_path = cleaned_srt_path(srt_path)
        write_srt(cleaned_path, cleaned_segments)
        total_processing_seconds = (time.monotonic() - self._run_started_monotonic
                                    if self._run_started_monotonic else 0.0)
        selected_duration = max(0.001, self.config.end_time - self.config.start_time)
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
                "stream_frame_interval": self.config.stream_frame_interval,
                "ring_buffer_seconds": self.config.ring_buffer_seconds,
                "auxiliary_fallback_enabled": self.config.auxiliary_fallback_enabled,
                "preserve_separated_lines": self.config.preserve_separated_lines,
                "line_separation_gap_ratio": self.config.line_separation_gap_ratio,
                "auxiliary_ffmpeg_runs": self.auxiliary_ffmpeg_runs,
                "auxiliary_decode_ms": self.auxiliary_decode_ms,
                "fallback_used": self.fallback_used,
                "transition_debounce_skips": self.transition_debounce_skips,
                "total_processing_seconds": total_processing_seconds,
                "processing_ratio": total_processing_seconds / selected_duration,
                "ocr_runs": self.progress.ocr_runs,
                "ocr_cache_hits": self.ocr_cache_hits,
                "change_detections": self.change_detections,
                "candidate_collections": self.candidate_collections,
                "ffmpeg_process_count": self.ffmpeg_process_count,
                "ocr_engine": getattr(self.engine, "name", type(self.engine).__name__),
                "run_started_at": self._started_iso,
                "run_finished_at": datetime.now(timezone.utc).isoformat() if finished else None,
                "interrupted": interrupted,
                "cleaned_srt": str(cleaned_path),
            },
            "progress": asdict(self.progress),
            "segments": [segment.to_dict() for segment in segments],
            "events": [event.to_dict() for event in self.events],
            "cleaning": {
                "cleaned_segment_count": len(cleaned_segments),
                "removed_segment_count": len(removed_segments),
                "removed_segments": removed_segments,
            },
        }
        atomic_write_json(json_path, payload)

    @staticmethod
    def _should_debug(event: OcrEvent) -> bool:
        return bool(event.error or event.confidence < 0.5 or event.segment_action in {"start", "replace"})
