from __future__ import annotations

from hardsub_ocr.models import SubtitleSegment
from hardsub_ocr.subtitle.text_normalizer import normalize_text
from hardsub_ocr.subtitle.text_similarity import is_same_text, similarity


class SegmentBuilder:
    def __init__(self, similarity_threshold: float = 82, short_threshold: float = 94,
                 min_duration: float = 0.35, max_duration: float = 8.0,
                 blank_tolerance: int = 1, end_grace: float = 0.15,
                 empty_confirmation_count: int | None = None,
                 empty_confirmation_seconds: float = 0.4) -> None:
        self.threshold, self.short_threshold = similarity_threshold, short_threshold
        self.min_duration, self.max_duration = min_duration, max_duration
        self.blank_tolerance, self.end_grace = blank_tolerance, end_grace
        self.empty_confirmation_count = blank_tolerance + 1 if empty_confirmation_count is None else empty_confirmation_count
        self.empty_confirmation_seconds = 0.0 if empty_confirmation_count is None else empty_confirmation_seconds
        self.segments: list[SubtitleSegment] = []
        self.current: SubtitleSegment | None = None
        self.blank_count = 0
        self.blank_started_at: float | None = None

    def add(self, timestamp: float, text: str, confidence: float, frame_index: int) -> tuple[str, float]:
        text = normalize_text(text)
        previous_similarity = similarity(self.current.text, text) if self.current and text else 0.0
        if not text:
            if self.current:
                self.blank_count += 1
                if self.blank_started_at is None:
                    self.blank_started_at = timestamp
                enough_time = timestamp - self.blank_started_at + 1e-9 >= self.empty_confirmation_seconds
                if self.blank_count >= self.empty_confirmation_count and enough_time:
                    self._finish(max(self.current.last_seen_timestamp + self.end_grace, self.current.start_time + self.min_duration))
                    return "finish_blank", previous_similarity
            return "blank", previous_similarity
        self.blank_count = 0
        self.blank_started_at = None
        if self.current is None:
            self._start(timestamp, text, confidence, frame_index)
            return "start", previous_similarity
        if is_same_text(self.current.text, text, self.threshold, self.short_threshold):
            self.current.end_time = timestamp
            self.current.last_seen_timestamp = timestamp
            self.current.source_frames.append(frame_index)
            if confidence > self.current.confidence:
                self.current.text, self.current.confidence = text, confidence
            if timestamp - self.current.start_time >= self.max_duration:
                old_text, old_conf = self.current.text, self.current.confidence
                self._finish(timestamp)
                self._start(timestamp, old_text, old_conf, frame_index)
                return "split_max", previous_similarity
            return "extend", previous_similarity
        self._finish(max(timestamp - self.end_grace, self.current.start_time + self.min_duration))
        self._start(timestamp, text, confidence, frame_index)
        return "replace", previous_similarity

    def _start(self, timestamp: float, text: str, confidence: float, frame_index: int) -> None:
        start = max(timestamp, self.segments[-1].end_time) if self.segments else timestamp
        self.current = SubtitleSegment(len(self.segments) + 1, start, start, text, confidence,
                                       [frame_index], timestamp, timestamp)

    def _finish(self, end_time: float) -> None:
        if not self.current:
            return
        self.current.end_time = max(end_time, self.current.start_time + self.min_duration)
        if self.segments and self.current.start_time < self.segments[-1].end_time:
            self.current.start_time = self.segments[-1].end_time
            self.current.end_time = max(self.current.end_time, self.current.start_time + self.min_duration)
        if self.current.text:
            self.segments.append(self.current)
        self.current = None

    def finish(self, end_time: float) -> list[SubtitleSegment]:
        if self.current:
            self._finish(min(end_time, max(self.current.last_seen_timestamp + self.end_grace,
                                           self.current.start_time + self.min_duration)))
        for index, segment in enumerate(self.segments, 1):
            segment.index = index
        return self.segments
