from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float
    codec: str = ""


@dataclass(slots=True)
class OcrResult:
    text: str = ""
    normalized_text: str = ""
    confidence: float = 0.0
    raw_result: Any = None
    boxes: list[Any] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass(slots=True)
class SubtitleSegment:
    index: int
    start_time: float
    end_time: float
    text: str
    confidence: float
    source_frames: list[int] = field(default_factory=list)
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OcrEvent:
    frame_index: int
    timestamp: float
    detected_text: str = ""
    normalized_text: str = ""
    confidence: float = 0.0
    frame_change_score: float = 1.0
    previous_similarity_score: float = 0.0
    segment_action: str = ""
    crop: str = ""
    preprocessing_mode: str = ""
    processing_time_ms: float = 0.0
    error: str = ""
    debug_image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Progress:
    frames: int = 0
    ocr_runs: int = 0
    ocr_skips: int = 0
    segments: int = 0
    current_time: float = 0.0
    total_duration: float = 0.0
    elapsed: float = 0.0
    eta: float | None = None

