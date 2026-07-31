from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Crop:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            raise ValueError("crop은 0 이상의 x,y와 양수 width,height여야 합니다.")

    @classmethod
    def parse(cls, value: str) -> "Crop":
        try:
            parts = [int(x.strip()) for x in value.split(",")]
        except ValueError as exc:
            raise ValueError("crop 형식은 x,y,width,height입니다.") from exc
        if len(parts) != 4:
            raise ValueError("crop 형식은 x,y,width,height입니다.")
        return cls(*parts)

    def __str__(self) -> str:
        return f"{self.x},{self.y},{self.width},{self.height}"


@dataclass(slots=True)
class OcrConfig:
    input_path: Path
    start_time: float
    end_time: float
    crop: Crop
    output_dir: Path = Path("output")
    interval: float = 0.5
    change_threshold: float = 0.045
    similarity_threshold: float = 82.0
    short_similarity_threshold: float = 94.0
    ffmpeg_threads: int = 2
    preprocess_mode: str = "gray2x"
    save_debug_images: bool = False
    min_duration: float = 0.35
    max_duration: float = 8.0
    blank_tolerance: int = 1
    end_grace: float = 0.15
    transition_settle_seconds: float = 0.15
    candidate_window_seconds: float = 0.6
    candidate_frame_count: int = 3
    candidate_consensus_enabled: bool = True
    line_overlap_dedup_enabled: bool = True
    suspicious_suffix_removal_enabled: bool = True
    line_overlap_max_chars: int = 6
    candidate_consensus_threshold: float = 88.0
    unstable_suffix_max_chars: int = 3
    empty_confirmation_count: int = 2
    empty_confirmation_seconds: float = 0.4
    processing_mode: str = "fast"

    def validate(self, video_width: int | None = None, video_height: int | None = None) -> None:
        if not self.input_path.is_file():
            raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {self.input_path}")
        if self.start_time < 0 or self.end_time <= self.start_time:
            raise ValueError("종료 시간은 시작 시간보다 커야 합니다.")
        if self.interval <= 0:
            raise ValueError("샘플링 간격은 0보다 커야 합니다.")
        if self.transition_settle_seconds < 0 or self.candidate_window_seconds <= 0:
            raise ValueError("전환 안정화 시간은 0 이상, 후보 수집 구간은 0보다 커야 합니다.")
        if not 1 <= self.candidate_frame_count <= 3:
            raise ValueError("후보 프레임 수는 1~3이어야 합니다.")
        if not 1 <= self.line_overlap_max_chars <= 6:
            raise ValueError("줄 중복 검사 길이는 1~6이어야 합니다.")
        if not 1 <= self.unstable_suffix_max_chars <= 3:
            raise ValueError("의심 접미 최대 길이는 1~3이어야 합니다.")
        if not 0 <= self.candidate_consensus_threshold <= 100:
            raise ValueError("후보 합의 임계값은 0~100이어야 합니다.")
        if self.empty_confirmation_count < 1 or self.empty_confirmation_seconds < 0:
            raise ValueError("빈 결과 확인 횟수는 1 이상이고 확인 시간은 0 이상이어야 합니다.")
        if self.processing_mode not in {"fast", "precise"}:
            raise ValueError("처리 모드는 fast 또는 precise여야 합니다.")
        if video_width and self.crop.x + self.crop.width > video_width:
            raise ValueError("crop이 영상 가로 범위를 벗어납니다.")
        if video_height and self.crop.y + self.crop.height > video_height:
            raise ValueError("crop이 영상 세로 범위를 벗어납니다.")


@dataclass(slots=True)
class UserSettings:
    output_dir: str = "output"
    interval: float = 0.5
    ffmpeg_threads: int = 2
    change_threshold: float = 0.045
    similarity_threshold: float = 82.0
    test_duration: float = 30.0
    preprocess_mode: str = "gray2x"
    save_debug_images: bool = False
    transition_settle_seconds: float = 0.15
    candidate_window_seconds: float = 0.6
    candidate_frame_count: int = 3
    candidate_consensus_enabled: bool = True
    line_overlap_dedup_enabled: bool = True
    suspicious_suffix_removal_enabled: bool = True
    processing_mode: str = "fast"
    recent_video: str = ""
    recent_crops: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "UserSettings":
        if not path.exists():
            return cls()
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            allowed = cls.__dataclass_fields__.keys()
            return cls(**{k: v for k, v in data.items() if k in allowed})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
