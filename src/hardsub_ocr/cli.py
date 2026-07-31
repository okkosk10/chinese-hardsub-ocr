from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

from hardsub_ocr.config import Crop, OcrConfig
from hardsub_ocr.detection.image_preprocessor import MODES
from hardsub_ocr.pipeline import OcrPipeline
from hardsub_ocr.utils.logging_config import configure_logging
from hardsub_ocr.utils.file_utils import output_paths
from hardsub_ocr.utils.process_priority import set_low_priority
from hardsub_ocr.utils.timecode import format_timecode, parse_timecode
from hardsub_ocr.video.video_probe import probe_video, require_ffmpeg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="중국어 하드서브를 OCR하여 SRT/JSON으로 저장합니다.")
    parser.add_argument("--input", type=Path, required=True, help="입력 영상")
    parser.add_argument("--start", default="0", help="시작(HH:MM:SS.mmm 또는 초)")
    parser.add_argument("--end", help="종료(HH:MM:SS.mmm 또는 초), 기본값은 영상 끝")
    parser.add_argument("--crop", type=Crop.parse, required=True, help="x,y,width,height")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--change-threshold", type=float, default=0.045)
    parser.add_argument("--similarity-threshold", type=float, default=82.0)
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--test-seconds", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preprocess-mode", choices=MODES, default="gray2x")
    parser.add_argument("--processing-mode", choices=("fast", "precise"), default="fast")
    parser.add_argument("--transition-settle-seconds", type=float, default=0.15)
    parser.add_argument("--candidate-window-seconds", type=float, default=0.6)
    parser.add_argument("--candidate-frame-count", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--candidate-consensus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-overlap-dedup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suspicious-suffix-removal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-consensus-threshold", type=float, default=88.0)
    parser.add_argument("--line-overlap-max-chars", type=int, default=6)
    parser.add_argument("--unstable-suffix-max-chars", type=int, default=3)
    parser.add_argument("--empty-confirmation-count", type=int, default=2)
    parser.add_argument("--empty-confirmation-seconds", type=float, default=0.4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_ffmpeg()
        info = probe_video(args.input)
        start = parse_timecode(args.start)
        end = parse_timecode(args.end) if args.end else info.duration
        if args.test_seconds:
            end = min(end, start + args.test_seconds)
        config = OcrConfig(args.input.resolve(), start, end, args.crop, args.output_dir.resolve(), args.interval,
                           args.change_threshold, args.similarity_threshold, ffmpeg_threads=args.ffmpeg_threads,
                           preprocess_mode=args.preprocess_mode, save_debug_images=args.save_debug_images,
                           transition_settle_seconds=args.transition_settle_seconds,
                           candidate_window_seconds=args.candidate_window_seconds,
                           candidate_frame_count=args.candidate_frame_count,
                           candidate_consensus_enabled=args.candidate_consensus,
                           line_overlap_dedup_enabled=args.line_overlap_dedup,
                           suspicious_suffix_removal_enabled=args.suspicious_suffix_removal,
                           line_overlap_max_chars=args.line_overlap_max_chars,
                           candidate_consensus_threshold=args.candidate_consensus_threshold,
                           unstable_suffix_max_chars=args.unstable_suffix_max_chars,
                           empty_confirmation_count=args.empty_confirmation_count,
                           empty_confirmation_seconds=args.empty_confirmation_seconds,
                           processing_mode=args.processing_mode)
        paths = output_paths(config.input_path, config.output_dir)
        configure_logging(paths[2], args.verbose)
        set_low_priority()
        pipeline = OcrPipeline(config, callback=lambda p, e: print(
            f"\r{format_timecode(p.current_time)} | frames={p.frames} OCR={p.ocr_runs} skip={p.ocr_skips} segments={p.segments}",
            end="", flush=True))
        signal.signal(signal.SIGINT, lambda *_: pipeline.cancel())
        results = pipeline.run()
        print("\n" + ("중단된 결과를 저장했습니다." if pipeline.cancel_event.is_set() else "완료했습니다."))
        for path in results[:2]:
            print(path)
        return 130 if pipeline.cancel_event.is_set() else 0
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
