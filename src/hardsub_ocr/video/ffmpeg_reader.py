from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path
import subprocess
import threading

import numpy as np

from hardsub_ocr.config import Crop
from hardsub_ocr.video.video_probe import require_ffmpeg


class FFmpegFrameReader:
    def __init__(self, path: Path, start: float, end: float, crop: Crop, interval: float, threads: int = 2) -> None:
        self.path, self.start, self.end = path, start, end
        self.crop, self.interval, self.threads = crop, interval, threads
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr = bytearray()
        self._stopped = False

    def command(self) -> list[str]:
        ffmpeg, _ = require_ffmpeg()
        return [ffmpeg, "-hide_banner", "-loglevel", "warning", "-ss", f"{self.start:.3f}",
                "-i", str(self.path), "-t", f"{self.end - self.start:.3f}", "-threads", str(self.threads),
                "-vf", f"crop={self.crop.width}:{self.crop.height}:{self.crop.x}:{self.crop.y},fps=1/{self.interval}",
                "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    def timestamp_for_index(self, index: int) -> float:
        return self.start + index * self.interval

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        command = self.command()
        logging.getLogger("hardsub_ocr").debug("FFmpeg command: %s", subprocess.list2cmdline(command))
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert self.process.stdout and self.process.stderr
        thread = threading.Thread(target=lambda: self._stderr.extend(self.process.stderr.read()), daemon=True)
        thread.start()
        size = self.crop.width * self.crop.height * 3
        index = 0
        while True:
            data = self.process.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                self.stop()
                raise RuntimeError(f"FFmpeg raw frame이 불완전합니다: {len(data)}/{size} bytes")
            yield index, self.timestamp_for_index(index), np.frombuffer(data, np.uint8).reshape(self.crop.height, self.crop.width, 3)
            index += 1
        code = self.process.wait()
        thread.join(timeout=1)
        if code != 0 and not self._stopped:
            raise RuntimeError(f"FFmpeg 실패({code}): {self.stderr_text[-2000:]}")

    @property
    def stderr_text(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")

    def stop(self) -> None:
        self._stopped = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
