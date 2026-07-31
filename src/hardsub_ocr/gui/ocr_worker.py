from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from hardsub_ocr.config import OcrConfig
from hardsub_ocr.pipeline import OcrPipeline


class OcrWorker(QObject):
    progress = Signal(object, object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: OcrConfig) -> None:
        super().__init__()
        self.pipeline = OcrPipeline(config, callback=lambda p, e: self.progress.emit(p, e))

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(self.pipeline.run())
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def cancel(self) -> None:
        self.pipeline.cancel()

