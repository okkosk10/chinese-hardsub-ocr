from __future__ import annotations

from PySide6.QtCore import QRect, Signal
from PySide6.QtMultimediaWidgets import QVideoWidget

from hardsub_ocr.config import Crop
from hardsub_ocr.gui.crop_overlay import CropOverlay
from hardsub_ocr.video.frame_mapper import Rect, map_widget_crop, video_display_rect


class VideoPreview(QVideoWidget):
    cropSelected = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self._video_size = (0, 0)
        self.overlay = CropOverlay(self)
        self.overlay.selectionChanged.connect(self._map_selection)

    def set_video_size(self, width: int, height: int) -> None:
        self._video_size = (width, height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())
        self.overlay.raise_()

    def _map_selection(self, rect: QRect) -> None:
        width, height = self._video_size
        display = video_display_rect(self.width(), self.height(), width, height)
        try:
            crop = map_widget_crop(Rect(rect.x(), rect.y(), rect.width(), rect.height()), display, width, height)
            self.cropSelected.emit(crop)
        except ValueError:
            pass

