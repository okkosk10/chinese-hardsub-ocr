from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class CropOverlay(QWidget):
    selectionChanged = Signal(QRect)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self._start: QPoint | None = None
        self._rect = QRect()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self._rect = QRect(self._start, self._start)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            self._rect = QRect(self._start, event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._start is not None:
            self._rect = QRect(self._start, event.position().toPoint()).normalized()
            self._start = None
            if self._rect.width() > 4 and self._rect.height() > 4:
                self.selectionChanged.emit(self._rect)
            self.update()

    def clear(self) -> None:
        self._rect = QRect()
        self.update()

    def paintEvent(self, event) -> None:
        if self._rect.isEmpty():
            return
        painter = QPainter(self)
        painter.fillRect(self._rect, QColor(0, 170, 255, 45))
        painter.setPen(QPen(QColor(0, 190, 255), 2))
        painter.drawRect(self._rect)

