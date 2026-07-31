from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from hardsub_ocr.gui.main_window import MainWindow
from hardsub_ocr.utils.process_priority import set_low_priority


def main() -> int:
    set_low_priority()
    app = QApplication(sys.argv)
    app.setApplicationName("중국어 하드서브 OCR")
    window = MainWindow(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

