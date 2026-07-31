from __future__ import annotations

from pathlib import Path
import subprocess

from PySide6.QtCore import QThread, QUrl, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

from hardsub_ocr.config import Crop, OcrConfig, UserSettings
from hardsub_ocr.detection.image_preprocessor import MODES
from hardsub_ocr.gui.ocr_worker import OcrWorker
from hardsub_ocr.gui.video_preview import VideoPreview
from hardsub_ocr.utils.file_utils import output_paths
from hardsub_ocr.utils.logging_config import configure_logging
from hardsub_ocr.utils.timecode import format_timecode, parse_timecode
from hardsub_ocr.video.video_probe import probe_video, require_ffmpeg


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("중국어 하드서브 OCR")
        self.resize(1180, 820)
        self.project_root = Path(__file__).resolve().parents[3]
        self.settings_path = self.project_root / "settings.json"
        self.settings = UserSettings.load(self.settings_path)
        self.video_path: Path | None = None
        self.video_info = None
        self.thread: QThread | None = None
        self.worker: OcrWorker | None = None
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.5)
        self.player.setAudioOutput(self.audio)
        self._build_ui()
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(lambda ms: self.timeline.setMaximum(max(1, ms)))
        self.player.errorOccurred.connect(lambda *_: self._log("미리보기 재생 오류. FFmpeg OCR은 별도로 실행할 수 있습니다."))
        try:
            require_ffmpeg()
            self._log("FFmpeg/ffprobe 확인 완료")
        except RuntimeError as exc:
            self._log(str(exc))
        if self.settings.recent_video and Path(self.settings.recent_video).exists():
            self._load_video(Path(self.settings.recent_video))

    def _build_ui(self) -> None:
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        file_row = QHBoxLayout(); root.addLayout(file_row)
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        choose = QPushButton("영상 선택"); choose.clicked.connect(self._choose_video)
        file_row.addWidget(choose); file_row.addWidget(self.path_edit, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal); root.addWidget(splitter, 1)
        left = QWidget(); left_layout = QVBoxLayout(left)
        self.preview = VideoPreview(); self.preview.cropSelected.connect(self._crop_selected)
        self.player.setVideoOutput(self.preview); left_layout.addWidget(self.preview, 1)
        controls = QHBoxLayout(); left_layout.addLayout(controls)
        for text, slot in (("재생", self.player.play), ("일시정지", self.player.pause), ("정지", self.player.stop)):
            button = QPushButton(text); button.clicked.connect(slot); controls.addWidget(button)
        self.timeline = QSlider(Qt.Orientation.Horizontal); self.timeline.sliderMoved.connect(self.player.setPosition)
        controls.addWidget(self.timeline, 1)
        self.position_label = QLabel("00:00:00.000 / 00:00:00.000"); controls.addWidget(self.position_label)
        time_row = QGridLayout(); left_layout.addLayout(time_row)
        self.start_edit, self.end_edit = QLineEdit("00:00:00.000"), QLineEdit("00:00:00.000")
        time_row.addWidget(QLabel("시작"), 0, 0); time_row.addWidget(self.start_edit, 0, 1)
        set_start = QPushButton("현재 위치 → 시작"); set_start.clicked.connect(lambda: self.start_edit.setText(format_timecode(self.player.position()/1000)))
        time_row.addWidget(set_start, 0, 2)
        time_row.addWidget(QLabel("종료"), 1, 0); time_row.addWidget(self.end_edit, 1, 1)
        set_end = QPushButton("현재 위치 → 종료"); set_end.clicked.connect(lambda: self.end_edit.setText(format_timecode(self.player.position()/1000)))
        time_row.addWidget(set_end, 1, 2)
        splitter.addWidget(left)
        panel = QWidget(); form = QFormLayout(panel)
        self.crop_edit = QLineEdit("0,0,1,1"); form.addRow("crop", self.crop_edit)
        clear = QPushButton("선택 영역 초기화"); clear.clicked.connect(self.preview.overlay.clear); form.addRow(clear)
        self.output_edit = QLineEdit(str(Path(self.settings.output_dir).resolve()))
        output_btn = QPushButton("출력 폴더 선택"); output_btn.clicked.connect(self._choose_output)
        outrow = QHBoxLayout(); outrow.addWidget(self.output_edit); outrow.addWidget(output_btn); form.addRow("출력", outrow)
        self.interval = QDoubleSpinBox(); self.interval.setRange(.1, 10); self.interval.setValue(self.settings.interval); self.interval.setSuffix(" 초")
        self.change = QDoubleSpinBox(); self.change.setDecimals(3); self.change.setRange(0, 1); self.change.setSingleStep(.005); self.change.setValue(self.settings.change_threshold)
        self.similarity = QDoubleSpinBox(); self.similarity.setRange(0, 100); self.similarity.setValue(self.settings.similarity_threshold)
        self.threads = QSpinBox(); self.threads.setRange(1, 8); self.threads.setValue(self.settings.ffmpeg_threads)
        self.mode = QComboBox(); self.mode.addItems(MODES); self.mode.setCurrentText(self.settings.preprocess_mode)
        self.debug = QCheckBox(); self.debug.setChecked(self.settings.save_debug_images)
        self.processing_mode = QComboBox(); self.processing_mode.addItem("빠른 모드", "fast"); self.processing_mode.addItem("정밀 모드", "precise")
        index = self.processing_mode.findData(self.settings.processing_mode); self.processing_mode.setCurrentIndex(max(0, index))
        form.addRow("샘플 간격", self.interval); form.addRow("변화 임계값", self.change)
        form.addRow("유사도 임계값", self.similarity); form.addRow("FFmpeg 스레드", self.threads)
        form.addRow("전처리", self.mode); form.addRow("처리 모드", self.processing_mode); form.addRow("디버그 이미지", self.debug)
        advanced = QGroupBox("고급 OCR 안정화 설정"); advanced.setCheckable(True); advanced.setChecked(False)
        advanced_form = QFormLayout(advanced)
        self.settle = QDoubleSpinBox(); self.settle.setRange(0, 2); self.settle.setSingleStep(.05); self.settle.setValue(self.settings.transition_settle_seconds); self.settle.setSuffix(" 초")
        self.candidate_window = QDoubleSpinBox(); self.candidate_window.setRange(.1, 3); self.candidate_window.setSingleStep(.1); self.candidate_window.setValue(self.settings.candidate_window_seconds); self.candidate_window.setSuffix(" 초")
        self.candidate_count = QSpinBox(); self.candidate_count.setRange(1, 3); self.candidate_count.setValue(self.settings.candidate_frame_count)
        self.consensus = QCheckBox(); self.consensus.setChecked(self.settings.candidate_consensus_enabled)
        self.line_dedup = QCheckBox(); self.line_dedup.setChecked(self.settings.line_overlap_dedup_enabled)
        self.suffix_removal = QCheckBox(); self.suffix_removal.setChecked(self.settings.suspicious_suffix_removal_enabled)
        advanced_form.addRow("전환 안정화", self.settle); advanced_form.addRow("후보 수집 구간", self.candidate_window)
        advanced_form.addRow("후보 프레임 수", self.candidate_count); advanced_form.addRow("후보 합의", self.consensus)
        advanced_form.addRow("줄 경계 중복 제거", self.line_dedup); advanced_form.addRow("의심 접미 제거", self.suffix_removal)
        form.addRow(advanced)
        self.test_button = QPushButton("30초 시험 OCR"); self.test_button.clicked.connect(lambda: self._start_ocr(True))
        self.full_button = QPushButton("전체 구간 OCR"); self.full_button.clicked.connect(lambda: self._start_ocr(False))
        self.stop_button = QPushButton("중지"); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self._cancel)
        form.addRow(self.test_button); form.addRow(self.full_button); form.addRow(self.stop_button)
        open_out = QPushButton("결과 폴더 열기"); open_out.clicked.connect(self._open_output); form.addRow(open_out)
        splitter.addWidget(panel); splitter.setSizes([800, 360])
        self.progress = QProgressBar(); root.addWidget(self.progress)
        self.status = QLabel("대기 중"); root.addWidget(self.status)
        bottom = QSplitter(Qt.Orientation.Horizontal); root.addWidget(bottom)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.result = QPlainTextEdit(); self.result.setReadOnly(True)
        bottom.addWidget(self.log); bottom.addWidget(self.result)

    def _choose_video(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "영상 선택", "", "영상 (*.mp4 *.mkv *.avi *.mov *.wmv);;모든 파일 (*)")
        if name: self._load_video(Path(name))

    def _load_video(self, path: Path) -> None:
        try:
            info = probe_video(path)
        except Exception as exc:
            QMessageBox.critical(self, "영상 오류", str(exc)); return
        self.video_path, self.video_info = path, info
        self.path_edit.setText(str(path)); self.preview.set_video_size(info.width, info.height)
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.end_edit.setText(format_timecode(info.duration))
        remembered = self.settings.recent_crops.get(str(path))
        self.crop_edit.setText(remembered or f"0,{round(info.height*.65)},{info.width},{round(info.height*.30)}")
        self.settings.recent_video = str(path); self.settings.save(self.settings_path)
        self._log(f"영상: {info.width}x{info.height}, {format_timecode(info.duration)}, {info.codec}")

    def _crop_selected(self, crop: Crop) -> None:
        self.crop_edit.setText(str(crop)); self._log(f"crop 선택: {crop}")

    def _position_changed(self, ms: int) -> None:
        if not self.timeline.isSliderDown(): self.timeline.setValue(ms)
        duration = self.player.duration()/1000
        self.position_label.setText(f"{format_timecode(ms/1000)} / {format_timecode(duration)}")

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "출력 폴더", self.output_edit.text())
        if path: self.output_edit.setText(path)

    def _make_config(self, test: bool) -> OcrConfig:
        if not self.video_path or not self.video_info: raise ValueError("영상을 먼저 선택하세요.")
        start, end = parse_timecode(self.start_edit.text()), parse_timecode(self.end_edit.text())
        if test: end = min(end, start + self.settings.test_duration, self.video_info.duration)
        config = OcrConfig(self.video_path, start, end, Crop.parse(self.crop_edit.text()), Path(self.output_edit.text()),
                           self.interval.value(), self.change.value(), self.similarity.value(),
                           ffmpeg_threads=self.threads.value(), preprocess_mode=self.mode.currentText(),
                           save_debug_images=self.debug.isChecked(),
                           transition_settle_seconds=self.settle.value(),
                           candidate_window_seconds=self.candidate_window.value(),
                           candidate_frame_count=self.candidate_count.value(),
                           candidate_consensus_enabled=self.consensus.isChecked(),
                           line_overlap_dedup_enabled=self.line_dedup.isChecked(),
                           suspicious_suffix_removal_enabled=self.suffix_removal.isChecked(),
                           processing_mode=self.processing_mode.currentData())
        config.validate(self.video_info.width, self.video_info.height); return config

    def _start_ocr(self, test: bool) -> None:
        try: config = self._make_config(test)
        except Exception as exc: QMessageBox.warning(self, "설정 오류", str(exc)); return
        self.settings.recent_crops[str(self.video_path)] = self.crop_edit.text()
        self.settings.output_dir, self.settings.interval = self.output_edit.text(), self.interval.value()
        self.settings.ffmpeg_threads, self.settings.change_threshold = self.threads.value(), self.change.value()
        self.settings.similarity_threshold, self.settings.preprocess_mode = self.similarity.value(), self.mode.currentText()
        self.settings.save_debug_images = self.debug.isChecked()
        self.settings.transition_settle_seconds, self.settings.candidate_window_seconds = self.settle.value(), self.candidate_window.value()
        self.settings.candidate_frame_count = self.candidate_count.value()
        self.settings.candidate_consensus_enabled, self.settings.line_overlap_dedup_enabled = self.consensus.isChecked(), self.line_dedup.isChecked()
        self.settings.suspicious_suffix_removal_enabled = self.suffix_removal.isChecked()
        self.settings.processing_mode = self.processing_mode.currentData(); self.settings.save(self.settings_path)
        configure_logging(output_paths(config.input_path, config.output_dir)[2])
        self.thread = QThread(self); self.worker = OcrWorker(config); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished); self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.test_button.setEnabled(False); self.full_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.progress.setValue(0); self._log("OCR 시작"); self.thread.start()

    def _on_progress(self, progress, event) -> None:
        percent = int(100 * max(0, progress.current_time - parse_timecode(self.start_edit.text())) / max(.001, progress.total_duration))
        self.progress.setValue(min(100, percent))
        eta = format_timecode(progress.eta) if progress.eta is not None else "계산 중"
        current = format_timecode(progress.current_time)
        elapsed = format_timecode(progress.elapsed)
        self.status.setText(f"{current} | 프레임 {progress.frames} | OCR {progress.ocr_runs} | 생략 {progress.ocr_skips} | "
                            f"자막 {progress.segments} | 경과 {elapsed} | 남은 시간 {eta}")
        if event and event.normalized_text:
            self.result.setPlainText(event.normalized_text)
            self._log(f"{format_timecode(event.timestamp)} [{event.confidence:.2f}] {event.normalized_text.replace(chr(10), ' / ')}")

    def _on_finished(self, paths) -> None:
        self._reset_running(); self.progress.setValue(100); self.status.setText("완료 (또는 중단 결과 저장 완료)")
        self._log("저장: " + ", ".join(str(p) for p in paths[:2]))

    def _on_failed(self, message: str) -> None:
        self._reset_running(); self.status.setText("실패"); self._log("오류: " + message); QMessageBox.critical(self, "OCR 오류", message)

    def _reset_running(self) -> None:
        self.test_button.setEnabled(True); self.full_button.setEnabled(True); self.stop_button.setEnabled(False)

    def _cancel(self) -> None:
        if self.worker: self.worker.cancel(); self.status.setText("중단 및 결과 저장 중…")

    def _open_output(self) -> None:
        path = Path(self.output_edit.text()); path.mkdir(parents=True, exist_ok=True); subprocess.Popen(["explorer", str(path)])

    def _log(self, message: str) -> None: self.log.appendPlainText(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker: self.worker.cancel()
        self.settings.save(self.settings_path); super().closeEvent(event)
