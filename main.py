
"""
Video Frame Extractor with GUI (PySide6)

Prerequisites:
- Python 3.8+ recommended.
- Install dependencies: `pip install -r requirements.txt`

Run the application:
- `python main.py`
"""
from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import imageio
import numpy as np
from PySide6.QtCore import (
    Qt,
    QSize,
    QTimer,
    QThread,
    Signal,
)
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QStatusBar,
    QCheckBox,
)


@dataclass
class VideoMetadata:
    """Container for basic video metadata."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


class FrameExtractionWorker(QThread):
    """
    Background worker to extract frames without blocking the UI.

    Emits:
        progress_changed (int): Percentage 0-100.
        status_message (str): Human readable status.
        finished (bool, str): Success flag and message.
    """

    progress_changed = Signal(int)
    status_message = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        pattern: str,
        image_format: str,
        start_frame: int,
        end_frame: int,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.pattern = pattern
        self.image_format = image_format.lower()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation; the worker checks between frames."""
        self._cancelled = True

    def run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video for extraction.")

            total_frames = max(1, self.end_frame - self.start_frame + 1)
            os.makedirs(self.output_dir, exist_ok=True)

            for offset, frame_idx in enumerate(range(self.start_frame, self.end_frame + 1)):
                if self._cancelled:
                    self.finished.emit(False, "Extraction cancelled.")
                    return

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.finished.emit(False, f"Failed to read frame {frame_idx}.")
                    return

                filename = self.pattern.format(index=frame_idx)
                if not filename.lower().endswith(f".{self.image_format}"):
                    filename = f"{filename}.{self.image_format}"
                out_path = os.path.join(self.output_dir, filename)

                save_ok = cv2.imwrite(out_path, frame)
                if not save_ok:
                    self.finished.emit(False, f"Failed to save frame to {out_path}")
                    return

                percent = int(((offset + 1) / total_frames) * 100)
                self.progress_changed.emit(percent)
                self.status_message.emit(f"Saved frame {frame_idx} ({percent}%)")

            self.finished.emit(True, f"Saved {total_frames} frames to {self.output_dir}")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))
        finally:
            if cap is not None:
                cap.release()


class GifExportWorker(QThread):
    """
    Background worker to build an animated GIF.

    Note: GIF generation can be slow and memory-heavy for long videos.
    """

    progress_changed = Signal(int)
    status_message = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        video_path: str,
        output_path: str,
        start_frame: int,
        end_frame: int,
        gif_fps: float,
        scale_percent: int,
        frame_skip: int,
        loop_count: int,
        max_frames: int = 800,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.gif_fps = max(0.1, gif_fps)
        self.scale_percent = max(1, scale_percent)
        self.frame_skip = max(1, frame_skip)
        self.loop_count = max(0, loop_count)
        self.max_frames = max_frames
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation during GIF creation."""
        self._cancelled = True

    def run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        writer = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video for GIF export.")

            raw_count = self.end_frame - self.start_frame + 1
            frames_to_process = max(1, math.ceil(raw_count / self.frame_skip))
            # Clamp to avoid excessive GIF sizes.
            frames_to_process = min(frames_to_process, self.max_frames)
            final_end_frame = self.start_frame + (frames_to_process - 1) * self.frame_skip

            writer = imageio.get_writer(
                self.output_path,
                mode="I",
                duration=1.0 / self.gif_fps,
                loop=self.loop_count,
            )

            written = 0
            for frame_idx in range(self.start_frame, final_end_frame + 1, self.frame_skip):
                if self._cancelled:
                    self.finished.emit(False, "GIF export cancelled.")
                    return

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.finished.emit(False, f"Failed to read frame {frame_idx}.")
                    return

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.scale_percent != 100:
                    new_w = max(1, int(rgb.shape[1] * self.scale_percent / 100))
                    new_h = max(1, int(rgb.shape[0] * self.scale_percent / 100))
                    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

                writer.append_data(rgb)
                written += 1
                percent = int((written / frames_to_process) * 100)
                self.progress_changed.emit(percent)
                self.status_message.emit(f"GIF frames added: {written}/{frames_to_process}")

            message = f"GIF saved to {self.output_path}"
            if raw_count > frames_to_process * self.frame_skip:
                message += " (limited for size)"
            self.finished.emit(True, message)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))
        finally:
            if writer is not None:
                writer.close()
            if cap is not None:
                cap.release()


class VideoFrameExtractorApp(QMainWindow):
    """Main application window for video frame extraction and GIF export."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Frame Extractor with GIF Export (Experimental)")
        self.resize(1200, 800)
        self._apply_dark_theme()

        self.metadata: Optional[VideoMetadata] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.current_frame_index: int = 0
        self.playing: bool = False
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)

        self.extract_worker: Optional[FrameExtractionWorker] = None
        self.gif_worker: Optional[GifExportWorker] = None

        self._build_ui()
        self._update_controls_enabled(False)

    def _apply_dark_theme(self) -> None:
        """Set a simple dark palette for a modern look."""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.Highlight, QColor(90, 120, 200))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)

    def _build_ui(self) -> None:
        """Construct all widgets and layouts."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Preview area
        self.preview_label = QLabel("Open a video to begin.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(QSize(640, 360))
        self.preview_label.setStyleSheet("QLabel { background-color: #222; border: 1px solid #444; }")
        main_layout.addWidget(self.preview_label, stretch=2)

        # Navigation controls
        controls_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open Video")
        self.btn_prev = QPushButton("Prev Frame")
        self.btn_next = QPushButton("Next Frame")
        self.btn_play = QPushButton("Play")
        self.btn_jump_back = QPushButton("Jump -10")
        self.btn_jump_forward = QPushButton("Jump +10")
        controls_layout.addWidget(self.btn_open)
        controls_layout.addWidget(self.btn_prev)
        controls_layout.addWidget(self.btn_next)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_jump_back)
        controls_layout.addWidget(self.btn_jump_forward)
        main_layout.addLayout(controls_layout)

        # Timeline
        timeline_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider_label = QLabel("0 / 0")
        timeline_layout.addWidget(self.slider, stretch=4)
        timeline_layout.addWidget(self.slider_label)
        main_layout.addLayout(timeline_layout)

        # Extraction controls
        extraction_group = QGroupBox("Frame Extraction")
        extraction_layout = QVBoxLayout(extraction_group)

        out_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.btn_browse_output = QPushButton("Choose Output Dir")
        out_dir_layout.addWidget(self.output_dir_edit)
        out_dir_layout.addWidget(self.btn_browse_output)
        extraction_layout.addLayout(out_dir_layout)

        pattern_layout = QHBoxLayout()
        self.pattern_edit = QLineEdit("frame_{index:05d}")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg"])
        pattern_layout.addWidget(QLabel("Filename pattern"))
        pattern_layout.addWidget(self.pattern_edit)
        pattern_layout.addWidget(QLabel("Format"))
        pattern_layout.addWidget(self.format_combo)
        extraction_layout.addLayout(pattern_layout)

        range_layout = QHBoxLayout()
        self.range_start_spin = QSpinBox()
        self.range_end_spin = QSpinBox()
        self.range_start_spin.setMinimum(0)
        self.range_end_spin.setMinimum(0)
        self.range_start_spin.setMaximum(0)
        self.range_end_spin.setMaximum(0)
        range_layout.addWidget(QLabel("Start"))
        range_layout.addWidget(self.range_start_spin)
        range_layout.addWidget(QLabel("End"))
        range_layout.addWidget(self.range_end_spin)
        extraction_layout.addLayout(range_layout)

        button_layout = QHBoxLayout()
        self.btn_extract_current = QPushButton("Extract Current Frame")
        self.btn_extract_all = QPushButton("Extract All Frames")
        self.btn_extract_range = QPushButton("Extract Frame Range")
        button_layout.addWidget(self.btn_extract_current)
        button_layout.addWidget(self.btn_extract_all)
        button_layout.addWidget(self.btn_extract_range)
        extraction_layout.addLayout(button_layout)

        progress_layout = QHBoxLayout()
        self.extract_progress = QProgressBar()
        self.extract_progress.setValue(0)
        self.btn_cancel_extract = QPushButton("Cancel")
        self.btn_cancel_extract.setEnabled(False)
        progress_layout.addWidget(self.extract_progress, stretch=3)
        progress_layout.addWidget(self.btn_cancel_extract)
        extraction_layout.addLayout(progress_layout)

        main_layout.addWidget(extraction_group)

        # GIF Export section
        gif_group = QGroupBox("GIF Export (Experimental)")
        gif_layout = QVBoxLayout(gif_group)

        source_layout = QHBoxLayout()
        self.chk_gif_full = QCheckBox("Full video")
        self.chk_gif_full.setChecked(True)
        self.gif_range_start = QSpinBox()
        self.gif_range_end = QSpinBox()
        self.gif_range_start.setEnabled(False)
        self.gif_range_end.setEnabled(False)
        source_layout.addWidget(self.chk_gif_full)
        source_layout.addWidget(QLabel("Start"))
        source_layout.addWidget(self.gif_range_start)
        source_layout.addWidget(QLabel("End"))
        source_layout.addWidget(self.gif_range_end)
        gif_layout.addLayout(source_layout)

        gif_settings_layout = QFormLayout()
        self.gif_fps_spin = QDoubleSpinBox()
        self.gif_fps_spin.setRange(0.1, 120.0)
        self.gif_fps_spin.setValue(12.0)
        self.gif_scale_spin = QSpinBox()
        self.gif_scale_spin.setRange(1, 200)
        self.gif_scale_spin.setValue(100)
        self.gif_skip_spin = QSpinBox()
        self.gif_skip_spin.setRange(1, 30)
        self.gif_skip_spin.setValue(1)
        self.gif_loop_spin = QSpinBox()
        self.gif_loop_spin.setRange(0, 50)
        self.gif_loop_spin.setValue(0)
        gif_settings_layout.addRow("GIF FPS", self.gif_fps_spin)
        gif_settings_layout.addRow("Scale (%)", self.gif_scale_spin)
        gif_settings_layout.addRow("Frame skip (every Nth frame)", self.gif_skip_spin)
        gif_settings_layout.addRow("Loop count (0=infinite)", self.gif_loop_spin)
        gif_layout.addLayout(gif_settings_layout)

        gif_output_layout = QHBoxLayout()
        self.gif_output_edit = QLineEdit("output.gif")
        self.btn_gif_output = QPushButton("Choose GIF Path")
        gif_output_layout.addWidget(self.gif_output_edit)
        gif_output_layout.addWidget(self.btn_gif_output)
        gif_layout.addLayout(gif_output_layout)

        self.gif_estimate_label = QLabel("Estimated frames: 0")
        gif_layout.addWidget(self.gif_estimate_label)

        gif_progress_layout = QHBoxLayout()
        self.gif_progress = QProgressBar()
        self.gif_progress.setValue(0)
        self.btn_export_gif = QPushButton("Export GIF")
        self.btn_cancel_gif = QPushButton("Cancel")
        self.btn_cancel_gif.setEnabled(False)
        gif_progress_layout.addWidget(self.gif_progress, stretch=3)
        gif_progress_layout.addWidget(self.btn_export_gif)
        gif_progress_layout.addWidget(self.btn_cancel_gif)
        gif_layout.addLayout(gif_progress_layout)

        main_layout.addWidget(gif_group)

        # Metadata panel
        metadata_group = QGroupBox("Video Metadata")
        metadata_form = QFormLayout(metadata_group)
        self.meta_path_label = QLabel("-")
        self.meta_resolution_label = QLabel("-")
        self.meta_fps_label = QLabel("-")
        self.meta_framecount_label = QLabel("-")
        self.meta_duration_label = QLabel("-")
        metadata_form.addRow("Path", self.meta_path_label)
        metadata_form.addRow("Resolution", self.meta_resolution_label)
        metadata_form.addRow("FPS", self.meta_fps_label)
        metadata_form.addRow("Total Frames", self.meta_framecount_label)
        metadata_form.addRow("Duration (s)", self.meta_duration_label)
        main_layout.addWidget(metadata_group)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Idle")

        # Connections
        self.btn_open.clicked.connect(self.open_video)
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_jump_back.clicked.connect(lambda: self.jump_frames(-10))
        self.btn_jump_forward.clicked.connect(lambda: self.jump_frames(10))
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderMoved.connect(self.slider_moved)

        self.btn_browse_output.clicked.connect(self.choose_output_dir)
        self.btn_extract_current.clicked.connect(self.extract_current_frame)
        self.btn_extract_all.clicked.connect(self.extract_all_frames)
        self.btn_extract_range.clicked.connect(self.extract_frame_range)
        self.btn_cancel_extract.clicked.connect(self.cancel_extraction)

        self.chk_gif_full.stateChanged.connect(self.update_gif_source_state)
        self.btn_gif_output.clicked.connect(self.choose_gif_output)
        self.btn_export_gif.clicked.connect(self.export_gif)
        self.btn_cancel_gif.clicked.connect(self.cancel_gif_export)
        self.gif_range_start.valueChanged.connect(self.update_gif_estimate)
        self.gif_range_end.valueChanged.connect(self.update_gif_estimate)
        self.gif_skip_spin.valueChanged.connect(self.update_gif_estimate)
        self.gif_fps_spin.valueChanged.connect(self.update_gif_estimate)
        self.gif_scale_spin.valueChanged.connect(self.update_gif_estimate)
        self.gif_loop_spin.valueChanged.connect(self.update_gif_estimate)

    # ----------------------------------------------------------------------
    # Video loading and navigation
    # ----------------------------------------------------------------------
    def open_video(self) -> None:
        """Open a video file via dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv);;All Files (*)",
        )
        if not file_path:
            return
        self.load_video(file_path)

    def load_video(self, path: str) -> None:
        """Load video and initialize metadata."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0

        self.metadata = VideoMetadata(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
        )
        self.capture = cap
        self.current_frame_index = 0
        self.slider.setMaximum(max(0, frame_count - 1))
        self.range_start_spin.setMaximum(max(0, frame_count - 1))
        self.range_end_spin.setMaximum(max(0, frame_count - 1))
        self.gif_range_start.setMaximum(max(0, frame_count - 1))
        self.gif_range_end.setMaximum(max(0, frame_count - 1))
        self.range_end_spin.setValue(max(0, frame_count - 1))
        self.gif_range_end.setValue(max(0, frame_count - 1))
        self.update_metadata_display()
        self.go_to_frame(0)
        self._update_controls_enabled(True)
        self.status.showMessage("Video loaded.")
        self.update_gif_estimate()

    def update_metadata_display(self) -> None:
        """Display video metadata in the panel."""
        if not self.metadata:
            return
        self.meta_path_label.setText(self.metadata.path)
        self.meta_resolution_label.setText(f"{self.metadata.width} x {self.metadata.height}")
        self.meta_fps_label.setText(f"{self.metadata.fps:.2f}")
        self.meta_framecount_label.setText(str(self.metadata.frame_count))
        self.meta_duration_label.setText(f"{self.metadata.duration_sec:.2f}")

    def slider_pressed(self) -> None:
        """Pause playback while the slider is dragged."""
        if self.playing:
            self.toggle_play()

    def slider_released(self) -> None:
        """Seek to the frame after slider is released."""
        self.go_to_frame(self.slider.value())

    def slider_moved(self, value: int) -> None:
        """Update label while sliding."""
        if self.metadata:
            self.slider_label.setText(f"{value} / {self.metadata.frame_count - 1}")

    def go_to_frame(self, index: int) -> None:
        """Seek to a specific frame and show it."""
        if not self.capture or not self.metadata:
            return
        index = max(0, min(index, self.metadata.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            QMessageBox.warning(self, "Error", f"Unable to read frame {index}.")
            return
        self.current_frame_index = index
        self.slider.setValue(index)
        self.slider_label.setText(f"{index} / {self.metadata.frame_count - 1}")
        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Render the frame into the preview label."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """Ensure the preview scales with the window."""
        super().resizeEvent(event)
        if self.capture and self.metadata:
            self.go_to_frame(self.current_frame_index)

    def prev_frame(self) -> None:
        """Step back by one frame."""
        self.go_to_frame(self.current_frame_index - 1)

    def next_frame(self) -> None:
        """Step forward by one frame."""
        if self.metadata and self.current_frame_index >= self.metadata.frame_count - 1:
            if self.playing:
                self.toggle_play()
            return
        self.go_to_frame(self.current_frame_index + 1)

    def jump_frames(self, delta: int) -> None:
        """Jump by a delta (e.g., ?10 frames)."""
        self.go_to_frame(self.current_frame_index + delta)

    def toggle_play(self) -> None:
        """Play or pause simple frame-by-frame playback."""
        if not self.metadata or not self.capture:
            return
        if self.playing:
            self.play_timer.stop()
            self.btn_play.setText("Play")
            self.playing = False
            self.status.showMessage("Paused")
        else:
            interval_ms = int(1000 / self.metadata.fps) if self.metadata.fps > 0 else 33
            self.play_timer.start(max(15, interval_ms))
            self.btn_play.setText("Pause")
            self.playing = True
            self.status.showMessage("Playing")

    # ----------------------------------------------------------------------
    # Extraction logic
    # ----------------------------------------------------------------------
    def choose_output_dir(self) -> None:
        """Select output directory for extracted frames."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def _ensure_output_dir(self) -> Optional[str]:
        """Validate the output directory, asking to create if missing."""
        path = self.output_dir_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Output Directory", "Please choose an output directory.")
            return None
        if not os.path.exists(path):
            create = QMessageBox.question(
                self,
                "Create Directory",
                f"Directory does not exist:\n{path}\nCreate it?",
            )
            if create == QMessageBox.Yes:
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.critical(self, "Error", f"Failed to create directory: {exc}")
                    return None
            else:
                return None
        if not os.access(path, os.W_OK):
            QMessageBox.critical(self, "Error", "Output directory is not writable.")
            return None
        return path

    def extract_current_frame(self) -> None:
        """Extract only the currently displayed frame."""
        if not self._can_extract():
            return
        output_dir = self._ensure_output_dir()
        if not output_dir:
            return
        pattern = self.pattern_edit.text().strip() or "frame_{index:05d}"
        image_format = self.format_combo.currentText()
        frame_idx = self.current_frame_index

        cap = cv2.VideoCapture(self.metadata.path) if self.metadata else None
        if not cap or not cap.isOpened():
            QMessageBox.critical(self, "Error", "Unable to reopen video for extraction.")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QMessageBox.critical(self, "Error", f"Failed to read frame {frame_idx}.")
            return

        filename = pattern.format(index=frame_idx)
        if not filename.lower().endswith(f".{image_format}"):
            filename = f"{filename}.{image_format}"
        out_path = os.path.join(output_dir, filename)
        try:
            save_ok = cv2.imwrite(out_path, frame)
            if not save_ok:
                raise RuntimeError("cv2.imwrite returned False")
            QMessageBox.information(self, "Success", f"Frame saved to {out_path}")
            self.status.showMessage(f"Saved frame {frame_idx}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to save frame: {exc}")

    def _start_extraction_worker(self, start_frame: int, end_frame: int) -> None:
        """Create and run the frame extraction worker."""
        if not self.metadata:
            return
        output_dir = self._ensure_output_dir()
        if not output_dir:
            return
        pattern = self.pattern_edit.text().strip() or "frame_{index:05d}"
        image_format = self.format_combo.currentText()

        self.extract_worker = FrameExtractionWorker(
            video_path=self.metadata.path,
            output_dir=output_dir,
            pattern=pattern,
            image_format=image_format,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.extract_worker.progress_changed.connect(self.extract_progress.setValue)
        self.extract_worker.status_message.connect(self.status.showMessage)
        self.extract_worker.finished.connect(self._on_extract_finished)
        self.extract_progress.setValue(0)
        self.btn_cancel_extract.setEnabled(True)
        self._update_controls_enabled(False)
        self.status.showMessage("Extracting frames...")
        self.extract_worker.start()

    def extract_all_frames(self) -> None:
        """Extract all frames using a background worker."""
        if not self._can_extract():
            return
        if self.metadata and self.metadata.frame_count > 5000:
            warn = QMessageBox.question(
                self,
                "Large Extraction",
                "Extracting many frames can take time and disk space. Continue?",
            )
            if warn != QMessageBox.Yes:
                return
        self._start_extraction_worker(0, self.metadata.frame_count - 1)

    def extract_frame_range(self) -> None:
        """Extract a validated frame range."""
        if not self._can_extract():
            return
        start = self.range_start_spin.value()
        end = self.range_end_spin.value()
        if not self._validate_frame_range(start, end):
            return
        self._start_extraction_worker(start, end)

    def cancel_extraction(self) -> None:
        """Cancel running extraction worker."""
        if self.extract_worker and self.extract_worker.isRunning():
            self.extract_worker.cancel()
            self.status.showMessage("Cancelling extraction...")

    def _on_extract_finished(self, success: bool, message: str) -> None:
        """Handle extraction completion."""
        self.btn_cancel_extract.setEnabled(False)
        self._update_controls_enabled(True)
        self.extract_progress.setValue(0)
        if success:
            QMessageBox.information(self, "Extraction", message)
        else:
            QMessageBox.warning(self, "Extraction", message)
        self.status.showMessage(message)

    # ----------------------------------------------------------------------
    # GIF export
    # ----------------------------------------------------------------------
    def update_gif_source_state(self) -> None:
        """Enable/disable range inputs based on full-video checkbox."""
        full = self.chk_gif_full.isChecked()
        self.gif_range_start.setEnabled(not full)
        self.gif_range_end.setEnabled(not full)
        self.update_gif_estimate()

    def choose_gif_output(self) -> None:
        """Select a file path for GIF export."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GIF",
            self.gif_output_edit.text() or "output.gif",
            "GIF Files (*.gif);;All Files (*)",
        )
        if path:
            if not path.lower().endswith(".gif"):
                path += ".gif"
            self.gif_output_edit.setText(path)

    def update_gif_estimate(self) -> None:
        """Update the estimated number of frames for GIF export."""
        if not self.metadata:
            self.gif_estimate_label.setText("Estimated frames: 0")
            return
        start, end = 0, self.metadata.frame_count - 1
        if not self.chk_gif_full.isChecked():
            start = self.gif_range_start.value()
            end = self.gif_range_end.value()
        start = max(0, min(start, self.metadata.frame_count - 1))
        end = max(0, min(end, self.metadata.frame_count - 1))
        if end < start:
            end = start
        span = end - start + 1
        skip = max(1, self.gif_skip_spin.value())
        estimated = max(1, math.ceil(span / skip))
        if estimated > 800:
            note = " (clamped during export)"
        else:
            note = ""
        self.gif_estimate_label.setText(f"Estimated frames: {estimated}{note}")

    def export_gif(self) -> None:
        """Start GIF export in the background."""
        if not self._can_extract():
            return
        if not self.metadata:
            return
        start, end = 0, self.metadata.frame_count - 1
        if not self.chk_gif_full.isChecked():
            start = self.gif_range_start.value()
            end = self.gif_range_end.value()
            if not self._validate_frame_range(start, end):
                return

        output_path = self.gif_output_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "GIF Output", "Please choose an output path for the GIF.")
            return
        out_dir = os.path.dirname(output_path) or "."
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error", f"Cannot create directory: {exc}")
                return

        gif_fps = self.gif_fps_spin.value()
        scale_percent = self.gif_scale_spin.value()
        frame_skip = self.gif_skip_spin.value()
        loop_count = self.gif_loop_spin.value()

        self.gif_worker = GifExportWorker(
            video_path=self.metadata.path,
            output_path=output_path,
            start_frame=start,
            end_frame=end,
            gif_fps=gif_fps,
            scale_percent=scale_percent,
            frame_skip=frame_skip,
            loop_count=loop_count,
        )
        self.gif_worker.progress_changed.connect(self.gif_progress.setValue)
        self.gif_worker.status_message.connect(self.status.showMessage)
        self.gif_worker.finished.connect(self._on_gif_finished)

        self.gif_progress.setValue(0)
        self.btn_cancel_gif.setEnabled(True)
        self._update_controls_enabled(False)
        self.status.showMessage("Exporting GIF...")
        self.gif_worker.start()

    def cancel_gif_export(self) -> None:
        """Cancel a running GIF export."""
        if self.gif_worker and self.gif_worker.isRunning():
            self.gif_worker.cancel()
            self.status.showMessage("Cancelling GIF export...")

    def _on_gif_finished(self, success: bool, message: str) -> None:
        """Handle GIF export completion."""
        self.btn_cancel_gif.setEnabled(False)
        self._update_controls_enabled(True)
        self.gif_progress.setValue(0)
        if success:
            QMessageBox.information(self, "GIF Export", message)
        else:
            QMessageBox.warning(self, "GIF Export", message)
        self.status.showMessage(message)

    # ----------------------------------------------------------------------
    # Utility helpers
    # ----------------------------------------------------------------------
    def _can_extract(self) -> bool:
        """Check whether a video is loaded before extraction/export."""
        if not self.metadata or not self.capture:
            QMessageBox.warning(self, "No Video", "Please open a video first.")
            return False
        return True

    def _validate_frame_range(self, start: int, end: int) -> bool:
        """Ensure the frame range is valid."""
        if not self.metadata:
            return False
        if start < 0 or end < 0 or start >= self.metadata.frame_count or end >= self.metadata.frame_count:
            QMessageBox.warning(self, "Invalid Range", "Frame range is out of bounds.")
            return False
        if end < start:
            QMessageBox.warning(self, "Invalid Range", "End frame must be >= start frame.")
            return False
        return True

    def _update_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable controls based on worker activity."""
        controls = [
            self.btn_prev,
            self.btn_next,
            self.btn_play,
            self.btn_jump_back,
            self.btn_jump_forward,
            self.slider,
            self.btn_extract_current,
            self.btn_extract_all,
            self.btn_extract_range,
            self.btn_export_gif,
        ]
        for widget in controls:
            widget.setEnabled(enabled)
        # Always allow opening a video so the app is usable when nothing is loaded.
        self.btn_open.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Ensure workers are stopped on exit."""
        if self.extract_worker and self.extract_worker.isRunning():
            self.extract_worker.cancel()
            self.extract_worker.wait(1000)
        if self.gif_worker and self.gif_worker.isRunning():
            self.gif_worker.cancel()
            self.gif_worker.wait(1000)
        if self.capture:
            self.capture.release()
        super().closeEvent(event)


def main() -> None:
    """Entry point to launch the application."""
    app = QApplication(sys.argv)
    window = VideoFrameExtractorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
