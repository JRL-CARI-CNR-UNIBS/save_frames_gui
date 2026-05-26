#!/usr/bin/env python3
"""ROS 2 multi-camera RGB/depth preview and frame saver GUI.

The node intentionally reads the full YAML file itself. This allows camera
sections such as `rs1:` and `rs2:` at the YAML top level, while ROS receives
only a standard `config_file` parameter from the launch file.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

from PyQt5.QtCore import QLibraryInfo, Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    color_image_topic: str
    depth_image_topic: str = ""
    camera_info_topic: str = ""
    frames_approx_sync: bool = True
    depth_frame_encoding: str = "32FC1"
    depth_unit_in_meters: bool = True


@dataclass
class FrameSnapshot:
    color_bgr: Optional[np.ndarray]
    depth_m: Optional[np.ndarray]
    color_stamp: Optional[dict]
    depth_stamp: Optional[dict]
    color_frame_id: str
    depth_frame_id: str
    camera_info: Optional[dict]
    color_encoding: str
    depth_encoding: str


@dataclass
class SaveOptions:
    dataset_dir: str
    rgb_format: str = "png"
    depth_format: str = "npy"
    save_mode: str = "RGB + depth_m"


@dataclass
class SaveResult:
    camera_name: str
    saved_files: list
    warnings: list

    @property
    def saved(self) -> bool:
        return bool(self.saved_files)


class CameraStream:
    """Keeps the latest synchronized frames for one camera."""

    def __init__(self, node: Node, cfg: CameraConfig, queue_size: int, slop_sec: float):
        self.node = node
        self.cfg = cfg
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.color_bgr: Optional[np.ndarray] = None
        self.depth_m: Optional[np.ndarray] = None
        self.depth_raw: Optional[np.ndarray] = None
        self.color_stamp: Optional[dict] = None
        self.depth_stamp: Optional[dict] = None
        self.color_frame_id: str = ""
        self.depth_frame_id: str = ""
        self.color_encoding: str = ""
        self.depth_encoding: str = ""
        self.camera_info: Optional[dict] = None
        self.last_error: str = "Waiting for frames"
        self.frames_received = 0
        self.last_frame_wall_time = 0.0

        self._subs = []
        self._sync = None

        if cfg.depth_image_topic:
            color_sub = message_filters.Subscriber(node, Image, cfg.color_image_topic)
            depth_sub = message_filters.Subscriber(node, Image, cfg.depth_image_topic)
            self._subs.extend([color_sub, depth_sub])
            if cfg.frames_approx_sync:
                self._sync = message_filters.ApproximateTimeSynchronizer(
                    [color_sub, depth_sub], queue_size=queue_size, slop=slop_sec
                )
            else:
                self._sync = message_filters.TimeSynchronizer([color_sub, depth_sub], queue_size=queue_size)
            self._sync.registerCallback(self._frames_callback)
        else:
            self._subs.append(
                node.create_subscription(Image, cfg.color_image_topic, self._color_only_callback, 10)
            )

        if cfg.camera_info_topic:
            self._subs.append(
                node.create_subscription(CameraInfo, cfg.camera_info_topic, self._camera_info_callback, 10)
            )

    @staticmethod
    def _stamp_to_dict(msg: Image) -> dict:
        sec = int(msg.header.stamp.sec)
        nanosec = int(msg.header.stamp.nanosec)
        return {
            "sec": sec,
            "nanosec": nanosec,
            "float_sec": float(sec) + float(nanosec) * 1e-9,
        }

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        info = {
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": str(msg.distortion_model),
            "d": list(msg.d),
            "k": list(msg.k),
            "r": list(msg.r),
            "p": list(msg.p),
            "frame_id": str(msg.header.frame_id),
        }
        with self.lock:
            self.camera_info = info

    def _color_msg_to_bgr(self, msg: Image) -> np.ndarray:
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return np.ascontiguousarray(image)

    def _depth_msg_to_meters(self, msg: Image) -> Tuple[np.ndarray, np.ndarray]:
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        raw = np.ascontiguousarray(raw)
        depth_m = raw.astype(np.float32, copy=False)
        if not self.cfg.depth_unit_in_meters:
            depth_m = depth_m * 0.001
        return raw, np.ascontiguousarray(depth_m)

    def _frames_callback(self, color_msg: Image, depth_msg: Image) -> None:
        try:
            color_bgr = self._color_msg_to_bgr(color_msg)
            depth_raw, depth_m = self._depth_msg_to_meters(depth_msg)
            with self.lock:
                self.color_bgr = color_bgr
                self.depth_raw = depth_raw
                self.depth_m = depth_m
                self.color_stamp = self._stamp_to_dict(color_msg)
                self.depth_stamp = self._stamp_to_dict(depth_msg)
                self.color_frame_id = str(color_msg.header.frame_id)
                self.depth_frame_id = str(depth_msg.header.frame_id)
                self.color_encoding = color_msg.encoding
                self.depth_encoding = depth_msg.encoding
                self.frames_received += 1
                self.last_frame_wall_time = time.time()
                self.last_error = "OK"
        except Exception as exc:  # noqa: BLE001 - GUI must stay alive on malformed frames
            with self.lock:
                self.last_error = f"Frame conversion error: {exc}"
            self.node.get_logger().warning(f"[{self.cfg.name}] {self.last_error}")

    def _color_only_callback(self, color_msg: Image) -> None:
        try:
            color_bgr = self._color_msg_to_bgr(color_msg)
            with self.lock:
                self.color_bgr = color_bgr
                self.color_stamp = self._stamp_to_dict(color_msg)
                self.color_frame_id = str(color_msg.header.frame_id)
                self.color_encoding = color_msg.encoding
                self.frames_received += 1
                self.last_frame_wall_time = time.time()
                self.last_error = "OK"
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.last_error = f"Frame conversion error: {exc}"
            self.node.get_logger().warning(f"[{self.cfg.name}] {self.last_error}")

    def snapshot(self) -> FrameSnapshot:
        with self.lock:
            return FrameSnapshot(
                color_bgr=None if self.color_bgr is None else self.color_bgr.copy(),
                depth_m=None if self.depth_m is None else self.depth_m.copy(),
                color_stamp=None if self.color_stamp is None else dict(self.color_stamp),
                depth_stamp=None if self.depth_stamp is None else dict(self.depth_stamp),
                color_frame_id=self.color_frame_id,
                depth_frame_id=self.depth_frame_id,
                camera_info=None if self.camera_info is None else dict(self.camera_info),
                color_encoding=self.color_encoding,
                depth_encoding=self.depth_encoding,
            )

    def status_text(self) -> str:
        with self.lock:
            age = time.time() - self.last_frame_wall_time if self.last_frame_wall_time else None
            if age is None:
                return self.last_error
            return f"{self.last_error} · frames: {self.frames_received} · age: {age:.2f}s"


class SaveFramesGuiNode(Node):
    def __init__(self) -> None:
        super().__init__("save_frames_gui_node")
        self.declare_parameter("config_file", "")
        self.declare_parameter("dataset_dir", "/tmp/vision_system_dataset")
        self.declare_parameter("cameras", [])

        self.config_file = str(self.get_parameter("config_file").value or "")
        self.raw_config = self._load_yaml(self.config_file)
        node_params = self.raw_config.get("save_frames_gui_node", {}).get("ros__parameters", {})

        self.dataset_dir = str(
            node_params.get("dataset_dir")
            or self.get_parameter("dataset_dir").value
            or "/tmp/vision_system_dataset"
        )
        self.ui_refresh_rate_hz = float(node_params.get("ui_refresh_rate_hz", 30.0))
        self.sync_queue_size = int(node_params.get("sync_queue_size", 10))
        self.sync_slop_sec = float(node_params.get("sync_slop_sec", 0.05))
        self.save_all_service_name = str(node_params.get("save_all_service_name", "~/save_all_frames"))

        self.save_options_lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.save_options = SaveOptions(
            dataset_dir=self.dataset_dir,
            rgb_format="png",
            depth_format="npy",
            save_mode="RGB + depth_m",
        )

        camera_names = list(node_params.get("cameras", []))
        if not camera_names:
            camera_names = list(self.get_parameter("cameras").value or [])

        self.streams: Dict[str, CameraStream] = {}
        self.camera_configs: Dict[str, CameraConfig] = {}
        for camera_name in camera_names:
            cfg = self._camera_config_from_yaml(str(camera_name))
            if not cfg.color_image_topic:
                self.get_logger().warning(f"Skipping camera '{camera_name}': missing color_image_topic")
                continue
            self.camera_configs[cfg.name] = cfg
            self.streams[cfg.name] = CameraStream(
                self, cfg, queue_size=self.sync_queue_size, slop_sec=self.sync_slop_sec
            )
            self.get_logger().info(
                f"Camera '{cfg.name}': color='{cfg.color_image_topic}', depth='{cfg.depth_image_topic}', "
                f"approx_sync={cfg.frames_approx_sync}"
            )

        if not self.streams:
            self.get_logger().warning("No cameras configured. Check config_file and cameras list.")

        self.save_all_service = self.create_service(
            Trigger, self.save_all_service_name, self._save_all_frames_service_callback
        )
        self.get_logger().info(f"Save-all Trigger service ready: {self.save_all_service_name}")

    def update_save_options(
        self,
        dataset_dir: Optional[str] = None,
        rgb_format: Optional[str] = None,
        depth_format: Optional[str] = None,
        save_mode: Optional[str] = None,
    ) -> None:
        with self.save_options_lock:
            current = self.save_options
            self.save_options = SaveOptions(
                dataset_dir=str(dataset_dir if dataset_dir is not None else current.dataset_dir),
                rgb_format=str(rgb_format if rgb_format is not None else current.rgb_format),
                depth_format=str(depth_format if depth_format is not None else current.depth_format),
                save_mode=str(save_mode if save_mode is not None else current.save_mode),
            )

    def get_save_options(self) -> SaveOptions:
        with self.save_options_lock:
            return SaveOptions(
                dataset_dir=self.save_options.dataset_dir,
                rgb_format=self.save_options.rgb_format,
                depth_format=self.save_options.depth_format,
                save_mode=self.save_options.save_mode,
            )

    def _save_all_frames_service_callback(self, request, response):  # noqa: ANN001
        del request
        try:
            results = self.save_all_available()
            saved_results, warnings, saved_names, skipped_names, message = self.format_save_results(results)
            response.success = bool(saved_results)
            response.message = message
            if warnings:
                self.get_logger().warning(message.replace("\n", " | "))
            else:
                self.get_logger().info(message)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f"Save-all service failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def save_all_available(self, options: Optional[SaveOptions] = None) -> list:
        options = options or self.get_save_options()
        results = []
        with self.save_lock:
            for camera_name in self.streams:
                try:
                    results.append(self._save_camera_unlocked(camera_name, options, best_effort=True))
                except Exception as exc:  # noqa: BLE001
                    results.append(SaveResult(camera_name=camera_name, saved_files=[], warnings=[str(exc)]))
        return results

    @staticmethod
    def format_save_results(results: list) -> tuple:
        saved_results = [r for r in results if r.saved]
        warnings = []
        for result in results:
            warnings.extend(result.warnings)

        saved_names = ", ".join(r.camera_name for r in saved_results) or "none"
        skipped_names = ", ".join(r.camera_name for r in results if not r.saved) or "none"

        if warnings:
            message = (
                "Partial save completed. "
                f"Saved cameras: {saved_names}. "
                f"Skipped cameras: {skipped_names}. "
                "Details: " + " | ".join(warnings)
            )
        else:
            message = f"Saved frames for {len(saved_results)} cameras: {saved_names}"
        return saved_results, warnings, saved_names, skipped_names, message

    def save_camera(
        self, camera_name: str, options: Optional[SaveOptions] = None, best_effort: bool = False
    ) -> SaveResult:
        options = options or self.get_save_options()
        with self.save_lock:
            return self._save_camera_unlocked(camera_name, options, best_effort=best_effort)

    def _save_camera_unlocked(self, camera_name: str, options: SaveOptions, best_effort: bool = False) -> SaveResult:
        stream = self.streams[camera_name]
        cfg = self.camera_configs[camera_name]
        snapshot = stream.snapshot()
        mode = options.save_mode
        need_rgb = mode in ("RGB only", "RGB + depth_m")
        need_depth = mode in ("RGB + depth_m", "Depth only")
        warnings = []

        rgb_available = snapshot.color_bgr is not None
        depth_available = snapshot.depth_m is not None

        if need_rgb and not rgb_available:
            message = f"{camera_name}: RGB frame is not available yet."
            if best_effort:
                warnings.append(message)
            else:
                raise RuntimeError(message)

        if need_depth and not depth_available:
            message = f"{camera_name}: depth frame is not available yet."
            if best_effort:
                warnings.append(message)
            else:
                raise RuntimeError(message)

        save_rgb = need_rgb and rgb_available
        save_depth_frame = need_depth and depth_available

        if not save_rgb and not save_depth_frame:
            if best_effort:
                if not warnings:
                    warnings.append(f"{camera_name}: no requested frame data is available.")
                return SaveResult(camera_name=camera_name, saved_files=[], warnings=warnings)
            raise RuntimeError(f"{camera_name}: no requested frame data is available.")

        dataset_dir = Path(options.dataset_dir).expanduser()
        camera_dir = dataset_dir / camera_name
        color_dir = camera_dir / "color"
        depth_dir = camera_dir / "depth"
        meta_dir = camera_dir / "meta"
        color_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file_stem = f"{camera_name}_{timestamp}"
        saved_files = []

        if save_rgb:
            rgb_ext = options.rgb_format
            rgb_path = color_dir / f"{file_stem}_rgb.{rgb_ext}"
            if not cv2.imwrite(str(rgb_path), snapshot.color_bgr):
                raise RuntimeError(f"Unable to write RGB image: {rgb_path}")
            saved_files.append(str(rgb_path.relative_to(camera_dir)))

        if save_depth_frame:
            depth_path = save_depth(snapshot.depth_m, depth_dir, file_stem, options.depth_format)
            saved_files.append(str(depth_path.relative_to(camera_dir)))

        meta_path = meta_dir / f"{file_stem}_meta.json"
        meta_rel = str(meta_path.relative_to(camera_dir))

        meta = {
            "camera": camera_name,
            "dataset_layout": "per_camera_color_depth_meta_v1",
            "saved_at_local": datetime.now().isoformat(timespec="milliseconds"),
            "requested_mode": mode,
            "actual_saved": {
                "rgb": bool(save_rgb),
                "depth": bool(save_depth_frame),
            },
            "warnings": warnings,
            "rgb_format": options.rgb_format,
            "depth_format": options.depth_format,
            "color_stamp": snapshot.color_stamp,
            "depth_stamp": snapshot.depth_stamp,
            "color_frame_id": snapshot.color_frame_id,
            "depth_frame_id": snapshot.depth_frame_id,
            "color_encoding": snapshot.color_encoding,
            "depth_encoding": snapshot.depth_encoding or cfg.depth_frame_encoding,
            "depth_unit_saved": "meters_float32_for_npy_tiff32_exr; millimeters_uint16_for_png16",
            "topics": {
                "color_image_topic": cfg.color_image_topic,
                "depth_image_topic": cfg.depth_image_topic,
                "camera_info_topic": cfg.camera_info_topic,
            },
            "camera_info": snapshot.camera_info,
            "files": saved_files + [meta_rel],
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        saved_files.append(meta_rel)
        return SaveResult(camera_name=camera_name, saved_files=saved_files, warnings=warnings)

    def _load_yaml(self, path: str) -> dict:
        if not path:
            return {}
        config_path = Path(path).expanduser()
        if not config_path.exists():
            self.get_logger().warning(f"Config file does not exist: {config_path}")
            return {}
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Unable to read config file '{config_path}': {exc}")
            return {}

    def _camera_config_from_yaml(self, name: str) -> CameraConfig:
        data = self.raw_config.get(name, {}) or {}
        return CameraConfig(
            name=name,
            color_image_topic=str(data.get("color_image_topic", "")),
            depth_image_topic=str(data.get("depth_image_topic", "")),
            camera_info_topic=str(data.get("camera_info_topic", "")),
            frames_approx_sync=bool(data.get("frames_approx_sync", True)),
            depth_frame_encoding=str(data.get("depth_frame_encoding", "32FC1")),
            depth_unit_in_meters=bool(data.get("depth_unit_in_meters", True)),
        )


class CameraTab(QWidget):
    def __init__(self, camera_name: str) -> None:
        super().__init__()
        self.camera_name = camera_name
        self.rgb_label = QLabel("Waiting for RGB frame")
        self.depth_label = QLabel("Waiting for depth frame")
        self.status_label = QLabel("Waiting")
        self._build_ui()

    def _build_ui(self) -> None:
        self.rgb_label.setObjectName("ImagePanel")
        self.depth_label.setObjectName("ImagePanel")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setMinimumSize(420, 315)
        self.depth_label.setMinimumSize(420, 315)
        self.rgb_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.depth_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        rgb_box = QGroupBox("RGB")
        rgb_layout = QVBoxLayout(rgb_box)
        rgb_layout.addWidget(self.rgb_label)

        depth_box = QGroupBox("Depth preview")
        depth_layout = QVBoxLayout(depth_box)
        depth_layout.addWidget(self.depth_label)

        grid = QGridLayout()
        grid.addWidget(rgb_box, 0, 0)
        grid.addWidget(depth_box, 0, 1)
        grid.addWidget(self.status_label, 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        self.setLayout(grid)

    def update_images(self, snapshot: FrameSnapshot, status: str) -> None:
        if snapshot.color_bgr is not None:
            self.rgb_label.setPixmap(cv_bgr_to_qpixmap(snapshot.color_bgr, self.rgb_label.size()))
        if snapshot.depth_m is not None:
            self.depth_label.setPixmap(depth_m_to_qpixmap(snapshot.depth_m, self.depth_label.size()))
        self.status_label.setText(status)


class MainWindow(QMainWindow):
    def __init__(self, node: SaveFramesGuiNode) -> None:
        super().__init__()
        self.node = node
        self.setWindowTitle("ROS 2 RGB-D Dataset Saver")
        self.resize(1320, 820)
        self.camera_tabs: Dict[str, CameraTab] = {}
        self._build_ui()
        self._apply_style()

        refresh_ms = max(10, int(1000.0 / max(1.0, self.node.ui_refresh_rate_hz)))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_previews)
        self.timer.start(refresh_ms)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 12)
        root_layout.setSpacing(14)

        title = QLabel("RGB-D Dataset Saver")
        title.setObjectName("Title")
        subtitle = QLabel("Real-time multi-camera RGB/depth preview and frame saving")
        subtitle.setObjectName("Subtitle")

        header_layout = QVBoxLayout()
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addLayout(header_layout)

        controls = QFrame()
        controls.setObjectName("ControlsCard")
        controls_layout = QGridLayout(controls)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(10)

        self.dataset_edit = QLineEdit(self.node.dataset_dir)
        self.dataset_edit.setMinimumWidth(460)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._choose_dataset_dir)

        self.rgb_format_combo = QComboBox()
        self.rgb_format_combo.addItems(["png", "jpg", "bmp", "tiff"])

        self.depth_format_combo = QComboBox()
        self.depth_format_combo.addItems(["npy", "png16", "tiff32", "exr"])

        self.save_mode_combo = QComboBox()
        self.save_mode_combo.addItems(["RGB only", "RGB + depth_m", "Depth only"])
        self.save_mode_combo.setCurrentText("RGB + depth_m")

        self.dataset_edit.textChanged.connect(self._sync_save_options)
        self.rgb_format_combo.currentTextChanged.connect(self._sync_save_options)
        self.depth_format_combo.currentTextChanged.connect(self._sync_save_options)
        self.save_mode_combo.currentTextChanged.connect(self._sync_save_options)

        save_btn = QPushButton("Save active camera frame")
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self._save_active_camera)

        save_all_btn = QPushButton("Save all available frames")
        save_all_btn.clicked.connect(self._save_all_cameras)

        controls_layout.addWidget(QLabel("Dataset directory"), 0, 0)
        controls_layout.addWidget(self.dataset_edit, 0, 1)
        controls_layout.addWidget(browse_btn, 0, 2)
        controls_layout.addWidget(QLabel("RGB format"), 1, 0)
        controls_layout.addWidget(self.rgb_format_combo, 1, 1)
        controls_layout.addWidget(QLabel("Depth format"), 2, 0)
        controls_layout.addWidget(self.depth_format_combo, 2, 1)
        controls_layout.addWidget(QLabel("Save mode"), 3, 0)
        controls_layout.addWidget(self.save_mode_combo, 3, 1)
        controls_layout.addWidget(save_btn, 1, 2)
        controls_layout.addWidget(save_all_btn, 2, 2)
        controls_layout.setColumnStretch(1, 1)
        root_layout.addWidget(controls)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        if self.node.streams:
            for camera_name in self.node.streams:
                tab = CameraTab(camera_name)
                self.camera_tabs[camera_name] = tab
                self.tabs.addTab(tab, f"Camera {camera_name}")
        else:
            empty = QLabel("No cameras configured. Check config_file and the cameras list.")
            empty.setAlignment(Qt.AlignCenter)
            self.tabs.addTab(empty, "No cameras")
        root_layout.addWidget(self.tabs, stretch=1)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        self._sync_save_options()

    def _apply_style(self) -> None:
        app_font = QFont("Inter", 10)
        QApplication.instance().setFont(app_font)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #111827;
                color: #E5E7EB;
            }
            QLabel#Title {
                font-size: 28px;
                font-weight: 700;
                color: #F9FAFB;
            }
            QLabel#Subtitle {
                font-size: 13px;
                color: #9CA3AF;
            }
            QFrame#ControlsCard, QGroupBox {
                background: #1F2937;
                border: 1px solid #374151;
                border-radius: 14px;
            }
            QGroupBox {
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #D1D5DB;
            }
            QLabel#ImagePanel {
                background: #030712;
                border: 1px solid #374151;
                border-radius: 12px;
                color: #6B7280;
            }
            QLineEdit, QComboBox {
                background: #0B1220;
                color: #E5E7EB;
                border: 1px solid #4B5563;
                border-radius: 9px;
                padding: 8px;
            }
            QPushButton {
                background: #374151;
                color: #F9FAFB;
                border: 1px solid #4B5563;
                border-radius: 10px;
                padding: 9px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #4B5563;
            }
            QPushButton#PrimaryButton {
                background: #2563EB;
                border-color: #3B82F6;
            }
            QPushButton#PrimaryButton:hover {
                background: #1D4ED8;
            }
            QTabWidget::pane {
                border: 1px solid #374151;
                border-radius: 14px;
                top: -1px;
                background: #111827;
            }
            QTabBar::tab {
                background: #1F2937;
                color: #D1D5DB;
                padding: 10px 18px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #2563EB;
                color: #FFFFFF;
            }
            QStatusBar {
                background: #111827;
                color: #9CA3AF;
            }
            """
        )

    def _choose_dataset_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select dataset directory",
            self.dataset_edit.text() or str(Path.home()),
        )
        if directory:
            self.dataset_edit.setText(directory)

    def _sync_save_options(self, *_args) -> None:
        self.node.update_save_options(
            dataset_dir=self.dataset_edit.text(),
            rgb_format=self.rgb_format_combo.currentText(),
            depth_format=self.depth_format_combo.currentText(),
            save_mode=self.save_mode_combo.currentText(),
        )

    def _refresh_previews(self) -> None:
        for camera_name, stream in self.node.streams.items():
            tab = self.camera_tabs.get(camera_name)
            if tab is None:
                continue
            tab.update_images(stream.snapshot(), stream.status_text())

    def _active_camera_name(self) -> Optional[str]:
        idx = self.tabs.currentIndex()
        if idx < 0:
            return None
        widget = self.tabs.widget(idx)
        for name, tab in self.camera_tabs.items():
            if tab is widget:
                return name
        return None

    def _save_active_camera(self) -> None:
        camera_name = self._active_camera_name()
        if not camera_name:
            QMessageBox.warning(self, "Save failed", "No active camera.")
            return
        try:
            result = self.node.save_camera(camera_name, best_effort=False)
            self.statusBar().showMessage(f"Saved {camera_name}: {', '.join(result.saved_files)}")
            if result.warnings:
                QMessageBox.warning(self, "Saved with warnings", "\n".join(result.warnings))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

    def _save_all_cameras(self) -> None:
        """Save a best-effort snapshot for every configured camera."""
        results = self.node.save_all_available()
        saved_results, warnings, saved_names, skipped_names, message = self.node.format_save_results(results)

        if warnings:
            detail_lines = []
            detail_lines.append("Saved cameras: " + saved_names)
            detail_lines.append("Skipped cameras: " + skipped_names)
            detail_lines.append("")
            detail_lines.append("Details:")
            detail_lines.extend(warnings)
            QMessageBox.warning(self, "Partial save completed", "\n".join(detail_lines))
        else:
            QMessageBox.information(
                self,
                "Save completed",
                f"Saved frames for {len(saved_results)} cameras: {saved_names}",
            )
        self.statusBar().showMessage(
            f"Saved {len(saved_results)}/{len(results)} cameras. Skipped: {skipped_names}"
        )

    def _save_camera(self, camera_name: str, best_effort: bool = False) -> SaveResult:
        return self.node.save_camera(camera_name, best_effort=best_effort)


def save_depth(depth_m: np.ndarray, depth_dir: Path, file_stem: str, depth_format: str) -> Path:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    clean_depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    if depth_format == "npy":
        path = depth_dir / f"{file_stem}_depth_m.npy"
        np.save(str(path), clean_depth.astype(np.float32))
        return path

    if depth_format == "png16":
        path = depth_dir / f"{file_stem}_depth_mm.png"
        depth_mm = np.clip(clean_depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        if not cv2.imwrite(str(path), depth_mm):
            raise RuntimeError(f"Unable to write depth PNG16: {path}")
        return path

    if depth_format == "tiff32":
        path = depth_dir / f"{file_stem}_depth_m.tiff"
        if not cv2.imwrite(str(path), clean_depth.astype(np.float32)):
            raise RuntimeError(f"Unable to write depth TIFF32: {path}")
        return path

    if depth_format == "exr":
        path = depth_dir / f"{file_stem}_depth_m.exr"
        if not cv2.imwrite(str(path), clean_depth.astype(np.float32)):
            raise RuntimeError(
                f"Unable to write depth EXR: {path}. "
                "The OpenCV build may not have OpenEXR enabled."
            )
        return path

    raise ValueError(f"Unsupported depth format: {depth_format}")


def cv_bgr_to_qpixmap(image_bgr: np.ndarray, target_size) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def prefer_pyqt_platform_plugins() -> None:
    """Prefer PyQt platform plugins over OpenCV wheel Qt plugins.

    The non-headless `opencv-python` wheel may set Qt plugin paths to
    `cv2/qt/plugins`, which can break PyQt applications with the classic
    `Could not load the Qt platform plugin "xcb"` error. Resetting the
    platform plugin path before QApplication is created makes the GUI more
    robust.
    """
    try:
        plugins_path = Path(QLibraryInfo.location(QLibraryInfo.PluginsPath)) / "platforms"
        if plugins_path.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_path)
        os.environ.pop("QT_PLUGIN_PATH", None)
    except Exception:
        pass


def depth_m_to_qpixmap(depth_m: np.ndarray, target_size) -> QPixmap:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        preview = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-3
        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        normalized[~valid] = 0.0
        preview_gray = (normalized * 255.0).astype(np.uint8)
        preview = cv2.applyColorMap(preview_gray, cv2.COLORMAP_TURBO)
        preview[~valid] = (0, 0, 0)
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SaveFramesGuiNode()

    prefer_pyqt_platform_plugins()
    app = QApplication(sys.argv)
    app.setApplicationName("ROS 2 RGB-D Dataset Saver")
    window = MainWindow(node)
    window.show()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        exit_code = app.exec_()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
