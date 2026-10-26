import json
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.tracker import (
    DEFAULT_MAIN_BINDINGS,
    DEFAULT_SECONDARY_BINDINGS,
    HandTrackingWorker,
)

if getattr(sys, "frozen", False):
    SETTINGS_PATH = Path(sys.executable).resolve().parent / "cam_glove_settings.json"
else:
    SETTINGS_PATH = Path(__file__).resolve().parent.parent / "cam_glove_settings.json"

MAIN_GESTURE_LABELS: List[Tuple[str, str]] = [
    ("thumb_middle", "Thumb (4) + Middle (12):"),
    ("thumb_ring", "Thumb (4) + Ring (16):"),
    ("thumb_pinky", "Thumb (4) + Pinky (20):"),
    ("thumb_ring_mcp", "Thumb (4) + Ring Base (13):"),
]

SECONDARY_GESTURE_LABELS: List[Tuple[str, str]] = [
    ("one_finger", "1 Finger (Index):"),
    ("two_fingers", "2 Fingers (Index + Middle):"),
    ("three_fingers", "3 Fingers (Index + Mid + Ring):"),
]

MAIN_ACTION_OPTIONS: List[Tuple[str, str]] = [
    ("left_click", "Left Click (LMB)"),
    ("right_click", "Right Click (RMB)"),
    ("hold_lmb", "Click & Hold / Drag (LMB)"),
    ("hold_mmb", "Middle Mouse / Scroll (MMB)"),
    ("copy", "Copy (Ctrl+C)"),
    ("paste", "Paste (Ctrl+V)"),
    ("show_desktop", "Show Desktop (Win+D)"),
    ("none", "Disabled"),
]

SECONDARY_ACTION_OPTIONS: List[Tuple[str, str]] = [
    ("copy", "Copy (Ctrl+C)"),
    ("paste", "Paste (Ctrl+V)"),
    ("show_desktop", "Show Desktop (Win+D)"),
    ("left_click", "Left Click (LMB)"),
    ("right_click", "Right Click (RMB)"),
    ("none", "Disabled"),
]


class MainWindow(QMainWindow):
    """Main desktop application window with live camera preview and gesture binding controls."""

    def __init__(
        self,
        tracker: HandTrackingWorker,
        on_close_callback: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.tracker = tracker
        self._on_close_callback = on_close_callback

        self.setWindowTitle("Cam-Glove — Hand Tracking Control Center")
        self.resize(980, 540)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(14)

        # Left side: Camera Preview
        preview_group = QGroupBox("Camera Preview (Index 8 = Cursor)", self)
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("Starting camera...", self)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(580, 440)
        self.preview_label.setStyleSheet("background-color: #181818; color: #cccccc;")
        preview_layout.addWidget(self.preview_label)
        root_layout.addWidget(preview_group, stretch=3)

        # Right side: Settings & Gesture Remapping
        controls_panel = QWidget(self)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        # 1. Hand Priority Group
        hand_group = QGroupBox("Hand Priority", self)
        hand_form = QFormLayout(hand_group)
        self.main_hand_combo = QComboBox(self)
        self.main_hand_combo.addItem("Right Hand", "Right")
        self.main_hand_combo.addItem("Left Hand", "Left")
        self.main_hand_combo.currentIndexChanged.connect(self._on_main_hand_combo_changed)
        hand_form.addRow("Default Main Hand:", self.main_hand_combo)

        self.swap_hand_btn = QPushButton("Swap Active Main Hand (F2)", self)
        self.swap_hand_btn.clicked.connect(self.tracker.swap_active_main_hand)
        hand_form.addRow("", self.swap_hand_btn)
        controls_layout.addWidget(hand_group)

        # 2. Main Hand Gesture Bindings
        main_gestures_group = QGroupBox("Main Hand Gestures", self)
        main_form = QFormLayout(main_gestures_group)
        self.main_combos: Dict[str, QComboBox] = {}
        for g_key, g_label in MAIN_GESTURE_LABELS:
            combo = QComboBox(self)
            for act_key, act_label in MAIN_ACTION_OPTIONS:
                combo.addItem(act_label, act_key)
            combo.currentIndexChanged.connect(
                lambda _idx, k=g_key, c=combo: self._on_main_binding_changed(k, c)
            )
            self.main_combos[g_key] = combo
            main_form.addRow(g_label, combo)
        controls_layout.addWidget(main_gestures_group)

        # 3. Additional (2nd) Hand Gesture Bindings
        sec_gestures_group = QGroupBox("Additional (2nd) Hand Gestures", self)
        sec_form = QFormLayout(sec_gestures_group)
        self.sec_combos: Dict[str, QComboBox] = {}
        for g_key, g_label in SECONDARY_GESTURE_LABELS:
            combo = QComboBox(self)
            for act_key, act_label in SECONDARY_ACTION_OPTIONS:
                combo.addItem(act_label, act_key)
            combo.currentIndexChanged.connect(
                lambda _idx, k=g_key, c=combo: self._on_sec_binding_changed(k, c)
            )
            self.sec_combos[g_key] = combo
            sec_form.addRow(g_label, combo)
        controls_layout.addWidget(sec_gestures_group)

        # 4. Reset & Exit buttons
        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Defaults", self)
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.reset_btn)
        controls_layout.addLayout(btn_row)
        controls_layout.addStretch(1)

        root_layout.addWidget(controls_panel, stretch=2)

        # Connect tracker signals
        self.tracker.frame_ready.connect(self.update_preview_frame)
        self.tracker.main_hand_changed.connect(self._sync_main_hand_combo)

        # Load persisted settings (if any) and sync UI
        self._load_settings()
        self._sync_ui_from_tracker()

    def _sync_ui_from_tracker(self) -> None:
        self._sync_main_hand_combo(self.tracker.preferred_main_hand)
        for g_key, combo in self.main_combos.items():
            act = self.tracker.main_bindings.get(g_key, DEFAULT_MAIN_BINDINGS[g_key])
            idx = combo.findData(act)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

        for g_key, combo in self.sec_combos.items():
            act = self.tracker.secondary_bindings.get(g_key, DEFAULT_SECONDARY_BINDINGS[g_key])
            idx = combo.findData(act)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

    @Slot(str)
    def _sync_main_hand_combo(self, hand: str) -> None:
        idx = self.main_hand_combo.findData(hand)
        if idx >= 0:
            self.main_hand_combo.blockSignals(True)
            self.main_hand_combo.setCurrentIndex(idx)
            self.main_hand_combo.blockSignals(False)

    def _on_main_hand_combo_changed(self) -> None:
        hand = str(self.main_hand_combo.currentData())
        self.tracker.set_preferred_main_hand(hand)
        self._save_settings()

    def _on_main_binding_changed(self, gesture_key: str, combo: QComboBox) -> None:
        action_key = str(combo.currentData())
        self.tracker.set_main_binding(gesture_key, action_key)
        self._save_settings()

    def _on_sec_binding_changed(self, gesture_key: str, combo: QComboBox) -> None:
        action_key = str(combo.currentData())
        self.tracker.set_secondary_binding(gesture_key, action_key)
        self._save_settings()

    def _reset_defaults(self) -> None:
        self.tracker.reset_default_bindings()
        self.tracker.set_preferred_main_hand("Right")
        self._sync_ui_from_tracker()
        self._save_settings()

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data.get("preferred_main_hand"), str):
                self.tracker.preferred_main_hand = (
                    "Left" if data["preferred_main_hand"].lower() == "left" else "Right"
                )
            if isinstance(data.get("main_bindings"), dict):
                for k in DEFAULT_MAIN_BINDINGS:
                    if isinstance(data["main_bindings"].get(k), str):
                        self.tracker.main_bindings[k] = data["main_bindings"][k]
            if isinstance(data.get("secondary_bindings"), dict):
                for k in DEFAULT_SECONDARY_BINDINGS:
                    if isinstance(data["secondary_bindings"].get(k), str):
                        self.tracker.secondary_bindings[k] = data["secondary_bindings"][k]
        except Exception:
            pass

    def _save_settings(self) -> None:
        data = {
            "preferred_main_hand": self.tracker.preferred_main_hand,
            "main_bindings": self.tracker.main_bindings,
            "secondary_bindings": self.tracker.secondary_bindings,
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @Slot(QImage)
    def update_preview_frame(self, q_img: QImage) -> None:
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._on_close_callback is not None:
            self._on_close_callback()
        super().closeEvent(event)
