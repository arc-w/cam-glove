import json
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, Qt, Slot
from PySide6.QtGui import QCloseEvent, QColor, QIcon, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from actions.app_discovery import ProgramPickerDialog, format_exec_label
from core.tracker import (
    DEFAULT_MAIN_BINDINGS,
    DEFAULT_ROI_MARGINS,
    DEFAULT_SECONDARY_BINDINGS,
    HandTrackingWorker,
)
from gui.gesture_editor import CustomGestureDialog

if getattr(sys, "frozen", False):
    SETTINGS_PATH = Path(sys.executable).resolve().parent / "cam_glove_settings.json"
else:
    SETTINGS_PATH = Path(__file__).resolve().parent.parent / "cam_glove_settings.json"

PICK_EXEC_SENTINEL = "__pick_exec__"

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
    ("open_osk", "Open Virtual Keyboard"),
    (PICK_EXEC_SENTINEL, "Execute Program / File..."),
    ("none", "Disabled"),
]

SECONDARY_ACTION_OPTIONS: List[Tuple[str, str]] = [
    ("copy", "Copy (Ctrl+C)"),
    ("paste", "Paste (Ctrl+V)"),
    ("show_desktop", "Show Desktop (Win+D)"),
    ("open_osk", "Open Virtual Keyboard"),
    ("left_click", "Left Click (LMB)"),
    ("right_click", "Right Click (RMB)"),
    (PICK_EXEC_SENTINEL, "Execute Program / File..."),
    ("none", "Disabled"),
]


def create_app_icon() -> QIcon:
    """Generate a crisp multi-resolution Cam-Glove icon for the window and Windows System Tray."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64):
        pix = QPixmap(size, size)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        s = float(size)
        margin = s * 0.06
        radius = s * 0.24

        # Rounded dark badge background
        painter.setPen(QPen(QColor(0, 210, 255, 220), max(1.2, s * 0.05)))
        painter.setBrush(QColor(22, 26, 34, 245))
        painter.drawRoundedRect(
            margin,
            margin,
            s - 2 * margin,
            s - 2 * margin,
            radius,
            radius,
        )

        # Stylized hand / cursor ring inside badge
        cx, cy = s * 0.5, s * 0.54
        ring_r = s * 0.24
        painter.setPen(QPen(QColor(0, 230, 255), max(1.5, s * 0.08)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), ring_r, ring_r)

        # Center cursor dot + finger tip accent dots
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(80, 255, 160))
        dot_r = max(1.5, s * 0.08)
        painter.drawEllipse(QPointF(cx, cy), dot_r, dot_r)

        painter.setBrush(QColor(0, 210, 255))
        tip_r = max(1.2, s * 0.055)
        for fx, fy in ((0.30, 0.26), (0.43, 0.20), (0.57, 0.20), (0.70, 0.26)):
            painter.drawEllipse(QPointF(s * fx, s * fy), tip_r, tip_r)

        painter.end()
        icon.addPixmap(pix)
    return icon


class InteractivePreviewLabel(QLabel):
    """
    Camera preview label that lets the user drag the orange cursor-to-desktop mapping rectangle
    (corners, edges, or whole box) directly with the mouse.
    """

    def __init__(
        self,
        tracker: HandTrackingWorker,
        on_roi_changed: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("Starting camera...", parent)
        self.tracker = tracker
        self._on_roi_changed = on_roi_changed
        self.setMouseTracking(True)
        self._drag_mode: Optional[str] = None
        self._drag_start_norm: Tuple[float, float] = (0.0, 0.0)
        self._drag_start_margins: Tuple[float, float, float, float] = (0.15, 0.15, 0.10, 0.38)

    def _widget_to_norm(self, pos_x: float, pos_y: float) -> Optional[Tuple[float, float]]:
        pix = self.pixmap()
        if pix is None or pix.isNull() or pix.width() <= 0 or pix.height() <= 0:
            return None
        pw, ph = float(pix.width()), float(pix.height())
        ox = (self.width() - pw) * 0.5
        oy = (self.height() - ph) * 0.5
        nx = (pos_x - ox) / pw
        ny = (pos_y - oy) / ph
        return nx, ny

    def _hit_test(self, nx: float, ny: float) -> Optional[str]:
        x1 = self.tracker.margin_left
        x2 = 1.0 - self.tracker.margin_right
        y1 = self.tracker.margin_top
        y2 = 1.0 - self.tracker.margin_bottom
        tol = 0.04

        near_l = abs(nx - x1) <= tol and (y1 - tol <= ny <= y2 + tol)
        near_r = abs(nx - x2) <= tol and (y1 - tol <= ny <= y2 + tol)
        near_t = abs(ny - y1) <= tol and (x1 - tol <= nx <= x2 + tol)
        near_b = abs(ny - y2) <= tol and (x1 - tol <= nx <= x2 + tol)

        if near_l and near_t:
            return "tl"
        if near_r and near_t:
            return "tr"
        if near_l and near_b:
            return "bl"
        if near_r and near_b:
            return "br"
        if near_l:
            return "l"
        if near_r:
            return "r"
        if near_t:
            return "t"
        if near_b:
            return "b"
        if x1 < nx < x2 and y1 < ny < y2:
            return "move"
        return None

    def _update_cursor_for_mode(self, mode: Optional[str]) -> None:
        if mode in ("tl", "br"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif mode in ("tr", "bl"):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif mode in ("l", "r"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif mode in ("t", "b"):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif mode == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            if norm is not None:
                mode = self._hit_test(*norm)
                if mode is not None:
                    self._drag_mode = mode
                    self._drag_start_norm = norm
                    self._drag_start_margins = (
                        self.tracker.margin_left,
                        self.tracker.margin_right,
                        self.tracker.margin_top,
                        self.tracker.margin_bottom,
                    )
                    self._update_cursor_for_mode(mode)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        norm = self._widget_to_norm(event.position().x(), event.position().y())
        if norm is None:
            return
        nx, ny = norm

        if self._drag_mode is None:
            self._update_cursor_for_mode(self._hit_test(nx, ny))
            return

        sl, sr, st, sb = self._drag_start_margins
        sx, sy = self._drag_start_norm
        dx = nx - sx
        dy = ny - sy
        min_span = 0.15

        x1, x2 = sl, 1.0 - sr
        y1, y2 = st, 1.0 - sb

        if self._drag_mode == "move":
            w_box = x2 - x1
            h_box = y2 - y1
            new_x1 = max(0.0, min(1.0 - w_box, x1 + dx))
            new_y1 = max(0.0, min(1.0 - h_box, y1 + dy))
            x1, x2 = new_x1, new_x1 + w_box
            y1, y2 = new_y1, new_y1 + h_box
        else:
            if "l" in self._drag_mode:
                x1 = max(0.0, min(x2 - min_span, x1 + dx))
            if "r" in self._drag_mode:
                x2 = min(1.0, max(x1 + min_span, x2 + dx))
            if "t" in self._drag_mode:
                y1 = max(0.0, min(y2 - min_span, y1 + dy))
            if "b" in self._drag_mode:
                y2 = min(1.0, max(y1 + min_span, y2 + dy))

        self.tracker.set_mapping_margins(x1, 1.0 - x2, y1, 1.0 - y2)
        self._on_roi_changed()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_mode is not None:
            self._drag_mode = None
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            if norm is not None:
                self._update_cursor_for_mode(self._hit_test(*norm))
            self._on_roi_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    """Main desktop application window with live camera preview and gesture binding controls."""

    def __init__(
        self,
        tracker: HandTrackingWorker,
        on_close_callback: Optional[Callable[[], None]] = None,
        on_toggle_keyboard: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.tracker = tracker
        self._on_close_callback = on_close_callback
        self._on_toggle_keyboard = on_toggle_keyboard
        self._allow_close = False

        self.setWindowTitle("Cam-Glove — Hand Tracking Control Center")
        self.setWindowIcon(create_app_icon())
        self.resize(1080, 660)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(14)

        # Left side: Camera Preview (interactive orange box)
        preview_group = QGroupBox(
            "Camera Preview (Index 8 = Cursor | Drag Orange Box or Corners to Adjust Area)",
            self,
        )
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = InteractivePreviewLabel(
            tracker=self.tracker,
            on_roi_changed=self._on_preview_roi_dragged,
            parent=self,
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(580, 460)
        self.preview_label.setStyleSheet("background-color: #181818; color: #cccccc;")
        preview_layout.addWidget(self.preview_label)
        root_layout.addWidget(preview_group, stretch=3)

        # Right side: Scrollable Settings & Gesture Remapping
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        controls_panel = QWidget(scroll_area)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 4, 0)
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

        if self._on_toggle_keyboard is not None:
            self.vk_btn = QPushButton("⌨ Open / Hide Virtual Keyboard", self)
            self.vk_btn.clicked.connect(self._on_toggle_keyboard)
            hand_form.addRow("", self.vk_btn)

        controls_layout.addWidget(hand_group)

        # 1b. Cursor Mapping Area (Orange Box) Group
        roi_group = QGroupBox("Cursor Mapping Area (Orange Box)", self)
        roi_layout = QVBoxLayout(roi_group)
        roi_hint = QLabel(
            "Drag the orange box or its corners on the camera preview, or use sliders:",
            roi_group,
        )
        roi_hint.setWordWrap(True)
        roi_hint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        roi_layout.addWidget(roi_hint)

        roi_form = QFormLayout()
        self.roi_sliders: Dict[str, QSlider] = {}
        self.roi_value_labels: Dict[str, QLabel] = {}
        for edge_key, edge_label in (
            ("left", "Left Margin:"),
            ("right", "Right Margin:"),
            ("top", "Top Margin:"),
            ("bottom", "Bottom Margin:"),
        ):
            row_w = QWidget(roi_group)
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(8)

            slider = QSlider(Qt.Orientation.Horizontal, row_w)
            slider.setRange(0, 75)
            val_lbl = QLabel("15%", row_w)
            val_lbl.setFixedWidth(36)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            slider.valueChanged.connect(self._on_roi_slider_changed)
            self.roi_sliders[edge_key] = slider
            self.roi_value_labels[edge_key] = val_lbl

            row_h.addWidget(slider, stretch=1)
            row_h.addWidget(val_lbl)
            roi_form.addRow(edge_label, row_w)

        roi_layout.addLayout(roi_form)
        self.reset_roi_btn = QPushButton("Reset Orange Area", roi_group)
        self.reset_roi_btn.clicked.connect(self._reset_roi_margins)
        roi_layout.addWidget(self.reset_roi_btn)
        controls_layout.addWidget(roi_group)

        # 2. Main Hand Gesture Bindings
        main_gestures_group = QGroupBox("Main Hand Gestures", self)
        main_form = QFormLayout(main_gestures_group)
        self.main_combos: Dict[str, QComboBox] = {}
        for g_key, g_label in MAIN_GESTURE_LABELS:
            combo = self._create_action_combo(MAIN_ACTION_OPTIONS, is_main=True, gesture_key=g_key)
            self.main_combos[g_key] = combo
            main_form.addRow(g_label, combo)
        controls_layout.addWidget(main_gestures_group)

        # 3. Additional (2nd) Hand Gesture Bindings (Built-in + Custom)
        self.sec_gestures_group = QGroupBox("Additional (2nd) Hand Gestures", self)
        self.sec_group_layout = QVBoxLayout(self.sec_gestures_group)
        self.sec_form = QFormLayout()
        self.sec_group_layout.addLayout(self.sec_form)

        self.sec_combos: Dict[str, QComboBox] = {}
        for g_key, g_label in SECONDARY_GESTURE_LABELS:
            combo = self._create_action_combo(
                SECONDARY_ACTION_OPTIONS,
                is_main=False,
                gesture_key=g_key,
            )
            self.sec_combos[g_key] = combo
            self.sec_form.addRow(g_label, combo)

        # Container for dynamic custom gestures
        self.custom_rows_container = QWidget(self.sec_gestures_group)
        self.custom_rows_layout = QVBoxLayout(self.custom_rows_container)
        self.custom_rows_layout.setContentsMargins(0, 4, 0, 4)
        self.custom_rows_layout.setSpacing(6)
        self.sec_group_layout.addWidget(self.custom_rows_container)

        self.add_custom_gesture_btn = QPushButton("+ Create Custom Gesture...", self)
        self.add_custom_gesture_btn.clicked.connect(self._on_create_custom_gesture)
        self.sec_group_layout.addWidget(self.add_custom_gesture_btn)
        controls_layout.addWidget(self.sec_gestures_group)

        # 4. Reset button
        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Defaults", self)
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.reset_btn)
        controls_layout.addLayout(btn_row)
        controls_layout.addStretch(1)

        scroll_area.setWidget(controls_panel)
        root_layout.addWidget(scroll_area, stretch=2)

        # Connect tracker signals
        self.tracker.frame_ready.connect(self.update_preview_frame)
        self.tracker.main_hand_changed.connect(self._sync_main_hand_combo)

        # Load persisted settings (if any) and sync UI
        self._load_settings()
        self._rebuild_custom_gesture_rows()
        self._sync_ui_from_tracker()

    def _create_action_combo(
        self,
        options: List[Tuple[str, str]],
        is_main: bool,
        gesture_key: str,
    ) -> QComboBox:
        combo = QComboBox(self)
        for act_key, act_label in options:
            combo.addItem(act_label, act_key)
        combo.currentIndexChanged.connect(
            lambda _idx, k=gesture_key, c=combo, m=is_main: self._on_binding_combo_changed(k, c, m)
        )
        return combo

    def _ensure_combo_has_action(self, combo: QComboBox, action_key: str) -> int:
        idx = combo.findData(action_key)
        if idx >= 0:
            return idx
        if action_key.startswith("exec:"):
            path_str = action_key[5:].strip()
            label = format_exec_label(path_str)
            # Insert right before "Execute Program / File..."
            pick_idx = combo.findData(PICK_EXEC_SENTINEL)
            insert_at = pick_idx if pick_idx >= 0 else combo.count()
            combo.insertItem(insert_at, label, action_key)
            combo.setItemData(insert_at, path_str, Qt.ItemDataRole.ToolTipRole)
            return insert_at
        return -1

    def _rebuild_custom_gesture_rows(self) -> None:
        # Remove old custom gesture row widgets
        for gid in list(self.sec_combos.keys()):
            if gid not in DEFAULT_SECONDARY_BINDINGS:
                self.sec_combos.pop(gid, None)

        while self.custom_rows_layout.count() > 0:
            item = self.custom_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for cg in self.tracker.custom_gestures:
            gid = str(cg["id"])
            gname = str(cg.get("name", gid))

            row_widget = QWidget(self.custom_rows_container)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            lbl = QLabel(f"{gname}:", row_widget)
            lbl.setMinimumWidth(145)
            lbl.setWordWrap(True)
            row_layout.addWidget(lbl)

            combo = self._create_action_combo(
                SECONDARY_ACTION_OPTIONS,
                is_main=False,
                gesture_key=gid,
            )
            self.sec_combos[gid] = combo
            row_layout.addWidget(combo, stretch=1)

            del_btn = QPushButton("×", row_widget)
            del_btn.setFixedWidth(28)
            del_btn.setToolTip(f"Delete '{gname}'")
            del_btn.clicked.connect(lambda _checked=False, g=gid: self._on_delete_custom_gesture(g))
            row_layout.addWidget(del_btn)

            self.custom_rows_layout.addWidget(row_widget)

    def _on_create_custom_gesture(self) -> None:
        dlg = CustomGestureDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.created_gesture is not None:
            gesture = dlg.created_gesture
            self.tracker.add_custom_gesture(gesture, action_key="copy")
            self._rebuild_custom_gesture_rows()
            self._sync_ui_from_tracker()
            self._save_settings()

    def _on_delete_custom_gesture(self, gesture_id: str) -> None:
        self.tracker.remove_custom_gesture(gesture_id)
        self._rebuild_custom_gesture_rows()
        self._sync_ui_from_tracker()
        self._save_settings()

    def _sync_ui_from_tracker(self) -> None:
        self._sync_main_hand_combo(self.tracker.preferred_main_hand)
        self._sync_roi_sliders_from_tracker()
        for g_key, combo in self.main_combos.items():
            act = self.tracker.main_bindings.get(g_key, DEFAULT_MAIN_BINDINGS[g_key])
            combo.blockSignals(True)
            idx = self._ensure_combo_has_action(combo, act)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        for g_key, combo in self.sec_combos.items():
            default_act = DEFAULT_SECONDARY_BINDINGS.get(g_key, "none")
            act = self.tracker.secondary_bindings.get(g_key, default_act)
            combo.blockSignals(True)
            idx = self._ensure_combo_has_action(combo, act)
            if idx >= 0:
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

    def _sync_roi_sliders_from_tracker(self) -> None:
        mapping = {
            "left": self.tracker.margin_left,
            "right": self.tracker.margin_right,
            "top": self.tracker.margin_top,
            "bottom": self.tracker.margin_bottom,
        }
        for key, val in mapping.items():
            pct = int(round(val * 100))
            slider = self.roi_sliders.get(key)
            lbl = self.roi_value_labels.get(key)
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(pct)
                slider.blockSignals(False)
            if lbl is not None:
                lbl.setText(f"{pct}%")

    def _on_preview_roi_dragged(self) -> None:
        self._sync_roi_sliders_from_tracker()
        self._save_settings()

    def _on_roi_slider_changed(self) -> None:
        left = self.roi_sliders["left"].value() / 100.0
        right = self.roi_sliders["right"].value() / 100.0
        top = self.roi_sliders["top"].value() / 100.0
        bottom = self.roi_sliders["bottom"].value() / 100.0
        self.tracker.set_mapping_margins(left, right, top, bottom)
        self._sync_roi_sliders_from_tracker()
        self._save_settings()

    def _reset_roi_margins(self) -> None:
        self.tracker.reset_mapping_margins()
        self._sync_roi_sliders_from_tracker()
        self._save_settings()

    def _on_binding_combo_changed(
        self,
        gesture_key: str,
        combo: QComboBox,
        is_main: bool,
    ) -> None:
        action_key = str(combo.currentData())
        if action_key == PICK_EXEC_SENTINEL:
            picker = ProgramPickerDialog(self)
            if picker.exec() == QDialog.DialogCode.Accepted and picker.selected_path:
                action_key = f"exec:{picker.selected_path}"
                combo.blockSignals(True)
                idx = self._ensure_combo_has_action(combo, action_key)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.blockSignals(False)
            else:
                # Restore previous binding on cancel
                self._sync_ui_from_tracker()
                return

        if is_main:
            self.tracker.set_main_binding(gesture_key, action_key)
        else:
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
            if isinstance(data.get("custom_gestures"), list):
                self.tracker.custom_gestures = [
                    g for g in data["custom_gestures"] if isinstance(g, dict) and "id" in g
                ]
            if isinstance(data.get("main_bindings"), dict):
                for k in DEFAULT_MAIN_BINDINGS:
                    if isinstance(data["main_bindings"].get(k), str):
                        self.tracker.main_bindings[k] = data["main_bindings"][k]
            if isinstance(data.get("secondary_bindings"), dict):
                for k, v in data["secondary_bindings"].items():
                    if isinstance(k, str) and isinstance(v, str):
                        self.tracker.secondary_bindings[k] = v
            if isinstance(data.get("roi_margins"), dict):
                rm = data["roi_margins"]
                self.tracker.set_mapping_margins(
                    float(rm.get("left", DEFAULT_ROI_MARGINS["left"])),
                    float(rm.get("right", DEFAULT_ROI_MARGINS["right"])),
                    float(rm.get("top", DEFAULT_ROI_MARGINS["top"])),
                    float(rm.get("bottom", DEFAULT_ROI_MARGINS["bottom"])),
                )
        except Exception:
            pass

    def _save_settings(self) -> None:
        data = {
            "preferred_main_hand": self.tracker.preferred_main_hand,
            "main_bindings": self.tracker.main_bindings,
            "secondary_bindings": self.tracker.secondary_bindings,
            "custom_gestures": self.tracker.custom_gestures,
            "roi_margins": {
                "left": round(self.tracker.margin_left, 4),
                "right": round(self.tracker.margin_right, 4),
                "top": round(self.tracker.margin_top, 4),
                "bottom": round(self.tracker.margin_bottom, 4),
            },
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @Slot(QImage)
    def update_preview_frame(self, q_img: QImage) -> None:
        if not self.isVisible():
            return
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def restore_from_tray(self) -> None:
        """Show and bring the control center window to the foreground from the system tray."""
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def force_close(self) -> None:
        """Allow the window to close completely during full application shutdown."""
        self._allow_close = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._allow_close:
            event.ignore()
            self.hide()
            return
        if self._on_close_callback is not None:
            self._on_close_callback()
        super().closeEvent(event)
