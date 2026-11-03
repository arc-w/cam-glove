import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
     QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

LANDMARK_NAMES: Dict[int, str] = {
    0: "Wrist (0)",
    1: "Thumb CMC (1)",
    2: "Thumb MCP (2)",
    3: "Thumb IP (3)",
    4: "Thumb Tip (4)",
    5: "Index Base (5)",
    6: "Index Mid (6)",
    7: "Index Top (7)",
    8: "Index Tip (8)",
    9: "Middle Base (9)",
    10: "Middle Mid (10)",
    11: "Middle Top (11)",
    12: "Middle Tip (12)",
    13: "Ring Base (13)",
    14: "Ring Mid (14)",
    15: "Ring Top (15)",
    16: "Ring Tip (16)",
    17: "Pinky Base (17)",
    18: "Pinky Mid (18)",
    19: "Pinky Top (19)",
    20: "Pinky Tip (20)",
}

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Normalized [0..1] 2D schematic coordinates for an open hand
EXTENDED_COORDS: Dict[int, Tuple[float, float]] = {
    0: (0.52, 0.88),
    # Thumb
    1: (0.37, 0.78),
    2: (0.26, 0.65),
    3: (0.18, 0.53),
    4: (0.11, 0.42),
    # Index
    5: (0.37, 0.54),
    6: (0.35, 0.37),
    7: (0.34, 0.23),
    8: (0.33, 0.11),
    # Middle
    9: (0.49, 0.51),
    10: (0.49, 0.33),
    11: (0.49, 0.19),
    12: (0.49, 0.07),
    # Ring
    13: (0.61, 0.54),
    14: (0.62, 0.37),
    15: (0.63, 0.23),
    16: (0.64, 0.11),
    # Pinky
    17: (0.72, 0.59),
    18: (0.75, 0.45),
    19: (0.78, 0.33),
    20: (0.80, 0.22),
}

# Curled/folded coordinates for each finger in "Show Fingers (Pose)" mode
CURLED_FINGER_COORDS: Dict[str, Dict[int, Tuple[float, float]]] = {
    "thumb": {
        1: (0.37, 0.78),
        2: (0.30, 0.68),
        3: (0.38, 0.62),
        4: (0.46, 0.63),
    },
    "index": {
        5: (0.37, 0.54),
        6: (0.36, 0.47),
        7: (0.38, 0.57),
        8: (0.40, 0.65),
    },
    "middle": {
        9: (0.49, 0.51),
        10: (0.49, 0.44),
        11: (0.50, 0.55),
        12: (0.50, 0.64),
    },
    "ring": {
        13: (0.61, 0.54),
        14: (0.61, 0.47),
        15: (0.60, 0.57),
        16: (0.59, 0.65),
    },
    "pinky": {
        17: (0.72, 0.59),
        18: (0.73, 0.52),
        19: (0.70, 0.61),
        20: (0.68, 0.68),
    },
}

FINGER_LANDMARKS: Dict[str, List[int]] = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

FINGER_DISPLAY_NAMES: Dict[str, str] = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "pinky": "Pinky",
}


class HandDiagramWidget(QWidget):
    """Interactive visual hand canvas for creating 2-point pinches or folding/extending finger poses."""

    def __init__(
        self,
        on_selection_changed: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(380, 400)
        self.setMouseTracking(True)
        self._on_selection_changed = on_selection_changed

        # Mode: "pose" (click fingers to extend/curl) or "pinch" (click 2 dots to connect)
        self.mode: str = "pose"

        # State for "pinch" mode
        self.selected_points: List[int] = [4, 8]
        self._hover_lm: Optional[int] = None

        # State for "pose" mode: True = Extended (Up), False = Curled (Down)
        self.finger_states: Dict[str, bool] = {
            "thumb": False,
            "index": True,
            "middle": False,
            "ring": False,
            "pinky": True,
        }
        self._hover_finger: Optional[str] = None

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()
        if self._on_selection_changed:
            self._on_selection_changed()

    def _current_coords(self) -> Dict[int, Tuple[float, float]]:
        if self.mode == "pinch":
            return dict(EXTENDED_COORDS)

        coords = {0: EXTENDED_COORDS[0]}
        for fname, lms in FINGER_LANDMARKS.items():
            if self.finger_states.get(fname, False):
                for lm in lms:
                    coords[lm] = EXTENDED_COORDS[lm]
            else:
                for lm in lms:
                    coords[lm] = CURLED_FINGER_COORDS[fname][lm]
        return coords

    def _to_px(self, norm_pt: Tuple[float, float]) -> QPointF:
        pad = 24.0
        w = max(1.0, self.width() - 2 * pad)
        h = max(1.0, self.height() - 2 * pad)
        return QPointF(pad + norm_pt[0] * w, pad + norm_pt[1] * h)

    def _finger_at_pos(self, pos: QPointF) -> Optional[str]:
        coords = self._current_coords()
        best_finger: Optional[str] = None
        best_dist = 32.0
        for fname, lms in FINGER_LANDMARKS.items():
            # Check distance to segments and joints of this finger
            for i in range(len(lms)):
                p1 = self._to_px(coords[lms[i]])
                d = math.hypot(pos.x() - p1.x(), pos.y() - p1.y())
                if d < best_dist:
                    best_dist = d
                    best_finger = fname
                if i + 1 < len(lms):
                    p2 = self._to_px(coords[lms[i + 1]])
                    mid = QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)
                    dm = math.hypot(pos.x() - mid.x(), pos.y() - mid.y())
                    if dm < best_dist:
                        best_dist = dm
                        best_finger = fname
        return best_finger

    def _landmark_at_pos(self, pos: QPointF) -> Optional[int]:
        coords = self._current_coords()
        best_lm: Optional[int] = None
        best_dist = 20.0
        for lm_idx, npt in coords.items():
            pt = self._to_px(npt)
            d = math.hypot(pos.x() - pt.x(), pos.y() - pt.y())
            if d < best_dist:
                best_dist = d
                best_lm = lm_idx
        return best_lm

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        if self.mode == "pose":
            f = self._finger_at_pos(pos)
            if f != self._hover_finger:
                self._hover_finger = f
                self.setCursor(
                    Qt.CursorShape.PointingHandCursor
                    if f is not None
                    else Qt.CursorShape.ArrowCursor
                )
                self.update()
        else:
            lm = self._landmark_at_pos(pos)
            if lm != self._hover_lm:
                self._hover_lm = lm
                self.setCursor(
                    Qt.CursorShape.PointingHandCursor
                    if lm is not None
                    else Qt.CursorShape.ArrowCursor
                )
                self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        if self.mode == "pose":
            fname = self._finger_at_pos(pos)
            if fname is not None:
                self.finger_states[fname] = not self.finger_states[fname]
                self.update()
                if self._on_selection_changed:
                    self._on_selection_changed()
        else:
            lm = self._landmark_at_pos(pos)
            if lm is not None:
                if lm in self.selected_points:
                    self.selected_points.remove(lm)
                else:
                    if len(self.selected_points) >= 2:
                        self.selected_points.pop(0)
                    self.selected_points.append(lm)
                self.update()
                if self._on_selection_changed:
                    self._on_selection_changed()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Dark card background
        painter.fillRect(self.rect(), QColor(24, 26, 32))

        coords = self._current_coords()
        pts = {idx: self._to_px(npt) for idx, npt in coords.items()}

        # Draw subtle palm polygon
        palm_poly = QPolygonF([pts[0], pts[1], pts[5], pts[9], pts[13], pts[17]])
        painter.setPen(QPen(QColor(70, 80, 95, 160), 2.0))
        painter.setBrush(QColor(42, 48, 60, 140))
        painter.drawPolygon(palm_poly)

        if self.mode == "pose":
            self._paint_pose_mode(painter, pts)
        else:
            self._paint_pinch_mode(painter, pts)

    def _paint_pose_mode(self, painter: QPainter, pts: Dict[int, QPointF]) -> None:
        # Draw each finger as a thick rounded capsule / skeleton that lights up when extended
        for fname, lms in FINGER_LANDMARKS.items():
            is_up = self.finger_states.get(fname, False)
            is_hovered = fname == self._hover_finger

            # Base bone from wrist/palm
            if is_up:
                capsule_color = QColor(0, 220, 170, 95 if is_hovered else 70)
                bone_color = QColor(60, 255, 190)
                node_color = QColor(0, 255, 160)
                badge_text = "UP"
                badge_color = QColor(80, 255, 190)
            else:
                capsule_color = QColor(90, 95, 110, 75 if is_hovered else 45)
                bone_color = QColor(110, 115, 130)
                node_color = QColor(130, 135, 150)
                badge_text = "FOLDED"
                badge_color = QColor(150, 155, 170)

            path = QPainterPath()
            path.moveTo(pts[lms[0]])
            for lm in lms[1:]:
                path.lineTo(pts[lm])

            # Outer finger body
            body_pen = QPen(
                capsule_color,
                28.0 if is_hovered else 24.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(body_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # Inner bone line
            bone_pen = QPen(bone_color, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(bone_pen)
            painter.drawPath(path)

            # Joints
            for lm in lms:
                r = 7.0 if lm == lms[-1] else 4.5
                painter.setPen(QPen(QColor(20, 24, 30), 1.5))
                painter.setBrush(node_color)
                painter.drawEllipse(pts[lm], r, r)

            # Finger label + UP/FOLDED badge near fingertip or knuckle
            anchor_pt = pts[lms[-1]] if is_up else pts[lms[1]]
            font = QFont("Segoe UI", 8, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(badge_color)
            label_y = anchor_pt.y() - 16.0 if is_up else anchor_pt.y() - 14.0
            painter.drawText(
                QPointF(anchor_pt.x() - 22.0, max(16.0, label_y)),
                f"{FINGER_DISPLAY_NAMES[fname]}",
            )
            sub_font = QFont("Segoe UI", 7)
            painter.setFont(sub_font)
            painter.drawText(
                QPointF(anchor_pt.x() - 18.0, max(28.0, label_y + 11.0)),
                badge_text,
            )

        # Wrist node
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
        painter.setBrush(QColor(90, 140, 200))
        painter.drawEllipse(pts[0], 6.0, 6.0)

        # Bottom hint
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor(190, 205, 225))
        painter.drawText(
            12,
            self.height() - 10,
            "Click directly on any finger above to toggle it UP (Extended) or FOLDED.",
        )

    def _paint_pinch_mode(self, painter: QPainter, pts: Dict[int, QPointF]) -> None:
        # Skeleton lines
        bone_pen = QPen(QColor(130, 145, 165), 2.5)
        painter.setPen(bone_pen)
        for a, b in HAND_CONNECTIONS:
            painter.drawLine(pts[a], pts[b])

        # Dashed connection line between the 2 selected points
        if len(self.selected_points) == 2:
            p_a = pts[self.selected_points[0]]
            p_b = pts[self.selected_points[1]]
            dash_pen = QPen(QColor(255, 195, 0), 3.0, Qt.PenStyle.DashLine)
            painter.setPen(dash_pen)
            painter.drawLine(p_a, p_b)

        # All 21 landmark dots
        font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font)
        for lm_idx, pt in pts.items():
            is_selected = lm_idx in self.selected_points
            is_hovered = lm_idx == self._hover_lm

            if is_selected:
                radius = 11.0
                fill = QColor(0, 240, 150)
                border = QColor(255, 255, 255)
            elif is_hovered:
                radius = 10.0
                fill = QColor(80, 200, 255)
                border = QColor(255, 255, 255)
            else:
                radius = 8.0
                fill = QColor(45, 115, 210)
                border = QColor(180, 200, 230)

            painter.setPen(QPen(border, 1.8))
            painter.setBrush(fill)
            painter.drawEllipse(pt, radius, radius)

            # Landmark number inside dot
            painter.setPen(QColor(15, 20, 25) if is_selected else QColor(245, 245, 250))
            text = str(lm_idx)
            offset_x = -3.5 if lm_idx < 10 else -6.5
            painter.drawText(QPointF(pt.x() + offset_x, pt.y() + 3.5), text)

        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor(190, 205, 225))
        painter.drawText(
            12,
            self.height() - 10,
            "Click any 2 numbered dots on the hand to trigger when they touch.",
        )


class CustomGestureDialog(QDialog):
    """Visual dialog for creating either a custom Finger Pose or a 2-Point Pinch gesture."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Custom Gesture (Additional Hand)")
        self.resize(450, 560)
        self.created_gesture: Optional[Dict[str, Any]] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Mode switcher
        mode_row = QHBoxLayout()
        self.pose_radio = QRadioButton("Show Fingers (Click Fingers to Fold / Extend)", self)
        self.pinch_radio = QRadioButton("Connect 2 Dots (Click 2 Points to Touch)", self)
        self.pose_radio.setChecked(True)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.pose_radio)
        mode_group.addButton(self.pinch_radio)
        mode_row.addWidget(self.pose_radio)
        mode_row.addWidget(self.pinch_radio)
        layout.addLayout(mode_row)

        # Interactive Hand Diagram
        self.diagram = HandDiagramWidget(on_selection_changed=self._update_auto_name, parent=self)
        layout.addWidget(self.diagram, stretch=1)

        self.pose_radio.toggled.connect(
            lambda checked: self.diagram.set_mode("pose" if checked else "pinch")
        )

        # Gesture Name field
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Gesture Name:", self))
        self.name_input = QLineEdit(self)
        name_row.addWidget(self.name_input, stretch=1)
        layout.addLayout(name_row)

        # Dialog buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save Gesture", self)
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._update_auto_name()

    def _update_auto_name(self) -> None:
        if self.diagram.mode == "pose":
            up_fingers = [
                FINGER_DISPLAY_NAMES[f]
                for f in ("thumb", "index", "middle", "ring", "pinky")
                if self.diagram.finger_states.get(f, False)
            ]
            if up_fingers:
                self.name_input.setText("Show: " + " + ".join(up_fingers))
            else:
                self.name_input.setText("Show: Closed Fist")
        else:
            pts = self.diagram.selected_points
            if len(pts) == 2:
                a, b = pts[0], pts[1]
                self.name_input.setText(f"Touch: {LANDMARK_NAMES[a]} + {LANDMARK_NAMES[b]}")
            elif len(pts) == 1:
                a = pts[0]
                self.name_input.setText(f"Select 2nd point ({LANDMARK_NAMES[a]} + ...)")
            else:
                self.name_input.setText("Select 2 points on the hand")

    def _on_save(self) -> None:
        name = self.name_input.text().strip()
        if self.diagram.mode == "pinch":
            if len(self.diagram.selected_points) != 2:
                QMessageBox.warning(
                    self,
                    "Select 2 Points",
                    "Please click two dots on the hand diagram to connect.",
                )
                return
            a, b = self.diagram.selected_points[0], self.diagram.selected_points[1]
            self.created_gesture = {
                "id": f"custom_pinch_{a}_{b}_{int(time.time() * 1000) % 100000}",
                "name": name or f"Touch ({a}) + ({b})",
                "type": "pinch",
                "point_a": a,
                "point_b": b,
                "threshold": 0.048,
            }
            self.accept()
        else:
            states = dict(self.diagram.finger_states)
            self.created_gesture = {
                "id": f"custom_pose_{int(time.time() * 1000) % 100000}",
                "name": name or "Custom Finger Pose",
                "type": "pose",
                "fingers": states,
            }
            self.accept()
