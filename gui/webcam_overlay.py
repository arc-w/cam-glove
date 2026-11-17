import ctypes
import ctypes.wintypes
import sys
from typing import Callable, Optional

from PySide6.QtCore import QPoint, Qt, Slot
from PySide6.QtGui import QHideEvent, QImage, QMouseEvent, QPixmap, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

HWND_TOPMOST = ctypes.wintypes.HWND(-1)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040


class WebcamOverlayWindow(QWidget):
    """
    Always-on-top, non-focus-stealing floating webcam preview window showing the live hand mesh
    and visible (read-only) orange cursor mapping boundary.
    """

    SIZE_PRESETS = [
        (320, 265),
        (420, 340),
        (540, 430),
        (660, 520),
    ]

    def __init__(
        self,
        on_visibility_changed: Optional[Callable[[bool], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_visibility_changed = on_visibility_changed
        self._size_idx = 1
        self._drag_offset: Optional[QPoint] = None
        self._hand_dragging = False
        self._hovered_btn: Optional[QPushButton] = None

        self.setWindowTitle("Cam-Glove Floating Webcam")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        w, h = self.SIZE_PRESETS[self._size_idx]
        self.setFixedSize(w, h)
        self._position_top_right()
        self._build_ui()
        self._apply_styles()

    def _position_top_right(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            x = avail.x() + avail.width() - self.width() - 24
            y = avail.y() + 24
            self.move(max(0, x), max(0, y))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 2, 0)
        header.setSpacing(6)

        self.title_label = QLabel("📷 Cam-Glove Live Mesh (Drag to move)", self)
        self.title_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header.addWidget(self.title_label, stretch=1)

        self.shrink_btn = QPushButton("−", self)
        self.shrink_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shrink_btn.setFixedSize(26, 24)
        self.shrink_btn.setToolTip("Smaller window")
        self.shrink_btn.setProperty("role", "header_btn")
        self.shrink_btn.clicked.connect(lambda: self._step_size(-1))
        header.addWidget(self.shrink_btn)

        self.grow_btn = QPushButton("+", self)
        self.grow_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.grow_btn.setFixedSize(26, 24)
        self.grow_btn.setToolTip("Larger window")
        self.grow_btn.setProperty("role", "header_btn")
        self.grow_btn.clicked.connect(lambda: self._step_size(1))
        header.addWidget(self.grow_btn)

        self.close_btn = QPushButton("✕", self)
        self.close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_btn.setFixedSize(28, 24)
        self.close_btn.setToolTip("Close floating webcam")
        self.close_btn.setProperty("role", "close_btn")
        self.close_btn.clicked.connect(self.hide)
        header.addWidget(self.close_btn)

        root.addLayout(header)

        # Read-only preview label (orange box is drawn on the frame but cannot be dragged here)
        self.video_label = QLabel("Waiting for camera...", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_label.setStyleSheet(
            "background-color: #0e1117; color: #98a6be; border: 1px solid #263144; border-radius: 6px;"
        )
        root.addWidget(self.video_label, stretch=1)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            WebcamOverlayWindow {
                background-color: #141820;
                border: 2px solid #00c8ff;
                border-radius: 10px;
            }
            QLabel {
                color: #b8dcff;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton[role="header_btn"] {
                background-color: #1f2837;
                color: #e6f0ff;
                border: 1px solid #3a4863;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton[role="header_btn"]:hover, QPushButton[hand_hover="true"] {
                background-color: #2e425c;
                border: 2px solid #00e5ff;
                color: #ffffff;
            }
            QPushButton[role="close_btn"] {
                background-color: #4a1f28;
                border: 1px solid #ff5566;
                border-radius: 4px;
                color: #ffccd2;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton[role="close_btn"]:hover {
                background-color: #7a2838;
                border: 2px solid #ff7788;
                color: #ffffff;
            }
            """
        )

    def _step_size(self, delta: int) -> None:
        new_idx = max(0, min(len(self.SIZE_PRESETS) - 1, self._size_idx + delta))
        if new_idx != self._size_idx:
            self._size_idx = new_idx
            w, h = self.SIZE_PRESETS[self._size_idx]
            self.setFixedSize(w, h)
            self._apply_win32_noactivate()

    def _apply_win32_noactivate(self) -> None:
        if sys.platform != "win32":
            return
        hwnd = int(self.winId())
        if not hwnd:
            return
        user32 = ctypes.windll.user32
        ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            ex_style | WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        )
        user32.SetWindowPos(
            ctypes.wintypes.HWND(hwnd),
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_win32_noactivate()
        if self._on_visibility_changed is not None:
            self._on_visibility_changed(True)

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        if self._on_visibility_changed is not None:
            self._on_visibility_changed(False)

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self._apply_win32_noactivate()

    @Slot(QImage)
    def update_preview_frame(self, q_img: QImage) -> None:
        if not self.isVisible():
            return
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
        super().mouseReleaseEvent(event)

    def _button_at_global(self, x: float, y: float) -> Optional[QPushButton]:
        pt = QPoint(int(round(x)), int(round(y)))
        if not self.isVisible() or not self.frameGeometry().contains(pt):
            return None
        local_pt = self.mapFromGlobal(pt)
        w = self.childAt(local_pt)
        while w is not None and not isinstance(w, QPushButton):
            w = w.parentWidget()
        return w if isinstance(w, QPushButton) else None

    def on_virtual_cursor_moved(self, x: float, y: float) -> None:
        if not self.isVisible():
            return
        pt = QPoint(int(round(x)), int(round(y)))
        if self._hand_dragging and self._drag_offset is not None:
            self.move(pt - self._drag_offset)
            return

        btn = self._button_at_global(x, y)
        if btn is not self._hovered_btn:
            if self._hovered_btn is not None:
                self._hovered_btn.setProperty("hand_hover", "false")
                self._hovered_btn.style().unpolish(self._hovered_btn)
                self._hovered_btn.style().polish(self._hovered_btn)
            self._hovered_btn = btn
            if self._hovered_btn is not None:
                self._hovered_btn.setProperty("hand_hover", "true")
                self._hovered_btn.style().unpolish(self._hovered_btn)
                self._hovered_btn.style().polish(self._hovered_btn)

    def handle_virtual_click(self, button: str, x: float, y: float) -> bool:
        pt = QPoint(int(round(x)), int(round(y)))
        if not self.isVisible() or not self.frameGeometry().contains(pt):
            return False
        if button.lower() == "left":
            btn = self._button_at_global(x, y)
            if btn is not None:
                btn.animateClick()
        return True

    def handle_virtual_mouse_down(self, button: str, x: float, y: float) -> bool:
        pt = QPoint(int(round(x)), int(round(y)))
        if not self.isVisible() or not self.frameGeometry().contains(pt):
            return False
        if button.lower() == "left":
            btn = self._button_at_global(x, y)
            if btn is not None:
                btn.animateClick()
            else:
                self._hand_dragging = True
                self._drag_offset = pt - self.frameGeometry().topLeft()
        return True

    def handle_virtual_mouse_up(self, button: str, x: float, y: float) -> bool:
        if self._hand_dragging and button.lower() == "left":
            self._hand_dragging = False
            self._drag_offset = None
            return True
        pt = QPoint(int(round(x)), int(round(y)))
        return self.isVisible() and self.frameGeometry().contains(pt)
