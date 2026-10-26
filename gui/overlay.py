import ctypes
import ctypes.wintypes
import sys
import time
from typing import Optional
from PySide6.QtCore import QPointF, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

# Win32 constants for keeping the overlay above popup context menus without stealing focus
HWND_TOPMOST = ctypes.wintypes.HWND(-1)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_NOOWNERZORDER = 0x0200
SWP_NOSENDCHANGING = 0x0400

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


class CursorOverlay(QWidget):
    """Fullscreen transparent, click-through overlay rendering the virtual hand cursor at 60 FPS."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Cam-Glove Virtual Cursor Overlay")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
            | Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            self.setGeometry(geo)
            self._cursor_x = geo.width() / 2.0
            self._cursor_y = geo.height() / 2.0
        else:
            self.setGeometry(0, 0, 1920, 1080)
            self._cursor_x = 960.0
            self._cursor_y = 540.0

        self._hand_present = False
        self._lmb_active = False
        self._rmb_active = False
        self._hold_active = False
        self._mmb_active = False
        self._last_click_time = 0.0
        self._last_click_button = "left"
        self._tick_count = 0

        # 60 FPS repaint timer decoupling rendering from camera framerate
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(16)  # ~60 FPS
        self._render_timer.timeout.connect(self._on_frame_tick)
        self._render_timer.start()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_win32_topmost(init_exstyle=True)

    def _apply_win32_topmost(self, init_exstyle: bool = False) -> None:
        """Keep the overlay at the very top of the HWND_TOPMOST stack (above RMB context menus)."""
        if sys.platform != "win32":
            return
        hwnd = int(self.winId())
        if not hwnd:
            return
        user32 = ctypes.windll.user32
        if init_exstyle:
            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                ex_style
                | WS_EX_LAYERED
                | WS_EX_TRANSPARENT
                | WS_EX_TOPMOST
                | WS_EX_TOOLWINDOW
                | WS_EX_NOACTIVATE,
            )
        user32.SetWindowPos(
            ctypes.wintypes.HWND(hwnd),
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOSENDCHANGING,
        )

    def _on_frame_tick(self) -> None:
        self._tick_count += 1
        # Periodically re-assert HWND_TOPMOST (~10 times/sec) so newly opened popup menus stay behind cursor
        if self._tick_count % 6 == 0:
            self._apply_win32_topmost()
        self.update()

    @Slot(float, float)
    def update_cursor_position(self, x: float, y: float) -> None:
        self._cursor_x = x
        self._cursor_y = y

    @Slot(str, float, float)
    def on_click_triggered(self, button: str, x: float, y: float) -> None:
        self._cursor_x = x
        self._cursor_y = y
        self._last_click_button = button
        self._last_click_time = time.monotonic()
        # Immediately re-assert topmost after RMB/LMB click so context menu doesn't cover cursor
        QTimer.singleShot(30, self._apply_win32_topmost)
        QTimer.singleShot(100, self._apply_win32_topmost)

    @Slot(bool)
    def on_hand_presence_changed(self, present: bool) -> None:
        self._hand_present = present
        if not present:
            self._lmb_active = False
            self._rmb_active = False
            self._hold_active = False
            self._mmb_active = False
        self.update()

    @Slot(str, bool)
    def on_pinch_state_changed(self, button: str, active: bool) -> None:
        if button == "left":
            self._lmb_active = active
        elif button == "right":
            self._rmb_active = active
        elif button == "hold":
            self._hold_active = active
        elif button == "middle":
            self._mmb_active = active
        self._apply_win32_topmost()

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._hand_present:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        center = QPointF(self._cursor_x, self._cursor_y)

        # Determine active color scheme
        if self._hold_active:
            core_color = QColor(255, 175, 0, 245)
            glow_color = QColor(255, 160, 0, 150)
        elif self._mmb_active:
            core_color = QColor(255, 255, 110, 245)
            glow_color = QColor(255, 240, 80, 150)
        elif self._lmb_active:
            core_color = QColor(0, 255, 160, 240)
            glow_color = QColor(0, 255, 160, 140)
        elif self._rmb_active:
            core_color = QColor(255, 90, 180, 240)
            glow_color = QColor(255, 90, 180, 140)
        else:
            core_color = QColor(0, 210, 255, 235)
            glow_color = QColor(0, 180, 255, 110)

        # Outer radial glow
        glow_radius = 28.0
        gradient = QRadialGradient(center, glow_radius)
        gradient.setColorAt(0.0, glow_color)
        gradient.setColorAt(0.55, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 45))
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, glow_radius, glow_radius)

        # Expanding ripple ring on recent click
        elapsed = time.monotonic() - self._last_click_time
        if 0.0 <= elapsed < 0.28:
            progress = elapsed / 0.28
            ripple_radius = 12.0 + progress * 26.0
            alpha = int((1.0 - progress) * 220)
            ripple_color = (
                QColor(0, 255, 160, alpha)
                if self._last_click_button == "left"
                else QColor(255, 90, 180, alpha)
            )
            ripple_pen = QPen(ripple_color, 2.2)
            painter.setPen(ripple_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, ripple_radius, ripple_radius)

        # Crisp outer ring
        is_engaged = self._lmb_active or self._rmb_active or self._hold_active or self._mmb_active
        ring_radius = 8.5 if is_engaged else 12.0
        ring_pen = QPen(QColor(255, 255, 255, 220), 2.0)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, ring_radius, ring_radius)

        # Precision inner core dot
        dot_radius = 5.0 if is_engaged else 3.5
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(core_color)
        painter.drawEllipse(center, dot_radius, dot_radius)
