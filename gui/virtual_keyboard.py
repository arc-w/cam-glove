import ctypes
import ctypes.wintypes
import sys
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Win32 extended styles so clicking the virtual keyboard never steals focus from the target app
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

VK_MAP: Dict[str, int] = {
    "BACKSPACE": 0x08,
    "TAB": 0x09,
    "ENTER": 0x0D,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "ALT": 0x12,
    "CAPSLOCK": 0x14,
    "ESC": 0x1B,
    "SPACE": 0x20,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "DELETE": 0x2E,
    "WIN": 0x5B,
}

EXTENDED_VKS = {0x25, 0x26, 0x27, 0x28, 0x2E, 0x5B}

SHIFT_MAP: Dict[str, str] = {
    "`": "~",
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
}

# Each key spec: (key_id, normal_label, width_stretch)
KEYBOARD_ROWS: List[List[Tuple[str, str, int]]] = [
    [
        ("`", "`", 10),
        ("1", "1", 10),
        ("2", "2", 10),
        ("3", "3", 10),
        ("4", "4", 10),
        ("5", "5", 10),
        ("6", "6", 10),
        ("7", "7", 10),
        ("8", "8", 10),
        ("9", "9", 10),
        ("0", "0", 10),
        ("-", "-", 10),
        ("=", "=", 10),
        ("BACKSPACE", "⌫ Back", 18),
    ],
    [
        ("TAB", "⇥ Tab", 14),
        ("q", "q", 10),
        ("w", "w", 10),
        ("e", "e", 10),
        ("r", "r", 10),
        ("t", "t", 10),
        ("y", "y", 10),
        ("u", "u", 10),
        ("i", "i", 10),
        ("o", "o", 10),
        ("p", "p", 10),
        ("[", "[", 10),
        ("]", "]", 10),
        ("\\", "\\", 14),
    ],
    [
        ("CAPS", "⇪ Caps", 17),
        ("a", "a", 10),
        ("s", "s", 10),
        ("d", "d", 10),
        ("f", "f", 10),
        ("g", "g", 10),
        ("h", "h", 10),
        ("j", "j", 10),
        ("k", "k", 10),
        ("l", "l", 10),
        (";", ";", 10),
        ("'", "'", 10),
        ("ENTER", "↵ Enter", 21),
    ],
    [
        ("SHIFT", "⇧ Shift", 21),
        ("z", "z", 10),
        ("x", "x", 10),
        ("c", "c", 10),
        ("v", "v", 10),
        ("b", "b", 10),
        ("n", "n", 10),
        ("m", "m", 10),
        (",", ",", 10),
        (".", ".", 10),
        ("/", "/", 10),
        ("UP", "↑", 12),
        ("SHIFT_R", "⇧ Shift", 15),
    ],
    [
        ("CTRL", "Ctrl", 14),
        ("WIN", "Win", 12),
        ("ALT", "Alt", 12),
        ("SPACE", "Space", 64),
        ("LEFT", "←", 12),
        ("DOWN", "↓", 12),
        ("RIGHT", "→", 12),
        ("DELETE", "Del", 12),
    ],
]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.wintypes.DWORD), ("u", _INPUT_UNION)]


def _send_unicode_char(ch: str) -> None:
    if sys.platform != "win32" or not ch:
        return
    code = ord(ch[0])
    down = _INPUT(
        type=INPUT_KEYBOARD,
        u=_INPUT_UNION(
            ki=_KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=None)
        ),
    )
    up = _INPUT(
        type=INPUT_KEYBOARD,
        u=_INPUT_UNION(
            ki=_KEYBDINPUT(
                wVk=0,
                wScan=code,
                dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                time=0,
                dwExtraInfo=None,
            )
        ),
    )
    arr = (_INPUT * 2)(down, up)
    ctypes.windll.user32.SendInput(2, ctypes.byref(arr), ctypes.sizeof(_INPUT))


def _tap_vk(vk: int) -> None:
    if sys.platform != "win32":
        return
    ext = KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_VKS else 0
    ctypes.windll.user32.keybd_event(vk, 0, ext, 0)
    ctypes.windll.user32.keybd_event(vk, 0, ext | KEYEVENTF_KEYUP, 0)


def _key_down_vk(vk: int) -> None:
    if sys.platform != "win32":
        return
    ext = KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_VKS else 0
    ctypes.windll.user32.keybd_event(vk, 0, ext, 0)


def _key_up_vk(vk: int) -> None:
    if sys.platform != "win32":
        return
    ext = KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_VKS else 0
    ctypes.windll.user32.keybd_event(vk, 0, ext | KEYEVENTF_KEYUP, 0)


class VirtualKeyboardWindow(QWidget):
    """
    Always-on-top, non-focus-stealing On-Screen Virtual Keyboard designed for hand-tracking clicks.
    Runs inside Cam-Glove's process so Windows UIPI never blocks synthetic hand cursor clicks.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cam-Glove Virtual Keyboard")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._shift_active = False
        self._caps_active = False
        self._ctrl_active = False
        self._alt_active = False
        self._drag_offset: Optional[QPoint] = None
        self._hand_dragging = False
        self._hovered_btn: Optional[QPushButton] = None
        self._key_buttons: Dict[str, List[QPushButton]] = {}

        self.setFixedSize(880, 320)
        self._position_bottom_center()
        self._build_ui()
        self._apply_styles()

    def _position_bottom_center(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            x = avail.x() + (avail.width() - self.width()) // 2
            y = avail.y() + avail.height() - self.height() - 24
            self.move(max(0, x), max(0, y))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        # Header bar (draggable + quick actions)
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 4, 2)
        header.setSpacing(8)

        self.title_label = QLabel("⌨  Cam-Glove Virtual Keyboard  (Drag bar to move)", self)
        self.title_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header.addWidget(self.title_label, stretch=1)

        for label, key_id in (("Esc", "ESC"), ("Copy", "COPY"), ("Paste", "PASTE")):
            btn = QPushButton(label, self)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setFixedHeight(28)
            btn.setMinimumWidth(64)
            btn.setProperty("role", "header_btn")
            btn.clicked.connect(lambda _checked=False, k=key_id: self._on_key_clicked(k))
            header.addWidget(btn)

        close_btn = QPushButton("✕ Close", self)
        close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_btn.setFixedHeight(28)
        close_btn.setMinimumWidth(72)
        close_btn.setProperty("role", "close_btn")
        close_btn.clicked.connect(self.hide)
        header.addWidget(close_btn)

        root.addLayout(header)

        # Keyboard rows
        for row_spec in KEYBOARD_ROWS:
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)
            for key_id, label, stretch in row_spec:
                btn = QPushButton(label, self)
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                btn.setMinimumHeight(46)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                norm_id = "SHIFT" if key_id == "SHIFT_R" else key_id
                self._key_buttons.setdefault(norm_id, []).append(btn)
                btn.clicked.connect(lambda _checked=False, k=norm_id: self._on_key_clicked(k))
                row_layout.addWidget(btn, stretch=stretch)
            root.addLayout(row_layout)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            VirtualKeyboardWindow {
                background-color: #141820;
                border: 2px solid #00c8ff;
                border-radius: 10px;
            }
            QLabel {
                color: #b8dcff;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #232936;
                color: #f0f4ff;
                border: 1px solid #384357;
                border-radius: 6px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover, QPushButton[hand_hover="true"] {
                background-color: #2e425c;
                border: 2px solid #00e5ff;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #00b8d4;
                color: #10141c;
            }
            QPushButton[toggled="true"] {
                background-color: #0088a8;
                border: 2px solid #50ffaa;
                color: #ffffff;
            }
            QPushButton[role="header_btn"] {
                background-color: #1d2633;
                font-size: 12px;
                padding: 2px 10px;
            }
            QPushButton[role="close_btn"] {
                background-color: #4a1f28;
                border: 1px solid #ff5566;
                color: #ffccd2;
                font-size: 12px;
                padding: 2px 10px;
            }
            QPushButton[role="close_btn"]:hover {
                background-color: #7a2838;
                border: 2px solid #ff7788;
                color: #ffffff;
            }
            """
        )

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
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

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self._apply_win32_noactivate()

    def _update_key_labels(self) -> None:
        upper = self._shift_active ^ self._caps_active
        for key_id, btns in self._key_buttons.items():
            if len(key_id) == 1 and key_id.isalpha():
                text = key_id.upper() if upper else key_id.lower()
                for b in btns:
                    b.setText(text)
            elif key_id in SHIFT_MAP:
                text = SHIFT_MAP[key_id] if self._shift_active else key_id
                for b in btns:
                    b.setText(text)

        for mod_key, active in (
            ("SHIFT", self._shift_active),
            ("CAPS", self._caps_active),
            ("CTRL", self._ctrl_active),
            ("ALT", self._alt_active),
        ):
            for b in self._key_buttons.get(mod_key, []):
                b.setProperty("toggled", "true" if active else "false")
                b.style().unpolish(b)
                b.style().polish(b)

    def _on_key_clicked(self, key_id: str) -> None:
        if key_id == "SHIFT":
            self._shift_active = not self._shift_active
            self._update_key_labels()
            return
        if key_id == "CAPS":
            self._caps_active = not self._caps_active
            self._update_key_labels()
            return
        if key_id == "CTRL":
            self._ctrl_active = not self._ctrl_active
            self._update_key_labels()
            return
        if key_id == "ALT":
            self._alt_active = not self._alt_active
            self._update_key_labels()
            return

        if key_id == "COPY":
            _key_down_vk(VK_MAP["CTRL"])
            _tap_vk(0x43)  # C
            _key_up_vk(VK_MAP["CTRL"])
            return
        if key_id == "PASTE":
            _key_down_vk(VK_MAP["CTRL"])
            _tap_vk(0x56)  # V
            _key_up_vk(VK_MAP["CTRL"])
            return

        if key_id in VK_MAP:
            self._dispatch_with_modifiers(vk=VK_MAP[key_id])
            return

        # Printable character
        if len(key_id) == 1:
            if key_id.isalpha() and (self._ctrl_active or self._alt_active):
                self._dispatch_with_modifiers(vk=ord(key_id.upper()))
            else:
                if key_id.isalpha():
                    upper = self._shift_active ^ self._caps_active
                    out_ch = key_id.upper() if upper else key_id.lower()
                elif self._shift_active and key_id in SHIFT_MAP:
                    out_ch = SHIFT_MAP[key_id]
                else:
                    out_ch = key_id
                _send_unicode_char(out_ch)
                if self._shift_active or self._ctrl_active or self._alt_active:
                    self._shift_active = False
                    self._ctrl_active = False
                    self._alt_active = False
                    self._update_key_labels()

    def _dispatch_with_modifiers(self, vk: int) -> None:
        if self._ctrl_active:
            _key_down_vk(VK_MAP["CTRL"])
        if self._alt_active:
            _key_down_vk(VK_MAP["ALT"])
        if self._shift_active:
            _key_down_vk(VK_MAP["SHIFT"])

        _tap_vk(vk)

        if self._shift_active:
            _key_up_vk(VK_MAP["SHIFT"])
            self._shift_active = False
        if self._alt_active:
            _key_up_vk(VK_MAP["ALT"])
            self._alt_active = False
        if self._ctrl_active:
            _key_up_vk(VK_MAP["CTRL"])
            self._ctrl_active = False
        self._update_key_labels()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 42:
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
        """Intercept hand clicks over the virtual keyboard so target window focus is never stolen."""
        pt = QPoint(int(round(x)), int(round(y)))
        if not self.isVisible() or not self.frameGeometry().contains(pt):
            return False
        if button.lower() == "left":
            btn = self._button_at_global(x, y)
            if btn is not None:
                btn.animateClick()
        return True

    def handle_virtual_mouse_down(self, button: str, x: float, y: float) -> bool:
        """Support hand click-and-hold on keys or dragging the virtual keyboard title bar."""
        pt = QPoint(int(round(x)), int(round(y)))
        if not self.isVisible() or not self.frameGeometry().contains(pt):
            return False
        if button.lower() == "left":
            btn = self._button_at_global(x, y)
            if btn is not None:
                btn.animateClick()
            else:
                local_pt = self.mapFromGlobal(pt)
                if local_pt.y() <= 46:
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

