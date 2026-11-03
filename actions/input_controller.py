import ctypes
import ctypes.wintypes
import os
import subprocess
import sys
from typing import Optional
from PySide6.QtCore import QObject, Slot

try:
    from pynput.keyboard import Controller as PynputKeyboardController, Key
    from pynput.mouse import Button, Controller as PynputMouseController
except ImportError:
    PynputMouseController = None  # type: ignore[assignment]
    PynputKeyboardController = None  # type: ignore[assignment]
    Button = None  # type: ignore[assignment]
    Key = None  # type: ignore[assignment]

# WinAPI mouse_event & keybd_event flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_LWIN = 0x5B
VK_C = 0x43
VK_D = 0x44
VK_O = 0x4F
VK_V = 0x56


class InputController(QObject):
    """Dispatches LMB/MMB click-and-hold and LMB/RMB clicks without hijacking the OS cursor when idle."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        device_pixel_ratio: float = 1.0,
        restore_cursor: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.screen_width = max(1, screen_width)
        self.screen_height = max(1, screen_height)
        self.device_pixel_ratio = device_pixel_ratio
        self.restore_cursor = restore_cursor
        self._is_windows = sys.platform == "win32"
        self._pynput_mouse = PynputMouseController() if PynputMouseController is not None else None
        self._pynput_kb = PynputKeyboardController() if PynputKeyboardController is not None else None
        self._lmb_held = False
        self._mmb_held = False

    def _to_abs_winapi(self, x: float, y: float) -> tuple[int, int]:
        abs_x = max(0, min(65535, int(round((x / self.screen_width) * 65535))))
        abs_y = max(0, min(65535, int(round((y / self.screen_height) * 65535))))
        return abs_x, abs_y

    @Slot(str, float, float)
    def on_mouse_down(self, button: str, x: float, y: float) -> None:
        """Press and hold a mouse button ('left' or 'middle') at virtual cursor coordinates (x, y)."""
        btn = button.lower()
        if btn == "left":
            if self._lmb_held:
                return
            self._lmb_held = True
            down_flag = MOUSEEVENTF_LEFTDOWN
            pynput_btn = Button.left if Button is not None else None
        elif btn == "middle":
            if self._mmb_held:
                return
            self._mmb_held = True
            down_flag = MOUSEEVENTF_MIDDLEDOWN
            pynput_btn = Button.middle if Button is not None else None
        else:
            return

        if self._is_windows:
            abs_x, abs_y = self._to_abs_winapi(x, y)
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | down_flag,
                abs_x,
                abs_y,
                0,
                0,
            )
        elif self._pynput_mouse is not None and pynput_btn is not None:
            phys_x = int(round(x * self.device_pixel_ratio))
            phys_y = int(round(y * self.device_pixel_ratio))
            self._pynput_mouse.position = (phys_x, phys_y)
            self._pynput_mouse.press(pynput_btn)

    @Slot(float, float)
    def on_cursor_moved(self, x: float, y: float) -> None:
        """Move the main OS cursor to (x, y) while the hand is tracked."""
        if self._is_windows:
            abs_x, abs_y = self._to_abs_winapi(x, y)
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE,
                abs_x,
                abs_y,
                0,
                0,
            )
        elif self._pynput_mouse is not None:
            phys_x = int(round(x * self.device_pixel_ratio))
            phys_y = int(round(y * self.device_pixel_ratio))
            self._pynput_mouse.position = (phys_x, phys_y)

    @Slot(str, float, float)
    def on_mouse_up(self, button: str, x: float, y: float) -> None:
        """Release a held mouse button ('left' or 'middle') at virtual cursor coordinates (x, y)."""
        btn = button.lower()
        if btn == "left":
            if not self._lmb_held:
                return
            self._lmb_held = False
            up_flag = MOUSEEVENTF_LEFTUP
            pynput_btn = Button.left if Button is not None else None
        elif btn == "middle":
            if not self._mmb_held:
                return
            self._mmb_held = False
            up_flag = MOUSEEVENTF_MIDDLEUP
            pynput_btn = Button.middle if Button is not None else None
        else:
            return

        if self._is_windows:
            abs_x, abs_y = self._to_abs_winapi(x, y)
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | up_flag,
                abs_x,
                abs_y,
                0,
                0,
            )
        elif self._pynput_mouse is not None and pynput_btn is not None:
            phys_x = int(round(x * self.device_pixel_ratio))
            phys_y = int(round(y * self.device_pixel_ratio))
            self._pynput_mouse.position = (phys_x, phys_y)
            self._pynput_mouse.release(pynput_btn)

    def release_all(self) -> None:
        """Ensure held buttons are released on application exit."""
        if self._lmb_held:
            self._lmb_held = False
            if self._is_windows:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif self._pynput_mouse is not None and Button is not None:
                self._pynput_mouse.release(Button.left)
        if self._mmb_held:
            self._mmb_held = False
            if self._is_windows:
                ctypes.windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
            elif self._pynput_mouse is not None and Button is not None:
                self._pynput_mouse.release(Button.middle)

    @Slot(str)
    def trigger_secondary_action(self, action: str) -> None:
        """Execute secondary hand actions ('copy', 'paste', 'show_desktop', 'open_osk', or 'exec:<path>')."""
        if action.startswith("exec:"):
            target_path = action[5:].strip()
            if target_path:
                self._launch_target(target_path)
            return

        act = action.lower()
        if act == "open_osk":
            self._open_virtual_keyboard()
            return

        if self._is_windows:
            user32 = ctypes.windll.user32
            if act == "copy":
                user32.keybd_event(VK_CONTROL, 0, 0, 0)
                user32.keybd_event(VK_C, 0, 0, 0)
                user32.keybd_event(VK_C, 0, KEYEVENTF_KEYUP, 0)
                user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
            elif act == "paste":
                user32.keybd_event(VK_CONTROL, 0, 0, 0)
                user32.keybd_event(VK_V, 0, 0, 0)
                user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
                user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
            elif act == "show_desktop":
                user32.keybd_event(VK_LWIN, 0, 0, 0)
                user32.keybd_event(VK_D, 0, 0, 0)
                user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
                user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
        elif self._pynput_kb is not None and Key is not None:
            if act == "copy":
                with self._pynput_kb.pressed(Key.ctrl):
                    self._pynput_kb.tap("c")
            elif act == "paste":
                with self._pynput_kb.pressed(Key.ctrl):
                    self._pynput_kb.tap("v")
            elif act == "show_desktop":
                with self._pynput_kb.pressed(Key.cmd):
                    self._pynput_kb.tap("d")

    def _open_virtual_keyboard(self) -> None:
        """Toggle Windows On-Screen Keyboard via native Win+Ctrl+O shortcut (avoids UAC/Wow64 issues)."""
        if self._is_windows:
            user32 = ctypes.windll.user32
            user32.keybd_event(VK_LWIN, 0, 0, 0)
            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            user32.keybd_event(VK_O, 0, 0, 0)
            user32.keybd_event(VK_O, 0, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
        elif self._pynput_kb is not None and Key is not None:
            with self._pynput_kb.pressed(Key.cmd):
                with self._pynput_kb.pressed(Key.ctrl):
                    self._pynput_kb.tap("o")

    def _launch_target(self, target_path: str) -> None:
        """Launch a user-specified executable, Start Menu shortcut (.lnk), or file."""
        try:
            if self._is_windows and hasattr(os, "startfile"):
                os.startfile(target_path)
            else:
                subprocess.Popen([target_path], close_fds=True)
        except Exception as exc:
            print(f"Error launching '{target_path}': {exc}")

    @Slot(str, float, float)
    def trigger_click(self, button: str, x: float, y: float) -> None:
        """Dispatch a single mouse click ('left' or 'right') at virtual cursor coordinates (x, y)."""
        btn = button.lower()
        if btn not in ("left", "right"):
            return

        if self._is_windows:
            self._click_winapi(btn, x, y)
        elif self._pynput_mouse is not None and Button is not None:
            self._click_pynput(btn, x, y)

    def _click_winapi(self, button: str, x: float, y: float) -> None:
        user32 = ctypes.windll.user32

        prev_pt = ctypes.wintypes.POINT()
        if self.restore_cursor:
            user32.GetCursorPos(ctypes.byref(prev_pt))

        abs_x, abs_y = self._to_abs_winapi(x, y)

        if button == "left":
            down_flag = MOUSEEVENTF_LEFTDOWN
            up_flag = MOUSEEVENTF_LEFTUP
        else:
            down_flag = MOUSEEVENTF_RIGHTDOWN
            up_flag = MOUSEEVENTF_RIGHTUP

        flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | down_flag | up_flag
        user32.mouse_event(flags, abs_x, abs_y, 0, 0)

        if self.restore_cursor:
            user32.SetCursorPos(prev_pt.x, prev_pt.y)

    def _click_pynput(self, button: str, x: float, y: float) -> None:
        if self._pynput_mouse is None or Button is None:
            return
        prev_pos = self._pynput_mouse.position if self.restore_cursor else None
        phys_x = int(round(x * self.device_pixel_ratio))
        phys_y = int(round(y * self.device_pixel_ratio))

        self._pynput_mouse.position = (phys_x, phys_y)
        pynput_btn = Button.left if button == "left" else Button.right
        self._pynput_mouse.click(pynput_btn, 1)

        if prev_pos is not None:
            self._pynput_mouse.position = prev_pos
