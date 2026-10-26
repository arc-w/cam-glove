import signal
import sys
from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from actions.input_controller import InputController
from core.tracker import HandTrackingWorker
from gui.overlay import CursorOverlay

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None  # type: ignore[assignment]


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        geo = screen.geometry()
        screen_w, screen_h = geo.width(), geo.height()
        dpr = screen.devicePixelRatio()
    else:
        screen_w, screen_h = 1920, 1080
        dpr = 1.0

    # Initialize components
    overlay = CursorOverlay()
    input_controller = InputController(
        screen_width=screen_w,
        screen_height=screen_h,
        device_pixel_ratio=dpr,
    )
    tracker = HandTrackingWorker(
        screen_width=screen_w,
        screen_height=screen_h,
    )

    # Wire Qt Signals & Slots across threads
    tracker.cursor_moved.connect(overlay.update_cursor_position)
    tracker.cursor_moved.connect(input_controller.on_cursor_moved)
    tracker.hand_presence_changed.connect(overlay.on_hand_presence_changed)
    tracker.pinch_state_changed.connect(overlay.on_pinch_state_changed)
    tracker.mouse_down.connect(input_controller.on_mouse_down)
    tracker.mouse_up.connect(input_controller.on_mouse_up)
    tracker.click_triggered.connect(overlay.on_click_triggered)
    tracker.click_triggered.connect(input_controller.trigger_click)

    # Clean shutdown handler
    def shutdown() -> None:
        input_controller.release_all()
        if tracker.isRunning():
            tracker.stop()
        overlay.close()
        app.quit()

    tracker.finished.connect(shutdown)
    app.aboutToQuit.connect(input_controller.release_all)
    app.aboutToQuit.connect(tracker.stop)

    # Allow Ctrl+C in terminal to quit Qt event loop cleanly
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, shutdown))
    sigint_timer = QTimer()
    sigint_timer.setInterval(200)
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start()

    # Optional global ESC hotkey to exit easily while overlay is click-through
    kb_listener = None
    if pynput_keyboard is not None:
        def on_press(key) -> None:
            if key == pynput_keyboard.Key.esc:
                QTimer.singleShot(0, shutdown)

        kb_listener = pynput_keyboard.Listener(on_press=on_press)
        kb_listener.daemon = True
        kb_listener.start()

    overlay.show()
    tracker.start()

    print(
        "Cam-Glove started!\n"
        "  - Cursor: Index fingertip (landmark 8)\n"
        "  - Left Click (LMB): Pinch Thumb (4) + Middle (12)\n"
        "  - Right Click (RMB): Pinch Thumb (4) + Ring (16)\n"
        "  - Click & Hold (Drag): Pinch Thumb (4) + Pinky (20)\n"
        "  - Middle Mouse / Wheel Press (MMB): Touch Thumb (4) to Ring Base Knuckle (13)\n"
        "  - Exit: Press ESC anywhere, 'q' in preview, or Ctrl+C in terminal."
    )

    exit_code = app.exec()
    if kb_listener is not None:
        kb_listener.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
