import argparse
import signal
import sys

if hasattr(sys, "set_lazy_imports_filter"):
    sys.set_lazy_imports_filter(lambda *_: False)

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from actions.input_controller import InputController
from core.tracker import HandTrackingWorker
from gui.main_window import MainWindow, create_app_icon
from gui.overlay import CursorOverlay
from gui.virtual_keyboard import VirtualKeyboardWindow

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser(description="Cam-Glove Hand Tracking Controller")
    parser.add_argument(
        "--main-hand",
        choices=["right", "left", "Right", "Left"],
        default="Right",
        help="Default main hand when both hands appear at the same time (default: Right)",
    )
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0], *qt_args])
    app.setQuitOnLastWindowClosed(False)
    app_icon = create_app_icon()
    app.setWindowIcon(app_icon)

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
    virtual_keyboard = VirtualKeyboardWindow()
    input_controller = InputController(
        screen_width=screen_w,
        screen_height=screen_h,
        device_pixel_ratio=dpr,
    )
    tracker = HandTrackingWorker(
        screen_width=screen_w,
        screen_height=screen_h,
        preferred_main_hand=args.main_hand,
        show_cv_window=False,
    )

    def dispatch_click(button: str, x: float, y: float) -> None:
        if not virtual_keyboard.handle_virtual_click(button, x, y):
            input_controller.trigger_click(button, x, y)

    def dispatch_mouse_down(button: str, x: float, y: float) -> None:
        if not virtual_keyboard.handle_virtual_mouse_down(button, x, y):
            input_controller.on_mouse_down(button, x, y)

    def dispatch_mouse_up(button: str, x: float, y: float) -> None:
        if not virtual_keyboard.handle_virtual_mouse_up(button, x, y):
            input_controller.on_mouse_up(button, x, y)

    # Wire Qt Signals & Slots across threads
    tracker.cursor_moved.connect(overlay.update_cursor_position)
    tracker.cursor_moved.connect(virtual_keyboard.on_virtual_cursor_moved)
    tracker.cursor_moved.connect(input_controller.on_cursor_moved)
    tracker.hand_presence_changed.connect(overlay.on_hand_presence_changed)
    tracker.pinch_state_changed.connect(overlay.on_pinch_state_changed)
    tracker.mouse_down.connect(dispatch_mouse_down)
    tracker.mouse_up.connect(dispatch_mouse_up)
    tracker.click_triggered.connect(overlay.on_click_triggered)
    tracker.click_triggered.connect(dispatch_click)
    tracker.secondary_action_triggered.connect(input_controller.trigger_secondary_action)
    input_controller.toggle_virtual_keyboard_requested.connect(virtual_keyboard.toggle_visibility)

    shutting_down = False
    tray_icon: QSystemTrayIcon | None = None

    def shutdown() -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        if tray_icon is not None:
            tray_icon.hide()
        input_controller.release_all()
        if tracker.isRunning():
            tracker.stop()
        virtual_keyboard.close()
        overlay.close()
        main_window.force_close()
        app.quit()

    main_window = MainWindow(
        tracker=tracker,
        on_close_callback=shutdown,
        on_toggle_keyboard=virtual_keyboard.toggle_visibility,
    )

    # System tray icon & menu for reopening, pausing, or fully quitting Cam-Glove
    tray_icon = QSystemTrayIcon(app_icon, app)
    tray_icon.setToolTip("Cam-Glove — Hand Tracking Control Center (Active)")

    tray_menu = QMenu()
    open_action = QAction("Open Cam-Glove", tray_menu)
    open_action.triggered.connect(main_window.restore_from_tray)
    tray_menu.addAction(open_action)

    pause_action = QAction("Pause Tracking (F4)", tray_menu)
    pause_action.triggered.connect(tracker.toggle_paused)
    tray_menu.addAction(pause_action)

    vk_action = QAction("Open / Hide Virtual Keyboard", tray_menu)
    vk_action.triggered.connect(virtual_keyboard.toggle_visibility)
    tray_menu.addAction(vk_action)

    swap_action = QAction("Swap Active Main Hand (F2)", tray_menu)
    swap_action.triggered.connect(tracker.swap_active_main_hand)
    tray_menu.addAction(swap_action)

    tray_menu.addSeparator()

    quit_action = QAction("Quit Cam-Glove", tray_menu)
    quit_action.triggered.connect(shutdown)
    tray_menu.addAction(quit_action)

    tray_icon.setContextMenu(tray_menu)

    def on_paused_changed(paused: bool) -> None:
        if paused:
            input_controller.release_all()
            pause_action.setText("Resume Tracking (F4)")
            tray_icon.setToolTip("Cam-Glove — Tracking PAUSED (Press F4 to Resume)")
        else:
            pause_action.setText("Pause Tracking (F4)")
            tray_icon.setToolTip("Cam-Glove — Hand Tracking Control Center (Active)")

    tracker.tracking_paused_changed.connect(on_paused_changed)

    def on_tray_activated(reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            main_window.restore_from_tray()

    tray_icon.activated.connect(on_tray_activated)
    tray_icon.show()

    tracker.finished.connect(shutdown)
    app.aboutToQuit.connect(input_controller.release_all)
    app.aboutToQuit.connect(tracker.stop)

    # Allow Ctrl+C in terminal to quit Qt event loop cleanly
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, shutdown))
    sigint_timer = QTimer()
    sigint_timer.setInterval(200)
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start()

    # Global hotkeys: F2 to swap Main Hand (Right <-> Left), F4 to Pause/Resume Tracking
    kb_listener = None
    if pynput_keyboard is not None:
        def on_press(key) -> None:
            if key == pynput_keyboard.Key.f2:
                tracker.swap_active_main_hand()
            elif key == pynput_keyboard.Key.f4:
                tracker.toggle_paused()

        kb_listener = pynput_keyboard.Listener(on_press=on_press)
        kb_listener.daemon = True
        kb_listener.start()

    overlay.show()
    main_window.show()
    tracker.start()

    exit_code = app.exec()
    if kb_listener is not None:
        kb_listener.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
