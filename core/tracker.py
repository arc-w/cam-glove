import math
import os
from pathlib import Path
import sys
import time
from typing import Optional
import urllib.request

# Suppress verbose MediaPipe / TensorFlow Lite C++ logs
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal

from core.filters import EMAFilter2D, OneEuroFilter2D

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = _BASE_DIR / "hand_landmarker.task"

# MediaPipe Hand Landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model(model_path: Path = MODEL_PATH) -> Path:
    if not model_path.exists():
        if getattr(sys, "frozen", False):
            model_path = Path(sys.executable).resolve().parent / "hand_landmarker.task"
        if not model_path.exists():
            print(f"Downloading MediaPipe model to {model_path}...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _norm_dist(lm1, lm2) -> float:
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)


def _draw_hand(
    frame,
    hand_lms,
    w: int,
    h: int,
    lmb_pinched: bool = False,
    rmb_pinched: bool = False,
    hold_pinched: bool = False,
    mmb_pinched: bool = False,
) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 100, 0), cv2.FILLED)

    # Highlight base knuckle of ring finger (RING_MCP = 13) for MMB / wheel press
    cv2.circle(frame, pts[RING_MCP], 6, (255, 220, 120), 2)

    # Highlight index fingertip (cursor driver) and active pinch state
    if hold_pinched:
        color = (0, 165, 255)
    elif mmb_pinched:
        color = (255, 255, 100)
    elif lmb_pinched:
        color = (0, 255, 120)
    elif rmb_pinched:
        color = (180, 90, 255)
    else:
        color = (255, 210, 0)
    cv2.circle(frame, pts[INDEX_TIP], 10, color, cv2.FILLED)


class HandTrackingWorker(QThread):
    """Background worker thread for webcam capture, MediaPipe hand tracking, and gesture detection."""

    cursor_moved = Signal(float, float)
    click_triggered = Signal(str, float, float)
    mouse_down = Signal(str, float, float)
    mouse_up = Signal(str, float, float)
    pinch_state_changed = Signal(str, bool)
    hand_presence_changed = Signal(bool)

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        camera_index: int = 0,
        pinch_threshold: float = 0.045,
        release_threshold: float = 0.065,
        mmb_pinch_threshold: float = 0.032,
        mmb_release_threshold: float = 0.050,
        margin_x: float = 0.15,
        margin_top: float = 0.10,
        margin_bottom: float = 0.38,
        use_one_euro: bool = True,
        ema_alpha: float = 0.25,
        show_preview: bool = True,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_index = camera_index
        self.pinch_threshold = pinch_threshold
        self.release_threshold = release_threshold
        self.mmb_pinch_threshold = mmb_pinch_threshold
        self.mmb_release_threshold = mmb_release_threshold
        self.margin_x = margin_x
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.show_preview = show_preview

        if use_one_euro:
            self.smoother = OneEuroFilter2D(min_cutoff=1.2, beta=0.015, d_cutoff=1.0)
        else:
            self.smoother = EMAFilter2D(alpha=ema_alpha)

        self._running = False
        self._hand_present = False
        self._lmb_pinched = False
        self._rmb_pinched = False
        self._hold_pinched = False
        self._mmb_pinched = False
        self._last_smooth_x = screen_width / 2.0
        self._last_smooth_y = screen_height / 2.0

    def stop(self) -> None:
        """Request the worker loop to terminate cleanly."""
        self._running = False
        self.wait(3000)

    def run(self) -> None:
        model_path = ensure_model()

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam (index {self.camera_index}).")
            return

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self._running = True
        last_ts_ms = 0
        failed_reads = 0
        lost_hand_frames = 0
        window_name = "Cam-Glove Preview (Press ESC or Q to exit)"

        try:
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                while self._running:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        failed_reads += 1
                        if failed_reads > 15:
                            print("Error: Lost webcam feed. Make sure no other app is locking the camera.")
                            break
                        time.sleep(0.03)
                        continue
                    failed_reads = 0

                    # Flip horizontally for natural mirror movement
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    now = time.monotonic()
                    ts_ms = max(last_ts_ms + 1, int(now * 1000))
                    last_ts_ms = ts_ms

                    results = landmarker.detect_for_video(mp_image, ts_ms)
                    if results.hand_landmarks:
                        lost_hand_frames = 0
                        if not self._hand_present:
                            self._hand_present = True
                            self.hand_presence_changed.emit(True)

                        hand_lms = results.hand_landmarks[0]
                        thumb_tip = hand_lms[THUMB_TIP]
                        index_tip = hand_lms[INDEX_TIP]
                        middle_tip = hand_lms[MIDDLE_TIP]
                        ring_mcp = hand_lms[RING_MCP]
                        ring_tip = hand_lms[RING_TIP]
                        pinky_tip = hand_lms[PINKY_TIP]

                        # 1. Index fingertip (landmark 8) -> smoothed cursor coordinates
                        # Asymmetric vertical bounds keep the palm & lower fingers inside the camera
                        # when the index fingertip reaches the bottom of the active zone.
                        span_x = max(1e-5, 1.0 - 2.0 * self.margin_x)
                        span_y = max(1e-5, 1.0 - self.margin_top - self.margin_bottom)
                        norm_x = _clamp((index_tip.x - self.margin_x) / span_x)
                        norm_y = _clamp((index_tip.y - self.margin_top) / span_y)

                        raw_x = norm_x * self.screen_width
                        raw_y = norm_y * self.screen_height
                        smooth_x, smooth_y = self.smoother.update(raw_x, raw_y, now)
                        self._last_smooth_x, self._last_smooth_y = smooth_x, smooth_y

                        self.cursor_moved.emit(smooth_x, smooth_y)

                        # 2. Single Left Click (LMB): Thumb tip (4) + Middle fingertip (12)
                        lmb_dist = _norm_dist(thumb_tip, middle_tip)
                        if lmb_dist < self.pinch_threshold and not self._lmb_pinched:
                            self._lmb_pinched = True
                            self.pinch_state_changed.emit("left", True)
                            self.click_triggered.emit("left", smooth_x, smooth_y)
                        elif lmb_dist > self.release_threshold and self._lmb_pinched:
                            self._lmb_pinched = False
                            self.pinch_state_changed.emit("left", False)

                        # 3. Single Right Click (RMB): Thumb tip (4) + Ring fingertip (16)
                        rmb_dist = _norm_dist(thumb_tip, ring_tip)
                        if rmb_dist < self.pinch_threshold and not self._rmb_pinched:
                            self._rmb_pinched = True
                            self.pinch_state_changed.emit("right", True)
                            self.click_triggered.emit("right", smooth_x, smooth_y)
                        elif rmb_dist > self.release_threshold and self._rmb_pinched:
                            self._rmb_pinched = False
                            self.pinch_state_changed.emit("right", False)

                        # 4. Click-and-Hold (Drag): Thumb tip (4) + Pinky fingertip (20)
                        hold_dist = _norm_dist(thumb_tip, pinky_tip)
                        if hold_dist < self.pinch_threshold and not self._hold_pinched:
                            self._hold_pinched = True
                            self.pinch_state_changed.emit("hold", True)
                            self.mouse_down.emit("left", smooth_x, smooth_y)
                        elif hold_dist > self.release_threshold and self._hold_pinched:
                            self._hold_pinched = False
                            self.pinch_state_changed.emit("hold", False)
                            self.mouse_up.emit("left", smooth_x, smooth_y)

                        # 5. Middle Mouse Button / Wheel Press (MMB): Thumb tip (4) + Ring base knuckle (13)
                        mmb_dist = _norm_dist(thumb_tip, ring_mcp)
                        other_pinch_active = (
                            self._lmb_pinched
                            or self._rmb_pinched
                            or self._hold_pinched
                            or min(lmb_dist, rmb_dist, hold_dist) < 0.075
                        )
                        if (
                            mmb_dist < self.mmb_pinch_threshold
                            and not self._mmb_pinched
                            and not other_pinch_active
                        ):
                            self._mmb_pinched = True
                            self.pinch_state_changed.emit("middle", True)
                            self.mouse_down.emit("middle", smooth_x, smooth_y)
                        elif mmb_dist > self.mmb_release_threshold and self._mmb_pinched:
                            self._mmb_pinched = False
                            self.pinch_state_changed.emit("middle", False)
                            self.mouse_up.emit("middle", smooth_x, smooth_y)

                        if self.show_preview:
                            _draw_hand(
                                frame,
                                hand_lms,
                                w,
                                h,
                                lmb_pinched=self._lmb_pinched,
                                rmb_pinched=self._rmb_pinched,
                                hold_pinched=self._hold_pinched,
                                mmb_pinched=self._mmb_pinched,
                            )
                    else:
                        if self._hand_present:
                            self._hand_present = False
                            self.hand_presence_changed.emit(False)

                        # Safety release if hand leaves frame while holding LMB or MMB
                        if self._hold_pinched or self._mmb_pinched:
                            lost_hand_frames += 1
                            if lost_hand_frames > 10:
                                if self._hold_pinched:
                                    self._hold_pinched = False
                                    self.pinch_state_changed.emit("hold", False)
                                    self.mouse_up.emit("left", self._last_smooth_x, self._last_smooth_y)
                                if self._mmb_pinched:
                                    self._mmb_pinched = False
                                    self.pinch_state_changed.emit("middle", False)
                                    self.mouse_up.emit("middle", self._last_smooth_x, self._last_smooth_y)
                        if self._lmb_pinched:
                            self._lmb_pinched = False
                            self.pinch_state_changed.emit("left", False)
                        if self._rmb_pinched:
                            self._rmb_pinched = False
                            self.pinch_state_changed.emit("right", False)

                    if self.show_preview:
                        # Draw the active screen mapping zone so user sees where screen edges map
                        box_x1 = int(self.margin_x * w)
                        box_y1 = int(self.margin_top * h)
                        box_x2 = int((1.0 - self.margin_x) * w)
                        box_y2 = int((1.0 - self.margin_bottom) * h)
                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (100, 220, 255), 1)

                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            self._running = False
                            break
        finally:
            cap.release()
            if self.show_preview:
                cv2.destroyAllWindows()

