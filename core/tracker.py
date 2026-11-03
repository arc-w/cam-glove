import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

# Suppress verbose MediaPipe / TensorFlow Lite C++ logs
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

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
WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

DEFAULT_MAIN_BINDINGS: Dict[str, str] = {
    "thumb_middle": "left_click",
    "thumb_ring": "right_click",
    "thumb_pinky": "hold_lmb",
    "thumb_ring_mcp": "hold_mmb",
}

DEFAULT_SECONDARY_BINDINGS: Dict[str, str] = {
    "one_finger": "copy",
    "two_fingers": "paste",
    "three_fingers": "show_desktop",
}


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


def _format_action_banner(action: str) -> str:
    if action.startswith("exec:"):
        p = Path(action[5:].strip())
        return f"RUN: {p.stem or p.name}"
    if action == "open_osk":
        return "VIRTUAL KEYBOARD"
    return action.upper()


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


def _draw_secondary_hand(
    frame,
    hand_lms,
    w: int,
    h: int,
    active_label: Optional[str] = None,
    active_pinch_pair: Optional[Tuple[int, int]] = None,
) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (180, 140, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (200, 80, 255), cv2.FILLED)

    if active_pinch_pair is not None:
        pa, pb = active_pinch_pair
        cv2.line(frame, pts[pa], pts[pb], (0, 255, 180), 3)
        cv2.circle(frame, pts[pa], 7, (0, 255, 180), cv2.FILLED)
        cv2.circle(frame, pts[pb], 7, (0, 255, 180), cv2.FILLED)

    wx, wy = pts[WRIST]
    label = f"2ND ({active_label})" if active_label else "2ND HAND"
    cv2.putText(
        frame,
        label,
        (max(10, wx - 40), min(h - 10, wy + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 255),
        2,
    )


def _get_finger_states(hand_lms) -> Dict[str, bool]:
    """Return extended (True) vs curled (False) state for all 5 fingers."""
    idx_up = hand_lms[INDEX_TIP].y < hand_lms[INDEX_PIP].y - 0.015
    mid_up = hand_lms[MIDDLE_TIP].y < hand_lms[MIDDLE_PIP].y - 0.015
    ring_up = hand_lms[RING_TIP].y < hand_lms[RING_PIP].y - 0.015
    pinky_up = hand_lms[PINKY_TIP].y < hand_lms[PINKY_PIP].y - 0.015
    thumb_up = (
        _norm_dist(hand_lms[THUMB_TIP], hand_lms[PINKY_MCP])
        > _norm_dist(hand_lms[THUMB_MCP], hand_lms[PINKY_MCP]) * 1.22
    )
    return {
        "thumb": thumb_up,
        "index": idx_up,
        "middle": mid_up,
        "ring": ring_up,
        "pinky": pinky_up,
    }


def _detect_secondary_gesture(
    hand_lms,
    custom_gestures: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Evaluate custom 2-point pinches, custom finger poses, and built-in poses on the 2nd hand."""
    # 1. Check custom 2-point pinch gestures first
    for cg in custom_gestures:
        if cg.get("type") == "pinch":
            pa = int(cg.get("point_a", 4))
            pb = int(cg.get("point_b", 8))
            thresh = float(cg.get("threshold", 0.048))
            if 0 <= pa <= 20 and 0 <= pb <= 20:
                if _norm_dist(hand_lms[pa], hand_lms[pb]) < thresh:
                    return str(cg["id"]), (pa, pb)

    f_states = _get_finger_states(hand_lms)

    # 2. Check custom finger poses ("Show Fingers")
    for cg in custom_gestures:
        if cg.get("type") == "pose":
            target = cg.get("fingers", {})
            if (
                f_states["index"] == bool(target.get("index", False))
                and f_states["middle"] == bool(target.get("middle", False))
                and f_states["ring"] == bool(target.get("ring", False))
                and f_states["pinky"] == bool(target.get("pinky", False))
            ):
                want_thumb = bool(target.get("thumb", False))
                any_other_up = any(
                    bool(target.get(k, False)) for k in ("index", "middle", "ring", "pinky")
                )
                # Require thumb match if thumb was explicitly toggled UP or if pose is Fist/Thumb-only
                if want_thumb and not f_states["thumb"]:
                    continue
                if not any_other_up and f_states["thumb"] != want_thumb:
                    continue
                return str(cg["id"]), None

    # 3. Built-in 1, 2, 3 finger poses
    idx_up = f_states["index"]
    mid_up = f_states["middle"]
    ring_up = f_states["ring"]
    pinky_up = f_states["pinky"]

    if idx_up and not mid_up and not ring_up and not pinky_up:
        return "one_finger", None
    if idx_up and mid_up and not ring_up and not pinky_up:
        return "two_fingers", None
    if idx_up and mid_up and ring_up and not pinky_up:
        return "three_fingers", None
    return None, None


class HandTrackingWorker(QThread):
    """Background worker thread for webcam capture, MediaPipe hand tracking, and gesture detection."""

    cursor_moved = Signal(float, float)
    click_triggered = Signal(str, float, float)
    mouse_down = Signal(str, float, float)
    mouse_up = Signal(str, float, float)
    pinch_state_changed = Signal(str, bool)
    hand_presence_changed = Signal(bool)
    secondary_action_triggered = Signal(str)
    frame_ready = Signal(QImage)
    main_hand_changed = Signal(str)

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
        preferred_main_hand: str = "Right",
        use_one_euro: bool = True,
        ema_alpha: float = 0.25,
        show_cv_window: bool = False,
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
        self.preferred_main_hand = (
            "Left" if preferred_main_hand.strip().lower() == "left" else "Right"
        )
        self.show_cv_window = show_cv_window

        self.main_bindings: Dict[str, str] = dict(DEFAULT_MAIN_BINDINGS)
        self.secondary_bindings: Dict[str, str] = dict(DEFAULT_SECONDARY_BINDINGS)
        self.custom_gestures: List[Dict[str, Any]] = []

        if use_one_euro:
            self.smoother = OneEuroFilter2D(min_cutoff=1.2, beta=0.015, d_cutoff=1.0)
        else:
            self.smoother = EMAFilter2D(alpha=ema_alpha)

        self._running = False
        self._hand_present = False
        self._gesture_pinched: Dict[str, bool] = {
            "thumb_middle": False,
            "thumb_ring": False,
            "thumb_pinky": False,
            "thumb_ring_mcp": False,
        }
        self._gesture_active_action: Dict[str, str] = {}
        self._last_smooth_x = screen_width / 2.0
        self._last_smooth_y = screen_height / 2.0

        # Multi-hand tracking state
        self._main_wrist_pos: Optional[tuple[float, float]] = None
        self._swap_requested = False
        self._sec_candidate_gesture: Optional[str] = None
        self._sec_hold_frames = 0
        self._sec_neutral_frames = 0
        self._sec_latched_gesture: Optional[str] = None
        self._last_sec_banner = ""
        self._last_sec_banner_time = 0.0

    def set_preferred_main_hand(self, hand: str) -> None:
        new_hand = "Left" if hand.strip().lower() == "left" else "Right"
        if new_hand != self.preferred_main_hand:
            self.preferred_main_hand = new_hand
            self._swap_requested = True
            self.main_hand_changed.emit(self.preferred_main_hand)

    def swap_active_main_hand(self) -> str:
        """Toggle default main hand preference ('Right' <-> 'Left') and swap active hands if 2 are visible."""
        self.preferred_main_hand = "Left" if self.preferred_main_hand == "Right" else "Right"
        self._swap_requested = True
        self.main_hand_changed.emit(self.preferred_main_hand)
        return self.preferred_main_hand

    def toggle_main_hand(self) -> str:
        return self.swap_active_main_hand()

    def set_main_binding(self, gesture_key: str, action_key: str) -> None:
        self.main_bindings[gesture_key] = action_key

    def set_secondary_binding(self, gesture_key: str, action_key: str) -> None:
        self.secondary_bindings[gesture_key] = action_key

    def add_custom_gesture(self, gesture: Dict[str, Any], action_key: str = "none") -> None:
        gid = str(gesture["id"])
        self.custom_gestures = [g for g in self.custom_gestures if g.get("id") != gid]
        self.custom_gestures.append(gesture)
        self.secondary_bindings[gid] = action_key

    def remove_custom_gesture(self, gesture_id: str) -> None:
        self.custom_gestures = [g for g in self.custom_gestures if g.get("id") != gesture_id]
        self.secondary_bindings.pop(gesture_id, None)

    def reset_default_bindings(self) -> None:
        self.main_bindings = dict(DEFAULT_MAIN_BINDINGS)
        for k, v in DEFAULT_SECONDARY_BINDINGS.items():
            self.secondary_bindings[k] = v
        for cg in self.custom_gestures:
            gid = str(cg.get("id", ""))
            if gid and gid not in self.secondary_bindings:
                self.secondary_bindings[gid] = "none"

    def _activate_action(self, action: str, x: float, y: float) -> None:
        if action == "left_click":
            self.pinch_state_changed.emit("left", True)
            self.click_triggered.emit("left", x, y)
        elif action == "right_click":
            self.pinch_state_changed.emit("right", True)
            self.click_triggered.emit("right", x, y)
        elif action == "hold_lmb":
            self.pinch_state_changed.emit("hold", True)
            self.mouse_down.emit("left", x, y)
        elif action == "hold_mmb":
            self.pinch_state_changed.emit("middle", True)
            self.mouse_down.emit("middle", x, y)
        elif action in ("copy", "paste", "show_desktop", "open_osk") or action.startswith("exec:"):
            self.secondary_action_triggered.emit(action)

    def _deactivate_action(self, action: str, x: float, y: float) -> None:
        if action == "left_click":
            self.pinch_state_changed.emit("left", False)
        elif action == "right_click":
            self.pinch_state_changed.emit("right", False)
        elif action == "hold_lmb":
            self.pinch_state_changed.emit("hold", False)
            self.mouse_up.emit("left", x, y)
        elif action == "hold_mmb":
            self.pinch_state_changed.emit("middle", False)
            self.mouse_up.emit("middle", x, y)

    def _update_main_gesture(
        self,
        gesture_key: str,
        dist: float,
        pinch_thresh: float,
        release_thresh: float,
        smooth_x: float,
        smooth_y: float,
        blocked: bool = False,
    ) -> None:
        is_pinched = self._gesture_pinched[gesture_key]
        if dist < pinch_thresh and not is_pinched and not blocked:
            self._gesture_pinched[gesture_key] = True
            action = self.main_bindings.get(gesture_key, "none")
            self._gesture_active_action[gesture_key] = action
            self._activate_action(action, smooth_x, smooth_y)
        elif dist > release_thresh and is_pinched:
            self._gesture_pinched[gesture_key] = False
            action = self._gesture_active_action.pop(gesture_key, "none")
            self._deactivate_action(action, smooth_x, smooth_y)

    def _release_all_gestures(self) -> None:
        for g_key, pinched in list(self._gesture_pinched.items()):
            if pinched:
                self._gesture_pinched[g_key] = False
                action = self._gesture_active_action.pop(g_key, "none")
                self._deactivate_action(action, self._last_smooth_x, self._last_smooth_y)

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
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self._running = True
        last_ts_ms = 0
        failed_reads = 0
        lost_hand_frames = 0
        window_name = "Cam-Glove Preview"

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

                        hands_list = results.hand_landmarks
                        secondary_lms = None

                        if len(hands_list) == 1:
                            hand_lms = hands_list[0]
                            self._main_wrist_pos = (hand_lms[WRIST].x, hand_lms[WRIST].y)
                            self._swap_requested = False
                            self._sec_candidate_gesture = None
                            self._sec_hold_frames = 0
                            self._sec_latched_gesture = None
                        else:
                            h0, h1 = hands_list[0], hands_list[1]
                            w0 = (h0[WRIST].x, h0[WRIST].y)
                            w1 = (h1[WRIST].x, h1[WRIST].y)

                            if self._main_wrist_pos is not None:
                                d0 = math.hypot(w0[0] - self._main_wrist_pos[0], w0[1] - self._main_wrist_pos[1])
                                d1 = math.hypot(w1[0] - self._main_wrist_pos[0], w1[1] - self._main_wrist_pos[1])
                                if self._swap_requested:
                                    main_is_0 = d0 > d1
                                    self._swap_requested = False
                                else:
                                    main_is_0 = d0 <= d1
                            else:
                                if self.preferred_main_hand == "Right":
                                    main_is_0 = w0[0] >= w1[0]
                                else:
                                    main_is_0 = w0[0] < w1[0]
                                self._swap_requested = False

                            if main_is_0:
                                hand_lms, secondary_lms = h0, h1
                            else:
                                hand_lms, secondary_lms = h1, h0

                            self._main_wrist_pos = (hand_lms[WRIST].x, hand_lms[WRIST].y)

                        thumb_tip = hand_lms[THUMB_TIP]
                        index_tip = hand_lms[INDEX_TIP]
                        middle_tip = hand_lms[MIDDLE_TIP]
                        ring_mcp = hand_lms[RING_MCP]
                        ring_tip = hand_lms[RING_TIP]
                        pinky_tip = hand_lms[PINKY_TIP]

                        # 1. Index fingertip (landmark 8) -> smoothed cursor coordinates
                        span_x = max(1e-5, 1.0 - 2.0 * self.margin_x)
                        span_y = max(1e-5, 1.0 - self.margin_top - self.margin_bottom)
                        norm_x = _clamp((index_tip.x - self.margin_x) / span_x)
                        norm_y = _clamp((index_tip.y - self.margin_top) / span_y)

                        raw_x = norm_x * self.screen_width
                        raw_y = norm_y * self.screen_height
                        smooth_x, smooth_y = self.smoother.update(raw_x, raw_y, now)
                        self._last_smooth_x, self._last_smooth_y = smooth_x, smooth_y

                        self.cursor_moved.emit(smooth_x, smooth_y)

                        # Compute distances for Main Hand gestures
                        d_mid = _norm_dist(thumb_tip, middle_tip)
                        d_ring = _norm_dist(thumb_tip, ring_tip)
                        d_pinky = _norm_dist(thumb_tip, pinky_tip)
                        d_ring_mcp = _norm_dist(thumb_tip, ring_mcp)

                        self._update_main_gesture(
                            "thumb_middle",
                            d_mid,
                            self.pinch_threshold,
                            self.release_threshold,
                            smooth_x,
                            smooth_y,
                        )
                        self._update_main_gesture(
                            "thumb_ring",
                            d_ring,
                            self.pinch_threshold,
                            self.release_threshold,
                            smooth_x,
                            smooth_y,
                        )
                        self._update_main_gesture(
                            "thumb_pinky",
                            d_pinky,
                            self.pinch_threshold,
                            self.release_threshold,
                            smooth_x,
                            smooth_y,
                        )

                        fingertip_pinch_busy = (
                            self._gesture_pinched["thumb_middle"]
                            or self._gesture_pinched["thumb_ring"]
                            or self._gesture_pinched["thumb_pinky"]
                            or min(d_mid, d_ring, d_pinky) < 0.075
                        )
                        self._update_main_gesture(
                            "thumb_ring_mcp",
                            d_ring_mcp,
                            self.mmb_pinch_threshold,
                            self.mmb_release_threshold,
                            smooth_x,
                            smooth_y,
                            blocked=fingertip_pinch_busy,
                        )

                        # Additional (2nd) Hand Gestures (Custom + Built-in)
                        sec_action_display = None
                        sec_pinch_pair = None
                        if secondary_lms is not None:
                            sec_gesture, sec_pinch_pair = _detect_secondary_gesture(
                                secondary_lms,
                                self.custom_gestures,
                            )
                            if sec_gesture is not None:
                                mapped_action = self.secondary_bindings.get(sec_gesture, "none")
                                sec_action_display = (
                                    _format_action_banner(mapped_action)
                                    if mapped_action != "none"
                                    else sec_gesture
                                )
                                self._sec_neutral_frames = 0
                                if sec_gesture == self._sec_candidate_gesture:
                                    self._sec_hold_frames += 1
                                else:
                                    self._sec_candidate_gesture = sec_gesture
                                    self._sec_hold_frames = 1

                                if (
                                    self._sec_hold_frames == 8
                                    and self._sec_latched_gesture != sec_gesture
                                ):
                                    self._sec_latched_gesture = sec_gesture
                                    if mapped_action != "none":
                                        self._activate_action(mapped_action, smooth_x, smooth_y)
                                        self._deactivate_action(mapped_action, smooth_x, smooth_y)
                                        self._last_sec_banner = (
                                            f"2ND HAND: {_format_action_banner(mapped_action)}"
                                        )
                                        self._last_sec_banner_time = now
                            else:
                                self._sec_candidate_gesture = None
                                self._sec_hold_frames = 0
                                self._sec_neutral_frames += 1
                                if self._sec_neutral_frames >= 4:
                                    self._sec_latched_gesture = None

                        # Draw skeleton meshes and labels on preview frame
                        active_vals = set(self._gesture_active_action.values())
                        _draw_hand(
                            frame,
                            hand_lms,
                            w,
                            h,
                            lmb_pinched="left_click" in active_vals,
                            rmb_pinched="right_click" in active_vals,
                            hold_pinched="hold_lmb" in active_vals,
                            mmb_pinched="hold_mmb" in active_vals,
                        )
                        mwx, mwy = int(hand_lms[WRIST].x * w), int(hand_lms[WRIST].y * h)
                        cv2.putText(
                            frame,
                            "MAIN",
                            (max(10, mwx - 25), min(h - 10, mwy + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (100, 255, 180),
                            2,
                        )
                        if secondary_lms is not None:
                            _draw_secondary_hand(
                                frame,
                                secondary_lms,
                                w,
                                h,
                                active_label=sec_action_display,
                                active_pinch_pair=sec_pinch_pair,
                            )
                    else:
                        self._main_wrist_pos = None
                        self._sec_candidate_gesture = None
                        self._sec_hold_frames = 0
                        self._sec_latched_gesture = None
                        if self._hand_present:
                            self._hand_present = False
                            self.hand_presence_changed.emit(False)

                        if any(self._gesture_pinched.values()):
                            lost_hand_frames += 1
                            if lost_hand_frames > 10:
                                self._release_all_gestures()

                    # Draw active screen mapping zone and HUD text on preview frame
                    box_x1 = int(self.margin_x * w)
                    box_y1 = int(self.margin_top * h)
                    box_x2 = int((1.0 - self.margin_x) * w)
                    box_y2 = int((1.0 - self.margin_bottom) * h)
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (100, 220, 255), 1)

                    cv2.putText(
                        frame,
                        f"Default Main: {self.preferred_main_hand} (F2 to swap)",
                        (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (220, 240, 255),
                        2,
                    )
                    if now - self._last_sec_banner_time < 1.2 and self._last_sec_banner:
                        cv2.putText(
                            frame,
                            self._last_sec_banner,
                            (12, 52),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (120, 255, 180),
                            2,
                        )

                    # Emit annotated frame as QImage for the Qt main window preview
                    rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * w
                    q_img = QImage(
                        rgb_preview.data,
                        w,
                        h,
                        bytes_per_line,
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.frame_ready.emit(q_img)

                    if self.show_cv_window:
                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            self._running = False
                            break
        finally:
            self._release_all_gestures()
            cap.release()
            if self.show_cv_window:
                cv2.destroyAllWindows()
