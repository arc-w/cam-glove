import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

if hasattr(sys, "set_lazy_imports_filter"):
    sys.set_lazy_imports_filter(lambda *_: False)

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
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

REF_PALM_SCALE = 0.145

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

DEFAULT_ROI_MARGINS: Dict[str, float] = {
    "left": 0.15,
    "right": 0.15,
    "top": 0.10,
    "bottom": 0.38,
}

SECONDARY_HOLD_FRAMES = 7
SAFE_UNLOCK_HOLD_FRAMES = 6


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


def _compute_hand_scale_factor(hand_lms) -> float:
    """
    Compute a depth-invariant scale multiplier relative to REF_PALM_SCALE (0.145)
    using rigid palm bone lengths (Wrist to Index/Middle/Ring/Pinky MCP knuckles).
    """
    w = hand_lms[WRIST]
    d_idx = _norm_dist(w, hand_lms[INDEX_MCP])
    d_mid = _norm_dist(w, hand_lms[MIDDLE_MCP])
    d_ring = _norm_dist(w, hand_lms[RING_MCP])
    d_pinky = _norm_dist(w, hand_lms[PINKY_MCP])
    palm_span = _norm_dist(hand_lms[INDEX_MCP], hand_lms[PINKY_MCP])
    palm_size = max(d_idx, d_mid, d_ring, d_pinky, palm_span * 1.35)
    return _clamp(palm_size / REF_PALM_SCALE, 0.55, 1.85)


def _are_same_physical_hand(h0, h1) -> bool:
    """
    Return True if two MediaPipe landmark sets represent duplicate detections on the same physical hand.
    Checks palm/wrist/finger proximity scaled by hand size as well as bounding-box overlap.
    """
    hand_size = max(
        _norm_dist(h0[WRIST], h0[MIDDLE_TIP]),
        _norm_dist(h1[WRIST], h1[MIDDLE_TIP]),
        0.12,
    )

    # 1. Wrist proximity
    if _norm_dist(h0[WRIST], h1[WRIST]) < max(0.10, 0.42 * hand_size):
        return True

    # 2. Palm center proximity (WRIST=0, INDEX_MCP=5, MIDDLE_MCP=9, RING_MCP=13, PINKY_MCP=17)
    palm_indices = (0, 5, 9, 13, 17)
    p0_x = sum(h0[i].x for i in palm_indices) / len(palm_indices)
    p0_y = sum(h0[i].y for i in palm_indices) / len(palm_indices)
    p1_x = sum(h1[i].x for i in palm_indices) / len(palm_indices)
    p1_y = sum(h1[i].y for i in palm_indices) / len(palm_indices)
    if math.hypot(p0_x - p1_x, p0_y - p1_y) < max(0.10, 0.45 * hand_size):
        return True

    # 3. Four-finger knuckle + tip alignment (catches duplicate skeleton with flipped thumb)
    finger_indices = (5, 8, 9, 12, 13, 16, 17, 20)
    avg_finger_dist = sum(_norm_dist(h0[i], h1[i]) for i in finger_indices) / len(finger_indices)
    if avg_finger_dist < max(0.09, 0.35 * hand_size):
        return True

    # 4. Bounding-box intersection over smaller box area
    xs0 = [lm.x for lm in h0]
    ys0 = [lm.y for lm in h0]
    xs1 = [lm.x for lm in h1]
    ys1 = [lm.y for lm in h1]
    x_left = max(min(xs0), min(xs1))
    y_top = max(min(ys0), min(ys1))
    x_right = min(max(xs0), max(xs1))
    y_bottom = min(max(ys0), max(ys1))
    if x_right > x_left and y_bottom > y_top:
        inter_area = (x_right - x_left) * (y_bottom - y_top)
        area0 = max(1e-6, (max(xs0) - min(xs0)) * (max(ys0) - min(ys0)))
        area1 = max(1e-6, (max(xs1) - min(xs1)) * (max(ys1) - min(ys1)))
        if inter_area / min(area0, area1) > 0.55:
            return True

    return False


def _select_better_single_hand(h0, h1, handedness_list=None):
    """When h0 and h1 are duplicates on one physical hand, pick the better-formed skeleton."""
    spread0 = _norm_dist(h0[THUMB_TIP], h0[PINKY_MCP])
    spread1 = _norm_dist(h1[THUMB_TIP], h1[PINKY_MCP])
    if spread1 > spread0 * 1.15:
        return h1
    if spread0 > spread1 * 1.15:
        return h0
    if handedness_list and len(handedness_list) >= 2:
        try:
            score0 = float(handedness_list[0][0].score)
            score1 = float(handedness_list[1][0].score)
            if score1 > score0:
                return h1
        except Exception:
            pass
    return h0


ACTION_DISPLAY_NAMES: Dict[str, str] = {
    "left_click": "LEFT CLICK",
    "double_click": "DOUBLE CLICK",
    "right_click": "RIGHT CLICK",
    "hold_lmb": "HOLD LMB",
    "hold_mmb": "SCROLL / MMB",
    "copy": "COPY (CTRL+C)",
    "paste": "PASTE (CTRL+V)",
    "undo": "UNDO (CTRL+Z)",
    "select_all": "SELECT ALL",
    "show_desktop": "SHOW DESKTOP",
    "task_view": "TASK VIEW",
    "screenshot": "SCREENSHOT",
    "open_osk": "VIRTUAL KEYBOARD",
    "scroll_up": "SCROLL UP",
    "scroll_down": "SCROLL DOWN",
    "media_play_pause": "PLAY / PAUSE",
    "media_next": "NEXT TRACK",
    "media_prev": "PREV TRACK",
    "vol_up": "VOLUME UP",
    "vol_down": "VOLUME DOWN",
    "vol_mute": "MUTE AUDIO",
}


def _format_action_banner(action: str) -> str:
    if action.startswith("exec:"):
        p = Path(action[5:].strip())
        return f"RUN: {p.stem or p.name}"
    return ACTION_DISPLAY_NAMES.get(action, action.upper())


def _draw_hand(
    frame,
    hand_lms,
    w: int,
    h: int,
    lmb_pinched: bool = False,
    rmb_pinched: bool = False,
    hold_pinched: bool = False,
    mmb_pinched: bool = False,
    stabilized: bool = False,
    locked: bool = False,
    unlock_progress: float = 0.0,
) -> None:
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
    bone_color = (120, 140, 175) if locked else (200, 200, 200)
    joint_color = (80, 140, 255) if locked else (255, 100, 0)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], bone_color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, joint_color, cv2.FILLED)

    if locked:
        wx, wy = pts[WRIST]
        if 0.05 < unlock_progress < 1.0:
            end_angle = int(360 * _clamp(unlock_progress, 0.0, 1.0))
            cv2.ellipse(
                frame,
                (wx, wy),
                (18, 18),
                -90,
                0,
                end_angle,
                (0, 220, 255),
                3,
            )
        return

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
    if stabilized:
        cv2.circle(frame, pts[INDEX_TIP], 14, (120, 255, 220), 1)


def _draw_secondary_hand(
    frame,
    hand_lms,
    w: int,
    h: int,
    active_label: Optional[str] = None,
    active_pinch_pair: Optional[Tuple[int, int]] = None,
    hold_progress: float = 0.0,
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
    if 0.05 < hold_progress < 1.0:
        end_angle = int(360 * _clamp(hold_progress, 0.0, 1.0))
        cv2.ellipse(
            frame,
            (wx, wy),
            (18, 18),
            -90,
            0,
            end_angle,
            (0, 255, 180),
            3,
        )
    elif hold_progress >= 1.0:
        cv2.circle(frame, (wx, wy), 18, (0, 255, 140), 2)

    label = f"2ND ({active_label})" if active_label else "2ND HAND"
    cv2.putText(
        frame,
        label,
        (max(10, wx - 40), min(h - 10, wy + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 255),
        2,
    )


def _is_finger_extended(hand_lms, mcp_idx: int, pip_idx: int, tip_idx: int) -> bool:
    """
    Orientation-resilient finger extension check combining radial wrist distance
    and bone uncurling ratio so tilted hands are still recognized accurately.
    """
    wrist = hand_lms[WRIST]
    mcp = hand_lms[mcp_idx]
    pip = hand_lms[pip_idx]
    tip = hand_lms[tip_idx]

    d_wrist_tip = _norm_dist(wrist, tip)
    d_wrist_pip = _norm_dist(wrist, pip)
    d_mcp_tip = _norm_dist(mcp, tip)
    d_mcp_pip = max(1e-5, _norm_dist(mcp, pip))

    radial_extended = d_wrist_tip > d_wrist_pip * 1.12 and (d_mcp_tip / d_mcp_pip) > 1.32
    vertical_extended = tip.y < pip.y - 0.012 and d_wrist_tip > d_wrist_pip * 1.03
    return radial_extended or vertical_extended


def _get_finger_states(hand_lms) -> Dict[str, bool]:
    """Return extended (True) vs curled (False) state for all 5 fingers."""
    idx_up = _is_finger_extended(hand_lms, INDEX_MCP, INDEX_PIP, INDEX_TIP)
    mid_up = _is_finger_extended(hand_lms, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
    ring_up = _is_finger_extended(hand_lms, RING_MCP, RING_PIP, RING_TIP)
    pinky_up = _is_finger_extended(hand_lms, PINKY_MCP, PINKY_PIP, PINKY_TIP)

    thumb_up = (
        _norm_dist(hand_lms[THUMB_TIP], hand_lms[PINKY_MCP])
        > _norm_dist(hand_lms[THUMB_MCP], hand_lms[PINKY_MCP]) * 1.20
        and _norm_dist(hand_lms[THUMB_TIP], hand_lms[INDEX_MCP])
        > _norm_dist(hand_lms[THUMB_IP], hand_lms[INDEX_MCP]) * 1.05
    )
    return {
        "thumb": thumb_up,
        "index": idx_up,
        "middle": mid_up,
        "ring": ring_up,
        "pinky": pinky_up,
    }


def _is_peace_sign(hand_lms) -> bool:
    """Return True if the hand is showing a Peace Sign (Index + Middle extended, Ring + Pinky curled)."""
    f_states = _get_finger_states(hand_lms)
    return (
        f_states["index"]
        and f_states["middle"]
        and not f_states["ring"]
        and not f_states["pinky"]
    )


def _detect_secondary_gesture(
    hand_lms,
    custom_gestures: List[Dict[str, Any]],
    hand_scale: float = 1.0,
    sensitivity_mult: float = 1.0,
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """Evaluate custom 2-point pinches, custom finger poses, and built-in poses on the 2nd hand."""
    # 1. Check custom 2-point pinch gestures first (depth-scaled)
    for cg in custom_gestures:
        if cg.get("type") == "pinch":
            pa = int(cg.get("point_a", 4))
            pb = int(cg.get("point_b", 8))
            base_thresh = float(cg.get("threshold", 0.048))
            thresh = base_thresh * hand_scale * sensitivity_mult
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
    tracking_paused_changed = Signal(bool)
    fps_updated = Signal(float)
    hand_activated = Signal(str)

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
        self.margin_left = margin_x
        self.margin_right = margin_x
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.preferred_main_hand = (
            "Left" if preferred_main_hand.strip().lower() == "left" else "Right"
        )
        self.show_cv_window = show_cv_window

        # User-tunable smoothing (1..100, default 50) & pinch sensitivity (50..150%, default 100)
        self.smoothing_level: int = 50
        self.pinch_sensitivity: int = 100
        self.pinch_stabilization: bool = True
        self.safe_hand_detection: bool = False
        self.activation_delay: float = 2.0
        self.preview_enabled: bool = True
        self.paused: bool = False

        self.main_bindings: Dict[str, str] = dict(DEFAULT_MAIN_BINDINGS)
        self.secondary_bindings: Dict[str, str] = dict(DEFAULT_SECONDARY_BINDINGS)
        self.custom_gestures: List[Dict[str, Any]] = []

        if use_one_euro:
            self.smoother = OneEuroFilter2D(min_cutoff=1.2, beta=0.015, d_cutoff=1.0)
        else:
            self.smoother = EMAFilter2D(alpha=ema_alpha)

        self._running = False
        self._reopen_camera = False
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
        self._last_click_anchor_time = 0.0

        # Multi-hand & safe-unlock tracking state
        self._main_wrist_pos: Optional[tuple[float, float]] = None
        self._swap_requested = False
        self._main_hand_unlocked: bool = False
        self._main_unlock_frames: int = 0
        self._main_activated_time: float = 0.0
        self._sec_hand_unlocked: bool = False
        self._sec_unlock_frames: int = 0
        self._sec_activated_time: float = 0.0
        self._sec_lost_frames: int = 0
        self._sec_candidate_gesture: Optional[str] = None
        self._sec_hold_frames = 0
        self._sec_neutral_frames = 0
        self._sec_latched_gesture: Optional[str] = None
        self._last_sec_banner = ""
        self._last_sec_banner_time = 0.0

    def set_safe_hand_detection(self, enabled: bool) -> None:
        new_val = bool(enabled)
        if self.safe_hand_detection != new_val:
            self.safe_hand_detection = new_val
            if new_val:
                self._main_hand_unlocked = False
                self._main_unlock_frames = 0
                self._main_activated_time = 0.0
                self._sec_hand_unlocked = False
                self._sec_unlock_frames = 0
                self._sec_activated_time = 0.0
                self._release_all_gestures()
                if self._hand_present:
                    self._hand_present = False
                    self.hand_presence_changed.emit(False)

    def set_activation_delay(self, seconds: float) -> None:
        """Set post-activation grace period (0.0s .. 10.0s) before gesture commands execute."""
        self.activation_delay = _clamp(float(seconds), 0.0, 10.0)

    def set_paused(self, paused: bool) -> None:
        if self.paused != paused:
            self.paused = paused
            if paused:
                self._main_hand_unlocked = False
                self._main_unlock_frames = 0
                self._sec_hand_unlocked = False
                self._sec_unlock_frames = 0
                self._release_all_gestures()
                if self._hand_present:
                    self._hand_present = False
                    self.hand_presence_changed.emit(False)
            self.tracking_paused_changed.emit(self.paused)

    def toggle_paused(self) -> bool:
        self.set_paused(not self.paused)
        return self.paused

    def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_enabled = bool(enabled)

    def set_camera_index(self, index: int) -> None:
        idx = max(0, int(index))
        if idx != self.camera_index:
            self.camera_index = idx
            self._reopen_camera = True

    def set_smoothing_level(self, level: int) -> None:
        """Map slider 1..100 (1 = ultra responsive, 50 = balanced, 100 = ultra smooth) to OneEuroFilter."""
        self.smoothing_level = int(_clamp(float(level), 1.0, 100.0))
        norm = (self.smoothing_level - 1) / 99.0  # 0.0 (fast) -> 1.0 (smooth)
        min_cutoff = 2.6 * ((0.30 / 2.6) ** norm)
        beta = 0.035 * ((0.004 / 0.035) ** norm)
        if isinstance(self.smoother, OneEuroFilter2D):
            self.smoother.set_params(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)

    def set_pinch_sensitivity(self, pct: int) -> None:
        """Set pinch threshold multiplier percentage (60% = tighter pinch, 140% = easier pinch)."""
        self.pinch_sensitivity = int(_clamp(float(pct), 60.0, 140.0))

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

    def set_mapping_margins(
        self,
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> None:
        """Update the webcam-to-desktop cursor mapping rectangle margins (normalized 0..1)."""
        left = _clamp(left, 0.0, 0.80)
        right = _clamp(right, 0.0, 0.80)
        top = _clamp(top, 0.0, 0.80)
        bottom = _clamp(bottom, 0.0, 0.80)
        if left + right > 0.88:
            right = max(0.0, 0.88 - left)
        if top + bottom > 0.88:
            bottom = max(0.0, 0.88 - top)
        self.margin_left = left
        self.margin_right = right
        self.margin_top = top
        self.margin_bottom = bottom
        self.margin_x = 0.5 * (left + right)

    def reset_mapping_margins(self) -> None:
        self.set_mapping_margins(
            DEFAULT_ROI_MARGINS["left"],
            DEFAULT_ROI_MARGINS["right"],
            DEFAULT_ROI_MARGINS["top"],
            DEFAULT_ROI_MARGINS["bottom"],
        )

    def reset_default_bindings(self) -> None:
        self.main_bindings = dict(DEFAULT_MAIN_BINDINGS)
        for k, v in DEFAULT_SECONDARY_BINDINGS.items():
            self.secondary_bindings[k] = v
        for cg in self.custom_gestures:
            gid = str(cg.get("id", ""))
            if gid and gid not in self.secondary_bindings:
                self.secondary_bindings[gid] = "none"
        self.reset_mapping_margins()
        self.set_smoothing_level(50)
        self.set_pinch_sensitivity(100)

    def _activate_action(self, action: str, x: float, y: float) -> None:
        if action == "left_click":
            self._last_click_anchor_time = time.monotonic()
            self.pinch_state_changed.emit("left", True)
            self.click_triggered.emit("left", x, y)
        elif action == "double_click":
            self._last_click_anchor_time = time.monotonic()
            self.pinch_state_changed.emit("left", True)
            self.click_triggered.emit("double", x, y)
        elif action == "right_click":
            self._last_click_anchor_time = time.monotonic()
            self.pinch_state_changed.emit("right", True)
            self.click_triggered.emit("right", x, y)
        elif action == "hold_lmb":
            self._last_click_anchor_time = time.monotonic()
            self.pinch_state_changed.emit("hold", True)
            self.mouse_down.emit("left", x, y)
        elif action == "hold_mmb":
            self.pinch_state_changed.emit("middle", True)
            self.mouse_down.emit("middle", x, y)
        elif action != "none":
            self.secondary_action_triggered.emit(action)

    def _deactivate_action(self, action: str, x: float, y: float) -> None:
        if action in ("left_click", "double_click"):
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
        fps_frame_count = 0
        fps_last_time = time.monotonic()
        current_fps = 0.0
        window_name = "Cam-Glove Preview"

        try:
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                while self._running:
                    if self._reopen_camera:
                        self._reopen_camera = False
                        cap.release()
                        new_cap = cv2.VideoCapture(self.camera_index)
                        if new_cap.isOpened():
                            cap = new_cap
                        else:
                            cap = cv2.VideoCapture(0)
                            self.camera_index = 0

                    ok, frame = cap.read()
                    if not ok or frame is None:
                        failed_reads += 1
                        if failed_reads > 15:
                            print("Error: Lost webcam feed. Make sure no other app is locking the camera.")
                            break
                        time.sleep(0.03)
                        continue
                    failed_reads = 0

                    now = time.monotonic()
                    fps_frame_count += 1
                    if now - fps_last_time >= 0.5:
                        current_fps = fps_frame_count / (now - fps_last_time)
                        fps_frame_count = 0
                        fps_last_time = now
                        self.fps_updated.emit(current_fps)

                    # Flip horizontally for natural mirror movement
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape

                    if self.paused:
                        if self.preview_enabled or self.show_cv_window:
                            overlay_dark = frame.copy()
                            cv2.rectangle(overlay_dark, (0, 0), (w, h), (15, 18, 24), cv2.FILLED)
                            cv2.addWeighted(overlay_dark, 0.55, frame, 0.45, 0, frame)
                            cv2.putText(
                                frame,
                                "TRACKING PAUSED (Press F4 to Resume)",
                                (max(16, w // 2 - 205), h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.68,
                                (80, 210, 255),
                                2,
                            )
                            rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            q_img = QImage(
                                rgb_preview.data,
                                w,
                                h,
                                3 * w,
                                QImage.Format.Format_RGB888,
                            ).copy()
                            self.frame_ready.emit(q_img)
                        time.sleep(0.03)
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    ts_ms = max(last_ts_ms + 1, int(now * 1000))
                    last_ts_ms = ts_ms

                    results = landmarker.detect_for_video(mp_image, ts_ms)
                    if results.hand_landmarks:
                        lost_hand_frames = 0

                        hands_list = list(results.hand_landmarks)
                        if len(hands_list) >= 2 and _are_same_physical_hand(
                            hands_list[0], hands_list[1]
                        ):
                            hands_list = [
                                _select_better_single_hand(
                                    hands_list[0],
                                    hands_list[1],
                                    getattr(results, "handedness", None),
                                )
                            ]
                        secondary_lms = None

                        if len(hands_list) == 1:
                            hand_lms = hands_list[0]
                            self._main_wrist_pos = (hand_lms[WRIST].x, hand_lms[WRIST].y)
                            self._swap_requested = False
                            self._sec_candidate_gesture = None
                            self._sec_hold_frames = 0
                            self._sec_latched_gesture = None
                            self._sec_lost_frames += 1
                            if self._sec_lost_frames > 10:
                                self._sec_hand_unlocked = False
                                self._sec_unlock_frames = 0
                        else:
                            self._sec_lost_frames = 0
                            h0, h1 = hands_list[0], hands_list[1]
                            w0 = (h0[WRIST].x, h0[WRIST].y)
                            w1 = (h1[WRIST].x, h1[WRIST].y)

                            # When Safe Hand Detection is on and neither hand is unlocked yet,
                            # the first hand to show the Peace Sign becomes the Main Hand.
                            if self.safe_hand_detection and not self._main_hand_unlocked:
                                p0 = _is_peace_sign(h0)
                                p1 = _is_peace_sign(h1)
                                if p0 and not p1:
                                    main_is_0 = True
                                    self._swap_requested = False
                                elif p1 and not p0:
                                    main_is_0 = False
                                    self._swap_requested = False
                                elif self._main_wrist_pos is not None:
                                    d0 = math.hypot(w0[0] - self._main_wrist_pos[0], w0[1] - self._main_wrist_pos[1])
                                    d1 = math.hypot(w1[0] - self._main_wrist_pos[0], w1[1] - self._main_wrist_pos[1])
                                    main_is_0 = d0 <= d1
                                else:
                                    main_is_0 = (w0[0] >= w1[0]) if self.preferred_main_hand == "Right" else (w0[0] < w1[0])
                            elif self._main_wrist_pos is not None:
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

                        # Safe Hand Detection: require Peace Sign (Index + Middle extended) to unlock Main Hand
                        main_unlock_progress = 1.0
                        main_warmup_remaining = 0.0
                        if not self.safe_hand_detection:
                            main_active = True
                            main_commands_ready = True
                        else:
                            if not self._main_hand_unlocked:
                                if _is_peace_sign(hand_lms):
                                    self._main_unlock_frames += 1
                                    if self._main_unlock_frames >= SAFE_UNLOCK_HOLD_FRAMES:
                                        self._main_hand_unlocked = True
                                        self._main_unlock_frames = 0
                                        self._main_activated_time = now
                                        self._last_sec_banner = "MAIN HAND: ACTIVATED"
                                        self._last_sec_banner_time = now
                                        self.hand_activated.emit("Main Hand")
                                else:
                                    self._main_unlock_frames = 0
                            main_active = self._main_hand_unlocked
                            if not main_active:
                                main_unlock_progress = min(
                                    1.0, self._main_unlock_frames / float(SAFE_UNLOCK_HOLD_FRAMES)
                                )
                                main_commands_ready = False
                            else:
                                main_warmup_remaining = max(
                                    0.0, self.activation_delay - (now - self._main_activated_time)
                                )
                                main_commands_ready = main_warmup_remaining <= 0.0

                        thumb_tip = hand_lms[THUMB_TIP]
                        index_tip = hand_lms[INDEX_TIP]
                        middle_tip = hand_lms[MIDDLE_TIP]
                        ring_mcp = hand_lms[RING_MCP]
                        ring_tip = hand_lms[RING_TIP]
                        pinky_tip = hand_lms[PINKY_TIP]

                        # Depth-invariant hand scale + user sensitivity multiplier
                        hand_scale = _compute_hand_scale_factor(hand_lms)
                        sens_mult = self.pinch_sensitivity / 100.0
                        eff_pinch = self.pinch_threshold * hand_scale * sens_mult
                        eff_release = self.release_threshold * hand_scale * sens_mult
                        eff_mmb_pinch = self.mmb_pinch_threshold * hand_scale * sens_mult
                        eff_mmb_release = self.mmb_release_threshold * hand_scale * sens_mult

                        is_stabilized = False
                        smooth_x, smooth_y = self._last_smooth_x, self._last_smooth_y

                        if main_active:
                            if not self._hand_present:
                                self._hand_present = True
                                self.hand_presence_changed.emit(True)

                            # Compute distances for Main Hand gestures
                            d_mid = _norm_dist(thumb_tip, middle_tip)
                            d_ring = _norm_dist(thumb_tip, ring_tip)
                            d_pinky = _norm_dist(thumb_tip, pinky_tip)
                            d_ring_mcp = _norm_dist(thumb_tip, ring_mcp)

                            # Pre-Pinch Cursor Stabilization ("Pinch Anchor"):
                            damping_scale = 1.0
                            if self.pinch_stabilization and main_commands_ready:
                                active_vals_now = set(self._gesture_active_action.values())
                                is_dragging = (
                                    "hold_lmb" in active_vals_now or "hold_mmb" in active_vals_now
                                )
                                min_click_dist = min(d_mid, d_ring)
                                if now - self._last_click_anchor_time < 0.11 and not is_dragging:
                                    damping_scale = 0.08
                                    is_stabilized = True
                                elif min_click_dist < eff_pinch * 1.60 and not is_dragging:
                                    ratio = _clamp(
                                        (min_click_dist - eff_pinch * 0.85) / (eff_pinch * 0.75),
                                        0.0,
                                        1.0,
                                    )
                                    damping_scale = 0.12 + 0.88 * ratio
                                    is_stabilized = damping_scale < 0.65

                            # 1. Index fingertip (landmark 8) -> smoothed cursor coordinates
                            span_x = max(1e-5, 1.0 - self.margin_left - self.margin_right)
                            span_y = max(1e-5, 1.0 - self.margin_top - self.margin_bottom)
                            norm_x = _clamp((index_tip.x - self.margin_left) / span_x)
                            norm_y = _clamp((index_tip.y - self.margin_top) / span_y)

                            raw_x = norm_x * self.screen_width
                            raw_y = norm_y * self.screen_height
                            smooth_x, smooth_y = self.smoother.update(
                                raw_x,
                                raw_y,
                                now,
                                damping_scale=damping_scale,
                            )
                            self._last_smooth_x, self._last_smooth_y = smooth_x, smooth_y

                            self.cursor_moved.emit(smooth_x, smooth_y)

                            if main_commands_ready:
                                self._update_main_gesture(
                                    "thumb_middle",
                                    d_mid,
                                    eff_pinch,
                                    eff_release,
                                    smooth_x,
                                    smooth_y,
                                )
                                self._update_main_gesture(
                                    "thumb_ring",
                                    d_ring,
                                    eff_pinch,
                                    eff_release,
                                    smooth_x,
                                    smooth_y,
                                )
                                self._update_main_gesture(
                                    "thumb_pinky",
                                    d_pinky,
                                    eff_pinch,
                                    eff_release,
                                    smooth_x,
                                    smooth_y,
                                )

                                fingertip_pinch_busy = (
                                    self._gesture_pinched["thumb_middle"]
                                    or self._gesture_pinched["thumb_ring"]
                                    or self._gesture_pinched["thumb_pinky"]
                                    or min(d_mid, d_ring, d_pinky) < 0.075 * hand_scale
                                )
                                self._update_main_gesture(
                                    "thumb_ring_mcp",
                                    d_ring_mcp,
                                    eff_mmb_pinch,
                                    eff_mmb_release,
                                    smooth_x,
                                    smooth_y,
                                    blocked=fingertip_pinch_busy,
                                )
                            else:
                                self._release_all_gestures()
                        else:
                            if self._hand_present:
                                self._hand_present = False
                                self.hand_presence_changed.emit(False)
                            self._release_all_gestures()

                        # Additional (2nd) Hand Gestures (Custom + Built-in + Safe Unlock & Time Margin)
                        sec_action_display = None
                        sec_pinch_pair = None
                        sec_hold_progress = 0.0
                        if secondary_lms is not None:
                            sec_scale = _compute_hand_scale_factor(secondary_lms)
                            if self.safe_hand_detection and not self._main_hand_unlocked:
                                # Wait for the first hand to activate as Main Hand before activating 2nd hand
                                self._sec_unlock_frames = 0
                                sec_action_display = "PEACE TO UNLOCK"
                            elif self.safe_hand_detection and not self._sec_hand_unlocked:
                                if _is_peace_sign(secondary_lms):
                                    self._sec_unlock_frames += 1
                                    sec_hold_progress = min(
                                        1.0,
                                        self._sec_unlock_frames / float(SAFE_UNLOCK_HOLD_FRAMES),
                                    )
                                    sec_action_display = "UNLOCKING..."
                                    if self._sec_unlock_frames >= SAFE_UNLOCK_HOLD_FRAMES:
                                        self._sec_hand_unlocked = True
                                        self._sec_unlock_frames = 0
                                        self._sec_activated_time = now
                                        init_g, _ = _detect_secondary_gesture(
                                            secondary_lms,
                                            self.custom_gestures,
                                            hand_scale=sec_scale,
                                            sensitivity_mult=sens_mult,
                                        )
                                        self._sec_latched_gesture = init_g or "two_fingers"
                                        self._sec_candidate_gesture = None
                                        self._sec_hold_frames = 0
                                        self._last_sec_banner = "2ND HAND: ACTIVATED"
                                        self._last_sec_banner_time = now
                                        self.hand_activated.emit("Alternative Hand")
                                else:
                                    self._sec_unlock_frames = 0
                                    sec_action_display = "PEACE TO UNLOCK"
                            else:
                                sec_warmup_remaining = (
                                    max(0.0, self.activation_delay - (now - self._sec_activated_time))
                                    if self.safe_hand_detection
                                    else 0.0
                                )
                                if sec_warmup_remaining > 0.0:
                                    self._sec_candidate_gesture = None
                                    self._sec_hold_frames = 0
                                    sec_action_display = f"READY IN {sec_warmup_remaining:.1f}s"
                                    if self.activation_delay > 0:
                                        sec_hold_progress = _clamp(
                                            1.0 - (sec_warmup_remaining / self.activation_delay),
                                            0.0,
                                            1.0,
                                        )
                                else:
                                    sec_gesture, sec_pinch_pair = _detect_secondary_gesture(
                                        secondary_lms,
                                        self.custom_gestures,
                                        hand_scale=sec_scale,
                                        sensitivity_mult=sens_mult,
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

                                        sec_hold_progress = min(
                                            1.0, self._sec_hold_frames / float(SECONDARY_HOLD_FRAMES)
                                        )

                                        if (
                                            self._sec_hold_frames == SECONDARY_HOLD_FRAMES
                                            and self._sec_latched_gesture != sec_gesture
                                        ):
                                            self._sec_latched_gesture = sec_gesture
                                            if mapped_action != "none":
                                                self._activate_action(mapped_action, smooth_x, smooth_y)
                                                self._deactivate_action(
                                                    mapped_action, smooth_x, smooth_y
                                                )
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

                        # Draw skeleton meshes and labels on preview frame only when preview is visible
                        if self.preview_enabled or self.show_cv_window:
                            active_vals = set(self._gesture_active_action.values())
                            _draw_hand(
                                frame,
                                hand_lms,
                                w,
                                h,
                                lmb_pinched="left_click" in active_vals or "double_click" in active_vals,
                                rmb_pinched="right_click" in active_vals,
                                hold_pinched="hold_lmb" in active_vals,
                                mmb_pinched="hold_mmb" in active_vals,
                                stabilized=is_stabilized,
                                locked=not main_active,
                                unlock_progress=main_unlock_progress,
                            )
                            mwx, mwy = int(hand_lms[WRIST].x * w), int(hand_lms[WRIST].y * h)
                            if not main_active:
                                main_lbl = "MAIN (PEACE TO UNLOCK)"
                                main_col = (80, 200, 255)
                            elif not main_commands_ready:
                                main_lbl = f"MAIN (READY IN {main_warmup_remaining:.1f}s)"
                                main_col = (80, 230, 255)
                            else:
                                main_lbl = "MAIN"
                                main_col = (100, 255, 180)
                            cv2.putText(
                                frame,
                                main_lbl,
                                (max(10, mwx - 35), min(h - 10, mwy + 25)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                main_col,
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
                                    hold_progress=sec_hold_progress,
                                )
                    else:
                        self._main_wrist_pos = None
                        self._sec_candidate_gesture = None
                        self._sec_hold_frames = 0
                        self._sec_latched_gesture = None
                        if self._hand_present:
                            self._hand_present = False
                            self.hand_presence_changed.emit(False)

                        lost_hand_frames += 1
                        self._sec_lost_frames += 1
                        if lost_hand_frames > 10:
                            self._main_hand_unlocked = False
                            self._main_unlock_frames = 0
                            self._sec_hand_unlocked = False
                            self._sec_unlock_frames = 0
                            if any(self._gesture_pinched.values()):
                                self._release_all_gestures()

                    if self.preview_enabled or self.show_cv_window:
                        # Draw active screen mapping zone (orange box + corner handles) and HUD text
                        box_x1 = int(self.margin_left * w)
                        box_y1 = int(self.margin_top * h)
                        box_x2 = int((1.0 - self.margin_right) * w)
                        box_y2 = int((1.0 - self.margin_bottom) * h)
                        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (40, 175, 255), 2)
                        for cx, cy in ((box_x1, box_y1), (box_x2, box_y1), (box_x1, box_y2), (box_x2, box_y2)):
                            cv2.rectangle(frame, (cx - 5, cy - 5), (cx + 5, cy + 5), (40, 175, 255), cv2.FILLED)

                        cv2.putText(
                            frame,
                            f"Default Main: {self.preferred_main_hand} (F2 swap | F4 pause)",
                            (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (220, 240, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"{current_fps:.0f} FPS",
                            (w - 78, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (140, 255, 190),
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
