"""
=============================================================================
REAL-TIME SPINAL & WALK-DIRECTION ANALYSIS SERVER
=============================================================================
Merges:
  • walk_direction_detector.py  — Frontal / Sagittal / Oblique classification
  • spinal_analysis_complete.py — Kyphosis, Lordosis, Trunk Lean (clinical)

Exposes a WebSocket + HTTP API so any web front-end can:
  1. Stream live webcam frames to the server  (POST /frame  — base64 JPEG)
  2. Receive per-frame JSON analysis results  (WebSocket /ws)
  3. Query the rolling clinical summary       (GET  /summary)

Install:
    pip install mediapipe opencv-python numpy flask flask-sock scipy

Run:
    python realtime_analysis_server.py               # webcam preview OFF
    python realtime_analysis_server.py --preview     # show OpenCV window too
    python realtime_analysis_server.py --port 5050   # change port (default 5000)

Web-app usage:
    POST /frame   body: {"image": "<base64-jpeg-string>"}
                  → returns JSON result immediately (REST fallback)

    WS  /ws       send: {"image": "<base64-jpeg-string>"}
                  receive: JSON result per frame

    GET /summary  → rolling clinical averages for the last N valid frames
=============================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os, sys, math, time, base64, warnings, logging, threading, argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

# ── Numerical / image ────────────────────────────────────────────────────────
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not installed. Install: pip install opencv-python")

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── MediaPipe (Tasks API, ≥ 0.10) ───────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as _mpy
from mediapipe.tasks.python import vision as _vis

# ── Flask + WebSocket ────────────────────────────────────────────────────────
from flask import Flask, request, jsonify
try:
    from flask_sock import Sock
    import json as _json
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    warnings.warn("flask-sock not installed — WebSocket endpoint disabled. "
                  "Install: pip install flask-sock")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

print(f"[INFO] mediapipe {mp.__version__}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PUBLISHED CLINICAL REFERENCE VALUES
# ═══════════════════════════════════════════════════════════════════════════════
class PublishedStandards:
    """Ohlendorf et al. (2020) Sci Rep + Tokuda et al. (2017) J Phys Ther Sci."""

    KYPHOSIS_MEAN, KYPHOSIS_TR_LO, KYPHOSIS_TR_HI = 51.08, 31.63, 70.53
    KYPHOSIS_CI_LO, KYPHOSIS_CI_HI                = 49.14, 53.01

    LORDOSIS_MEAN, LORDOSIS_TR_LO, LORDOSIS_TR_HI = 32.86, 15.25, 50.47
    LORDOSIS_CI_LO, LORDOSIS_CI_HI                = 31.11, 34.62

    SAGITTAL_INCL_MEAN   = -3.4
    SAGITTAL_INCL_TR_LO  = -8.47
    SAGITTAL_INCL_TR_HI  =  1.66

    GAIT_TRUNK_LEAN_NORMAL_MEAN = 1.0
    GAIT_TRUNK_LEAN_NORMAL_SD   = 1.5
    GAIT_TRUNK_LEAN_MOD_MEAN    = 11.0
    GAIT_TRUNK_LEAN_MOD_SD      = 1.0

    COBB_NORMAL_LO, COBB_NORMAL_HI = 20.0, 40.0
    COBB_MILD_HI                   = 60.0

    @classmethod
    def classify_cobb(cls, deg: float) -> str:
        if deg < cls.COBB_NORMAL_LO:   return "below_normal"
        if deg <= cls.COBB_NORMAL_HI:  return "normal"
        if deg <= cls.COBB_MILD_HI:    return "mild"
        return "severe"

    @classmethod
    def classify_kyphosis_ohlendorf(cls, deg: float) -> str:
        if deg < cls.KYPHOSIS_TR_LO:   return "below_tolerance_region"
        if deg <= cls.KYPHOSIS_TR_HI:  return "within_tolerance_region"
        return "above_tolerance_region"

    @classmethod
    def classify_lordosis_ohlendorf(cls, deg: float) -> str:
        if deg < cls.LORDOSIS_TR_LO:   return "below_tolerance_region"
        if deg <= cls.LORDOSIS_TR_HI:  return "within_tolerance_region"
        return "above_tolerance_region"

    @classmethod
    def classify_trunk_lean(cls, deg: float) -> str:
        normal_hi = cls.GAIT_TRUNK_LEAN_NORMAL_MEAN + 2 * cls.GAIT_TRUNK_LEAN_NORMAL_SD
        mod_hi    = cls.GAIT_TRUNK_LEAN_MOD_MEAN    + 2 * cls.GAIT_TRUNK_LEAN_MOD_SD
        if deg < 0:            return "contralateral_lean"
        if deg <= normal_hi:   return "normal_gait_range"
        if deg <= mod_hi:      return "modified_lean_gait_range"
        return "excessive_lean"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WALK DIRECTION CLASSIFICATION  (from walk_direction_detector.py)
# ═══════════════════════════════════════════════════════════════════════════════
# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP           = 23, 24
L_ELBOW, R_ELBOW       = 13, 14
L_KNEE, R_KNEE         = 25, 26
NOSE                   = 0
L_EAR, R_EAR           = 7,  8

@dataclass
class WalkResult:
    label:      str    # Frontal | Sagittal | Oblique
    confidence: float
    score:      float  # raw frontal score 0→1

WALK_COLORS = {
    "Frontal":  (0, 220, 120),
    "Sagittal": (30, 140, 255),
    "Oblique":  (255, 180, 50),
}

def classify_walk_direction(lms, W: int, H: int) -> WalkResult:
    """Classify walking direction from MediaPipe pose landmarks."""
    def xy(i):
        l = lms[i]
        v = l.visibility if l.visibility is not None else 1.0
        return l.x * W, l.y * H, v

    lsx, lsy, lsv = xy(L_SHOULDER); rsx, rsy, rsv = xy(R_SHOULDER)
    lhx, lhy, lhv = xy(L_HIP);      rhx, rhy, rhv = xy(R_HIP)
    lex, ley, lev = xy(L_ELBOW);    rex, rey, rev = xy(R_ELBOW)
    lkx, lky, lkv = xy(L_KNEE);     rkx, rky, rkv = xy(R_KNEE)
    nx,  ny,  _   = xy(NOSE)
    _,   _,   lev2 = xy(L_EAR);     _,   _,   rev2 = xy(R_EAR)

    bw   = max(lsx, rsx, lhx, rhx) - min(lsx, rsx, lhx, rhx) + 1e-6
    f1   = abs(rsx - lsx) / bw
    f2   = 1.0 - abs(np.mean([lsv, lhv, lev, lkv]) - np.mean([rsv, rhv, rev, rkv]))
    f3   = float(lev2 < 0.35 and rev2 < 0.35)
    f4   = abs(rhx - lhx) / bw
    f5   = max(0.0, 1.0 - abs(nx - (lsx + rsx) / 2) / bw * 2)

    score = 0.35 * f1 + 0.30 * f2 + 0.15 * f3 + 0.10 * f4 + 0.10 * f5

    if   score >= 0.62: label, conf = "Frontal",  score
    elif score <= 0.38: label, conf = "Sagittal", 1.0 - score
    else:               label, conf = "Oblique",  1.0 - abs(score - 0.5) * 2

    return WalkResult(label, round(conf, 3), round(score, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPINAL ANGLE CALCULATORS  (from spinal_analysis_complete.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _seg_inclination(p_upper: np.ndarray, p_lower: np.ndarray) -> float:
    dx = float(p_lower[0] - p_upper[0])
    dy = float(p_lower[1] - p_upper[1])
    if abs(dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, abs(dy)))


def compute_kyphosis(kps: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Robust Cobb / kyphosis angle.
    Returns (cobb_deg, trunk_lean_deg).
    Requires keypoints: C7, T3, T8, L1; optionally Sacrum.
    """
    required = ["C7", "T3", "T8", "L1"]
    for k in required:
        if k not in kps:
            raise ValueError(f"Missing keypoint: {k}")

    incl = {
        "C7_T3": _seg_inclination(kps["C7"],  kps["T3"]),
        "T3_T8": _seg_inclination(kps["T3"],  kps["T8"]),
        "T8_L1": _seg_inclination(kps["T8"],  kps["L1"]),
    }

    lean_deg = 0.0
    if "Sacrum" in kps:
        signed_lean = _seg_inclination(kps["C7"], kps["Sacrum"])
        lean_deg    = abs(signed_lean)
        incl        = {k: v - signed_lean for k, v in incl.items()}

    values = sorted(incl.values())
    cobb_deg = abs(values[-1] - values[0])
    return cobb_deg, lean_deg


def compute_lordosis(kps: Dict[str, np.ndarray]) -> Optional[float]:
    """Estimate lumbar lordosis from L1, L3, L5/Sacrum."""
    if "L1" not in kps:
        return None
    l1 = kps["L1"]

    def incl(p1, p2):
        dx = float(p2[0] - p1[0]); dy = float(p2[1] - p1[1])
        return math.degrees(math.atan2(dx, abs(dy))) if abs(dy) > 1e-6 else 0.0

    if "L3" in kps and "L5" in kps:
        return abs(incl(l1, kps["L3"]) - incl(kps["L3"], kps["L5"]))
    if "L3" in kps and "Sacrum" in kps:
        return abs(incl(l1, kps["L3"]) - incl(kps["L3"], kps["Sacrum"]))
    if "Sacrum" in kps:
        return abs(incl(l1, kps["Sacrum"]))
    return None


def compute_trunk_lean_from_pose(lms, W: int, H: int) -> float:
    """
    Estimate trunk lean from MediaPipe pose landmarks.
    Uses shoulder midpoint (C7 proxy) vs hip midpoint (Sacrum proxy).
    """
    def pt(i): return np.array([lms[i].x * W, lms[i].y * H])
    shoulder_mid = (pt(L_SHOULDER) + pt(R_SHOULDER)) / 2.0
    hip_mid      = (pt(L_HIP)      + pt(R_HIP))      / 2.0
    delta_x = float(shoulder_mid[0] - hip_mid[0])
    delta_y = abs(float(shoulder_mid[1] - hip_mid[1]))
    if delta_y < 1e-6:
        return 0.0
    return math.degrees(math.atan2(abs(delta_x), delta_y))


def estimate_spinal_angles_from_pose(lms, W: int, H: int) -> Dict[str, Any]:
    """
    Derive spinal angle proxies directly from standard MediaPipe landmarks.

    Because MediaPipe Pose does NOT provide vertebral keypoints, we use
    anatomical proxies:
        C7  proxy → shoulder midpoint  (top of trunk)
        T8  proxy → chest midpoint     (mid-trunk)
        L1  proxy → navel height       (thoracolumbar junction proxy)
        Sacrum proxy → hip midpoint

    These are approximations; the SpinePose model gives clinical-grade values.
    For a web-app real-time preview these are accurate enough to be useful.
    """
    def pt(i): return np.array([lms[i].x * W, lms[i].y * H])

    sh_L, sh_R = pt(L_SHOULDER), pt(R_SHOULDER)
    hi_L, hi_R = pt(L_HIP),      pt(R_HIP)

    shoulder_mid = (sh_L + sh_R) / 2.0
    hip_mid      = (hi_L + hi_R) / 2.0

    # Interpolated proxies along the trunk axis
    # T8  ≈ 40% down from shoulder to hip
    # L1  ≈ 65% down from shoulder to hip
    t8_proxy  = shoulder_mid + 0.40 * (hip_mid - shoulder_mid)
    l1_proxy  = shoulder_mid + 0.65 * (hip_mid - shoulder_mid)

    kps = {
        "C7":    shoulder_mid,
        "T3":    shoulder_mid + 0.20 * (hip_mid - shoulder_mid),
        "T8":    t8_proxy,
        "L1":    l1_proxy,
        "L3":    shoulder_mid + 0.80 * (hip_mid - shoulder_mid),
        "L5":    hip_mid + 0.10 * (hip_mid - shoulder_mid),
        "Sacrum": hip_mid,
    }

    try:
        cobb_deg, lean_deg = compute_kyphosis(kps)
    except ValueError:
        cobb_deg, lean_deg = 0.0, 0.0

    lordosis_deg = compute_lordosis(kps)
    trunk_lean   = compute_trunk_lean_from_pose(lms, W, H)

    return {
        "kyphosis_deg":   round(cobb_deg, 2),
        "lordosis_deg":   round(lordosis_deg, 2) if lordosis_deg else None,
        "trunk_lean_deg": round(trunk_lean, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLINICAL CLASSIFICATION & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def build_clinical_report(kyphosis: Optional[float],
                          lordosis: Optional[float],
                          trunk_lean: Optional[float]) -> Dict[str, Any]:
    """Return per-frame clinical classification dict."""
    std = PublishedStandards
    report: Dict[str, Any] = {}

    if kyphosis is not None:
        report["kyphosis"] = {
            "value_deg":         kyphosis,
            "reference_mean":    std.KYPHOSIS_MEAN,
            "tolerance_region":  [std.KYPHOSIS_TR_LO, std.KYPHOSIS_TR_HI],
            "class_mendeley":    std.classify_cobb(kyphosis),
            "class_ohlendorf":   std.classify_kyphosis_ohlendorf(kyphosis),
            "within_CI":         std.KYPHOSIS_CI_LO <= kyphosis <= std.KYPHOSIS_CI_HI,
        }

    if lordosis is not None:
        report["lordosis"] = {
            "value_deg":        lordosis,
            "reference_mean":   std.LORDOSIS_MEAN,
            "tolerance_region": [std.LORDOSIS_TR_LO, std.LORDOSIS_TR_HI],
            "class_ohlendorf":  std.classify_lordosis_ohlendorf(lordosis),
            "within_CI":        std.LORDOSIS_CI_LO <= lordosis <= std.LORDOSIS_CI_HI,
        }

    if trunk_lean is not None:
        report["trunk_lean"] = {
            "value_deg":      trunk_lean,
            "reference_mean": std.GAIT_TRUNK_LEAN_NORMAL_MEAN,
            "classification": std.classify_trunk_lean(trunk_lean),
        }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SKELETON DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
BONES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
]

def draw_skeleton(frame: np.ndarray, lms, W: int, H: int):
    pts = [(int(l.x * W), int(l.y * H)) for l in lms]
    for a, b in BONES:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 220), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 180, 255),    1)


def draw_hud(frame: np.ndarray,
             walk: WalkResult,
             spinal: Dict[str, Any],
             fps: float = 0.0):
    H, W = frame.shape[:2]
    c = WALK_COLORS.get(walk.label, (200, 200, 200))

    # Semi-transparent top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (W, 110), (8, 8, 8), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # Walk direction label
    cv2.putText(frame, walk.label, (14, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, c, 2, cv2.LINE_AA)

    # Confidence bar
    bx, by, bw, bh = W - 205, 16, 180, 18
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * walk.confidence), by + bh), c, -1)
    cv2.putText(frame, f"{walk.confidence * 100:.0f}%", (bx, by + bh + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, c, 1, cv2.LINE_AA)

    if fps > 0:
        cv2.putText(frame, f"FPS {fps:.1f}", (W - 88, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)

    # Spinal metrics (bottom bar)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, H - 42), (W, H), (8, 8, 8), -1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)

    kyph  = spinal.get("kyphosis_deg")
    lord  = spinal.get("lordosis_deg")
    trunk = spinal.get("trunk_lean_deg")

    info_parts = []
    if kyph  is not None: info_parts.append(f"Kyphosis {kyph:.1f}°")
    if lord  is not None: info_parts.append(f"Lordosis {lord:.1f}°")
    if trunk is not None: info_parts.append(f"TrunkLean {trunk:.1f}°")

    cv2.putText(frame, "  |  ".join(info_parts), (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, f"frontal_score={walk.score:.3f}", (14, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 110, 110), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MEDIAPIPE MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════
import urllib.request, tempfile

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

def get_model() -> str:
    for candidate in [
        "pose_landmarker_lite.task",
        os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task"),
    ]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    tmp = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
    log.info("Downloading MediaPipe pose model (~5 MB) …")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    log.info(f"Saved to {tmp}")
    return tmp


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — POSE DETECTOR (MediaPipe Tasks API)
# ═══════════════════════════════════════════════════════════════════════════════

class PoseDetector:
    """Thread-safe wrapper around MediaPipe PoseLandmarker (IMAGE mode)."""

    def __init__(self, conf: float = 0.5):
        model  = get_model()
        base   = _mpy.BaseOptions(model_asset_path=model)
        opts   = _vis.PoseLandmarkerOptions(
            base_options=base,
            running_mode=_vis.RunningMode.IMAGE,
            min_pose_detection_confidence=conf,
            min_tracking_confidence=conf,
        )
        self._lm   = _vis.PoseLandmarker.create_from_options(opts)
        self._lock = threading.Lock()

    def process(self, frame: np.ndarray):
        """
        Process a single BGR frame.
        Returns (annotated_frame, landmarks | None).
        """
        H, W = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            det = self._lm.detect(img)

        if not det.pose_landmarks:
            return frame, None

        lms = det.pose_landmarks[0]
        draw_skeleton(frame, lms, W, H)
        return frame, lms

    def close(self):
        self._lm.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ROLLING SUMMARY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════
BUFFER_SIZE = 60   # keep last N valid frames for rolling averages

class RollingBuffer:
    """Thread-safe circular buffer for smoothed clinical metrics."""

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._lock   = threading.Lock()
        self._kyph   = deque(maxlen=maxlen)
        self._lord   = deque(maxlen=maxlen)
        self._trunk  = deque(maxlen=maxlen)
        self._walk   = deque(maxlen=maxlen)
        self._scores = deque(maxlen=maxlen)

    def push(self, spinal: Dict, walk: WalkResult):
        with self._lock:
            if spinal.get("kyphosis_deg") is not None:
                self._kyph.append(spinal["kyphosis_deg"])
            if spinal.get("lordosis_deg") is not None:
                self._lord.append(spinal["lordosis_deg"])
            if spinal.get("trunk_lean_deg") is not None:
                self._trunk.append(spinal["trunk_lean_deg"])
            self._walk.append(walk.label)
            self._scores.append(walk.score)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            def safe_mean(d): return round(float(np.mean(list(d))), 2) if d else None
            def mode(d):
                if not d: return None
                from collections import Counter
                return Counter(d).most_common(1)[0][0]
            std = PublishedStandards
            k   = safe_mean(self._kyph)
            l   = safe_mean(self._lord)
            t   = safe_mean(self._trunk)
            return {
                "frames_in_buffer": len(self._kyph),
                "kyphosis": {
                    "mean_deg":        k,
                    "class_mendeley":  std.classify_cobb(k)            if k else None,
                    "class_ohlendorf": std.classify_kyphosis_ohlendorf(k) if k else None,
                },
                "lordosis": {
                    "mean_deg":        l,
                    "class_ohlendorf": std.classify_lordosis_ohlendorf(l) if l else None,
                },
                "trunk_lean": {
                    "mean_deg":        t,
                    "classification":  std.classify_trunk_lean(t) if t else None,
                },
                "walk_direction": {
                    "dominant_label":  mode(self._walk),
                    "mean_frontal_score": round(float(np.mean(list(self._scores))), 3)
                                          if self._scores else None,
                },
            }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CORE ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_frame(detector: PoseDetector,
                  frame: np.ndarray,
                  buffer: RollingBuffer,
                  preview: bool = False,
                  fps: float = 0.0) -> Dict[str, Any]:
    """
    Run the full pipeline on one BGR frame.
    Returns a JSON-serialisable result dict.
    """
    H, W   = frame.shape[:2]
    frame, lms = detector.process(frame)

    if lms is None:
        return {"status": "no_person_detected", "frame_size": [W, H]}

    # ── Walk direction (from walk_direction_detector.py) ─────────────────────
    walk   = classify_walk_direction(lms, W, H)

    # ── Spinal angles (from spinal_analysis_complete.py via proxies) ─────────
    spinal = estimate_spinal_angles_from_pose(lms, W, H)

    # ── Clinical report ───────────────────────────────────────────────────────
    clinical = build_clinical_report(
        spinal.get("kyphosis_deg"),
        spinal.get("lordosis_deg"),
        spinal.get("trunk_lean_deg"),
    )

    # ── Rolling buffer ────────────────────────────────────────────────────────
    buffer.push(spinal, walk)

    # ── Optional preview window ───────────────────────────────────────────────
    if preview and CV2_AVAILABLE:
        annotated = draw_hud(frame.copy(), walk, spinal, fps)
        cv2.imshow("Real-Time Analysis", annotated)
        cv2.waitKey(1)

    return {
        "status":       "ok",
        "walk_direction": {
            "label":      walk.label,
            "confidence": walk.confidence,
            "frontal_score": walk.score,
        },
        "spinal_angles": spinal,
        "clinical":      clinical,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FLASK WEB SERVER
# ═══════════════════════════════════════════════════════════════════════════════

app     = Flask(__name__)
_detector: Optional[PoseDetector] = None
_buffer:   Optional[RollingBuffer] = None
_preview:  bool = False
_fps_timer: Dict = {"prev": time.time(), "fps": 0.0}

if WEBSOCKET_AVAILABLE:
    sock = Sock(app)


def _decode_frame(b64_image: str) -> np.ndarray:
    """Decode a base64-encoded JPEG/PNG string to a BGR numpy array."""
    if "," in b64_image:                  # strip data-URI header if present
        b64_image = b64_image.split(",", 1)[1]
    raw   = base64.b64decode(b64_image)
    buf   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def _compute_fps() -> float:
    now = time.time()
    dt  = now - _fps_timer["prev"] + 1e-9
    _fps_timer["fps"]  = 1.0 / dt
    _fps_timer["prev"] = now
    return _fps_timer["fps"]


# ── REST endpoint: POST /frame ────────────────────────────────────────────────
@app.route("/frame", methods=["POST"])
def frame_endpoint():
    """
    Analyse a single frame.

    Request JSON:
        { "image": "<base64-encoded JPEG or PNG>" }

    Response JSON:
        { "status": "ok", "walk_direction": {...}, "spinal_angles": {...}, "clinical": {...} }
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in request body"}), 400

    try:
        frame = _decode_frame(data["image"])
    except Exception as e:
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    fps    = _compute_fps()
    result = analyse_frame(_detector, frame, _buffer, _preview, fps)
    return jsonify(result)


# ── WebSocket endpoint: WS /ws ────────────────────────────────────────────────
if WEBSOCKET_AVAILABLE:
    @sock.route("/ws")
    def ws_endpoint(ws):
        """
        Bidirectional WebSocket.
        Client sends: JSON {"image": "<base64-string>"}
        Server sends: JSON result per frame
        """
        log.info("[WS] Client connected")
        while True:
            try:
                raw = ws.receive()
                if raw is None:
                    break
                data = _json.loads(raw)
                if "image" not in data:
                    ws.send(_json.dumps({"error": "Missing 'image' field"}))
                    continue
                frame  = _decode_frame(data["image"])
                fps    = _compute_fps()
                result = analyse_frame(_detector, frame, _buffer, _preview, fps)
                ws.send(_json.dumps(result))
            except Exception as e:
                try:
                    ws.send(_json.dumps({"error": str(e)}))
                except Exception:
                    break
        log.info("[WS] Client disconnected")


# ── GET /summary ──────────────────────────────────────────────────────────────
@app.route("/summary", methods=["GET"])
def summary_endpoint():
    """Return rolling clinical averages for the last BUFFER_SIZE valid frames."""
    return jsonify(_buffer.summary())


# ── GET /health ───────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mediapipe_version": mp.__version__})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — OPTIONAL WEBCAM PREVIEW THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def _webcam_loop(device: int = 0):
    """
    Standalone webcam preview loop (runs in a separate thread).
    Displays annotated video locally while the server is also running.
    Only starts when --preview flag is set.
    """
    if not CV2_AVAILABLE:
        log.warning("OpenCV not available — webcam preview disabled")
        return

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error(f"Cannot open webcam device {device}")
        return

    log.info(f"[Preview] Webcam {device} opened — press Q or ESC to quit")
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("[Preview] Frame read failed")
            break

        now = time.time(); fps = 1 / (now - prev_t + 1e-9); prev_t = now
        analyse_frame(_detector, frame, _buffer, preview=True, fps=fps)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global _detector, _buffer, _preview

    ap = argparse.ArgumentParser(
        description="Real-Time Spinal + Walk Direction Analysis Server")
    ap.add_argument("--port",    type=int,   default=5000,
                    help="HTTP port (default 5000)")
    ap.add_argument("--host",    default="0.0.0.0",
                    help="Bind address (default 0.0.0.0)")
    ap.add_argument("--conf",    type=float, default=0.5,
                    help="MediaPipe detection confidence (default 0.5)")
    ap.add_argument("--preview", action="store_true",
                    help="Show annotated webcam window (requires OpenCV display)")
    ap.add_argument("--webcam",  type=int,   default=0,
                    help="Webcam device index (default 0); used only with --preview")
    ap.add_argument("--buffer",  type=int,   default=BUFFER_SIZE,
                    help=f"Rolling buffer size in frames (default {BUFFER_SIZE})")
    args = ap.parse_args()

    _preview  = args.preview
    _buffer   = RollingBuffer(maxlen=args.buffer)

    log.info("Initialising MediaPipe PoseLandmarker …")
    _detector = PoseDetector(conf=args.conf)
    log.info("PoseDetector ready.")

    # Optional local webcam preview (separate thread)
    if args.preview:
        t = threading.Thread(target=_webcam_loop, args=(args.webcam,), daemon=True)
        t.start()

    log.info(f"Server starting on http://{args.host}:{args.port}")
    log.info("  POST /frame    — send base64 image, receive analysis JSON")
    log.info("  WS   /ws       — WebSocket stream (frame in, JSON out)")
    log.info("  GET  /summary  — rolling clinical summary")
    log.info("  GET  /health   — health check")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
