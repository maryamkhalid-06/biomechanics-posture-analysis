"""
Shoulder Alignment Analyzer — SpinePose Edition
================================================
Measures frontal-plane shoulder alignment angles from video using SpinePose's
37-keypoint model (which includes clavicles for richer shoulder analysis).

Keypoints used (SpineTrack indices):
  5  = left_shoulder
  6  = right_shoulder
  18 = neck
  33 = left_clavicle
  34 = right_clavicle
  19 = hip (mid-pelvis)
  11 = left_hip
  12 = right_hip

Camera view modes
-----------------
  --view front   (default)
      Standard front-facing camera.  The person faces the camera.
      In this case the image is a MIRROR of the real world:
        • anatomical LEFT  → appears on the RIGHT side of the frame (larger x)
        • anatomical RIGHT → appears on the LEFT  side of the frame (smaller x)
      Formula used:  angle = atan2( -(rs_y - ls_y),  rs_x - ls_x )
      (vector goes from anatomical-left to anatomical-right in image space,
       i.e. from RIGHT side of frame to LEFT side of frame)

  --view back
      Rear-facing camera.  The person's back faces the camera.
      No mirror flip: anatomical LEFT is on the LEFT of the frame.
        • anatomical LEFT  → smaller x
        • anatomical RIGHT → larger x
      Formula used:  angle = atan2( -(ls_y - rs_y),  ls_x - rs_x )
      (vector goes from anatomical-right to anatomical-left in image space)

  Both modes produce the same sign convention for the result:
    positive angle → left shoulder is higher than right
    negative angle → right shoulder is higher than left

Angles computed per frame:
  • shoulder_tilt_deg   : angle of the shoulder line vs. horizontal
  • clavicle_tilt_deg   : same but using clavicle endpoints
  • left_shoulder_height  : left shoulder Y relative to neck (pixels)
  • right_shoulder_height : right shoulder Y relative to neck (pixels)
  • shoulder_imbalance  : left_height − right_height  (+ = left higher)

Outputs:
  • <output_dir>/annotated_video.mp4   – frame-by-frame overlay
  • <output_dir>/shoulder_angles.csv   – per-frame numeric data
  • <output_dir>/angle_plot.png        – time-series graphs
  • <output_dir>/summary.txt           – statistical summary
  • <output_dir>/symmetry_bar_chart.png – severity bar chart
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec

# ── SpinePose import ────────────────────────────────────────────────────────
try:
    from spinepose import SpinePoseEstimator as _SpinePoseEstimator
    SPINEPOSE_AVAILABLE = True
except Exception as _spine_import_err:
    _SpinePoseEstimator = None
    SPINEPOSE_AVAILABLE = False
    import warnings as _w
    _w.warn(
        f"\n[!] Could not import SpinePose.\n"
        f"    Error: {_spine_import_err}\n\n"
        "    Possible fix — make sure NO file in this folder is named\n"
        "    spinepose.py or shoulder_analyzer.py (they shadow the package).\n"
        f"    Also verify: {sys.executable} -m pip show spinepose\n"
        "    Shoulder analysis via SpinePose will be unavailable.\n"
    )

# ── Keypoint indices ────────────────────────────────────────────────────────
KP = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "neck": 18,
    "hip": 19,
    "left_clavicle": 33,
    "right_clavicle": 34,
}

SCORE_THRESHOLD = 0.3
SMOOTH_WINDOW   = 7


# ── Geometry helpers ─────────────────────────────────────────────────────────

def select_primary_detection(keypoints, scores):
    """
    Select the highest-confidence person from a SpinePose result.

    Returns (None, None) when the detector produced no usable detections.
    """
    if keypoints is None or scores is None:
        return None, None

    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    if keypoints.size == 0 or scores.size == 0:
        return None, None

    if getattr(keypoints, "ndim", 0) == 3:
        if keypoints.shape[0] == 0 or scores.shape[0] == 0:
            return None, None
        person_scores = scores.sum(axis=-1)
        if person_scores.size == 0:
            return None, None
        best = int(np.argmax(person_scores))
        return keypoints[best], scores[best]

    return keypoints, scores


def angle_with_horizontal(p1, p2):
    """Signed angle (degrees) of segment p1→p2 w.r.t. horizontal.
    Positive = p2 is above p1 (image Y is flipped, so we negate dy).
    """
    dx = p2[0] - p1[0]
    dy = -(p2[1] - p1[1])
    return math.degrees(math.atan2(dy, dx))


def shoulder_tilt(ls, rs, view):
    """
    Compute shoulder tilt with correct sign convention regardless of camera view.

    Front camera (mirrored image):
      - anatomical LEFT  shoulder has LARGER x in the frame
      - anatomical RIGHT shoulder has SMALLER x in the frame
      - We build the vector from the image-right point (rs in frame = anatomical left)
        to the image-left point (ls in frame = anatomical right)... wait, that's
        confusing. Let's think in anatomical terms only:
        We want: positive result when anatomical LEFT is higher than RIGHT.
        In a front (mirror) image:
          ls keypoint (label=left) → RIGHT side of image (larger x)
          rs keypoint (label=right)→ LEFT  side of image (smaller x)
        Vector from rs→ls goes LEFT to RIGHT in the image = positive x direction.
        atan2( -(ls_y-rs_y), ls_x-rs_x )
        If ls is higher (smaller y) than rs:  dy = -(ls_y - rs_y) > 0  → positive ✓

    Back camera (non-mirrored):
      - anatomical LEFT  shoulder has SMALLER x in the frame
      - anatomical RIGHT shoulder has LARGER  x in the frame
      We want the same sign convention (positive = anatomical left higher).
      Vector from rs→ls goes RIGHT to LEFT in the image = negative x direction.
      atan2( -(ls_y-rs_y), ls_x-rs_x )
      If ls is higher (smaller y) than rs:  dy = -(ls_y-rs_y) > 0,
        but dx = ls_x - rs_x < 0  → angle near 180° — WRONG.
      Fix: reverse the vector direction: use rs→ls becomes ls→rs
        atan2( -(rs_y-ls_y), rs_x-ls_x )
        If ls is higher: rs_y > ls_y → dy = -(rs_y-ls_y) < 0  → WRONG again.

    Cleaner approach: always express as "how many degrees is anatomical-left
    higher than anatomical-right" by computing the signed height difference
    and dividing by half the shoulder width.  But that's not an angle.

    Simplest correct approach:
      In BOTH views, we want: angle of the shoulder line measured such that
      positive = anatomical left is higher.

      Step 1: find which keypoint is on which anatomical side.
        front: ls_kp is anatomical LEFT (but right side of image, larger x)
               rs_kp is anatomical RIGHT (left side of image, smaller x)
        back:  ls_kp is anatomical LEFT (left side of image, smaller x)
               rs_kp is anatomical RIGHT (right side of image, larger x)

      Step 2: compute the angle of the vector from anatomical-RIGHT to
              anatomical-LEFT (always left−right in anatomy).
        front: anatomical right = rs_kp (smaller x), anatomical left = ls_kp (larger x)
               vector: ls_kp - rs_kp → dx positive (pointing right in image, i.e., toward
               anatomical left since image is mirrored) → atan2 gives small positive angle
               when ls is higher.  ✓
        back:  anatomical right = rs_kp (larger x), anatomical left = ls_kp (smaller x)
               vector: ls_kp - rs_kp → dx negative (pointing left in image = toward
               anatomical left, consistent) → atan2 gives ~180° when shoulders level.  ✗

      Conclusion: for BACK view, we must flip the x-axis of the vector so that
      "toward anatomical left" always yields a positive x component.
    """
    if view == "front":
        # Mirror image: anatomical LEFT appears at larger x.
        # Vector from anatomical-right (rs) to anatomical-left (ls): dx = ls_x - rs_x > 0
        return angle_with_horizontal(rs, ls)
    else:  # back
        # Non-mirrored: anatomical LEFT appears at smaller x.
        # Vector from anatomical-right (rs) to anatomical-left (ls): dx = ls_x - rs_x < 0
        # → would give ~180°.  Fix: negate dx so the vector points in the
        #   "conceptually correct" direction, preserving dy sign.
        dx = -(ls[0] - rs[0])   # flip x to make "toward anatomical left" positive
        dy = -(ls[1] - rs[1])   # same dy sign convention as angle_with_horizontal
        return math.degrees(math.atan2(dy, dx))


def clavicle_tilt(lc, rc, view):
    """Same direction fix applied to clavicle keypoints."""
    if view == "front":
        return angle_with_horizontal(rc, lc)
    else:
        dx = -(lc[0] - rc[0])
        dy = -(lc[1] - rc[1])
        return math.degrees(math.atan2(dy, dx))


def relative_height(ref_pt, target_pt):
    """Signed pixel difference: positive when target is ABOVE ref (smaller Y)."""
    return ref_pt[1] - target_pt[1]


def moving_average(data, window):
    result = []
    buf = deque(maxlen=window)
    for v in data:
        buf.append(v)
        result.append(sum(buf) / len(buf))
    return result


# ── Overlay drawing ──────────────────────────────────────────────────────────

COLORS = {
    "left":    (0, 220, 100),
    "right":   (0, 120, 255),
    "axis":    (220, 220, 50),
    "text_bg": (20, 20, 20),
    "text_fg": (255, 255, 255),
    "warn":    (0, 60, 240),
}


def draw_keypoint(frame, pt, color, radius=7):
    cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
    cv2.circle(frame, (int(pt[0]), int(pt[1])), radius + 2, (255, 255, 255), 1)


def draw_line(frame, p1, p2, color, thickness=3):
    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
             color, thickness, cv2.LINE_AA)


def draw_angle_arc(frame, center, p1, p2, color, radius=40):
    a1 = math.degrees(math.atan2(-(p1[1] - center[1]), p1[0] - center[0]))
    a2 = math.degrees(math.atan2(-(p2[1] - center[1]), p2[0] - center[0]))
    start = min(a1, a2)
    end   = max(a1, a2)
    if end - start > 180:
        start, end = end, start + 360
    cv2.ellipse(frame,
                (int(center[0]), int(center[1])),
                (radius, radius),
                0, -end, -start,
                color, 2, cv2.LINE_AA)


def put_text_box(frame, text, origin, font_scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 4, y - h - 4), (x + w + 4, y + baseline + 2),
                  COLORS["text_bg"], -1)
    cv2.putText(frame, text, (x, y), font, font_scale,
                COLORS["text_fg"], thickness, cv2.LINE_AA)


def draw_horizontal_ref(frame, y, color=(180, 180, 180)):
    h, w = frame.shape[:2]
    dash = 15
    for x in range(0, w, dash * 2):
        cv2.line(frame, (x, int(y)), (min(x + dash, w), int(y)),
                 color, 1, cv2.LINE_AA)


def annotate_frame(frame, keypoints, scores, metrics, view):
    """Overlay keypoints, lines, angles and metrics on frame in place."""

    def kp(name):
        idx = KP[name]
        if scores[idx] < SCORE_THRESHOLD:
            return None
        return keypoints[idx, :2]

    ls = kp("left_shoulder")
    rs = kp("right_shoulder")
    lc = kp("left_clavicle")
    rc = kp("right_clavicle")
    neck = kp("neck")

    if ls is not None and rs is not None:
        mid_y = (ls[1] + rs[1]) / 2
        draw_horizontal_ref(frame, mid_y)

    if ls is not None and rs is not None:
        draw_line(frame, rs, ls, COLORS["axis"], 3)
        draw_keypoint(frame, ls, COLORS["left"])
        draw_keypoint(frame, rs, COLORS["right"])
        mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        h_ref = (mid[0] + 60, mid[1])
        draw_angle_arc(frame, mid, ls, h_ref, (220, 220, 50), radius=35)

    if lc is not None and rc is not None:
        draw_line(frame, rc, lc, (200, 80, 200), 2)
        draw_keypoint(frame, lc, (180, 60, 180), radius=5)
        draw_keypoint(frame, rc, (180, 60, 180), radius=5)

    if neck is not None:
        draw_keypoint(frame, neck, (50, 200, 220), radius=6)

    h, w = frame.shape[:2]
    panel_x = 14
    y_start = 30
    line_h  = 28

    hud = [
        ("Shoulder tilt",   metrics.get("shoulder_tilt_deg"),  "deg"),
        ("Clavicle tilt",   metrics.get("clavicle_tilt_deg"),  "deg"),
        ("L shoulder ht.",  metrics.get("left_shoulder_height"), "px"),
        ("R shoulder ht.",  metrics.get("right_shoulder_height"),"px"),
        ("Imbalance (L-R)", metrics.get("shoulder_imbalance"),  "px"),
    ]

    # Show camera view mode in the HUD
    put_text_box(frame, f"View: {view.upper()}", (panel_x, y_start - line_h),
                 font_scale=0.45)

    for i, (label, val, unit) in enumerate(hud):
        y = y_start + i * line_h
        if val is None:
            txt = f"{label}: N/A"
        else:
            sign = "+" if val > 0 else ""
            txt = f"{label}: {sign}{val:.1f} {unit}"
        put_text_box(frame, txt, (panel_x, y))

    frame_num = metrics.get("frame", 0)
    put_text_box(frame, f"Frame {frame_num}", (panel_x, h - 20), font_scale=0.45)

    return frame


# ── Metrics extraction ────────────────────────────────────────────────────────

def extract_metrics(keypoints, scores, frame_idx, view):
    """Return dict of all measurements for this frame."""
    m = {"frame": frame_idx}

    def kp(name):
        idx = KP[name]
        if scores[idx] < SCORE_THRESHOLD:
            return None
        return keypoints[idx, :2]

    ls = kp("left_shoulder")
    rs = kp("right_shoulder")
    lc = kp("left_clavicle")
    rc = kp("right_clavicle")
    neck = kp("neck")
    lh = kp("left_hip")
    rh = kp("right_hip")
    hip = kp("hip")

    # ── Shoulder tilt ──────────────────────────────────────────────────────
    # Uses view-aware formula so the result is always:
    #   positive → anatomical LEFT shoulder higher than RIGHT
    #   negative → anatomical RIGHT shoulder higher than LEFT
    # regardless of whether the video is front-facing (mirrored) or back-facing.
    if ls is not None and rs is not None:
        m["shoulder_tilt_deg"] = shoulder_tilt(ls, rs, view)
        m["shoulder_width_px"] = abs(ls[0] - rs[0])

        neck_ref = neck if neck is not None else (
            ((ls[0] + rs[0]) / 2, min(ls[1], rs[1]) - 20))

        m["left_shoulder_height"]  = relative_height(neck_ref, ls)
        m["right_shoulder_height"] = relative_height(neck_ref, rs)
        m["shoulder_imbalance"] = (m["left_shoulder_height"]
                                   - m["right_shoulder_height"])
    else:
        m["shoulder_tilt_deg"]    = None
        m["shoulder_width_px"]    = None
        m["left_shoulder_height"] = None
        m["right_shoulder_height"]= None
        m["shoulder_imbalance"]   = None

    # ── Clavicle tilt ───────────────────────────────────────────────────────
    if lc is not None and rc is not None:
        m["clavicle_tilt_deg"] = clavicle_tilt(lc, rc, view)
    else:
        m["clavicle_tilt_deg"] = None

    # ── Lateral shift (% of shoulder width) ────────────────────────────────
    if ls is not None and rs is not None:
        mid_sh_x = (ls[0] + rs[0]) / 2
        sw = m.get("shoulder_width_px") or 0

        if hip is not None:
            mid_hip_x = hip[0]
        elif lh is not None and rh is not None:
            mid_hip_x = (lh[0] + rh[0]) / 2
        else:
            mid_hip_x = None

        if mid_hip_x is not None and sw > 0:
            m["lateral_shift_pct"] = abs(mid_sh_x - mid_hip_x) / sw * 100
        else:
            m["lateral_shift_pct"] = None
    else:
        m["lateral_shift_pct"] = None

    # ── Trunk inclination ────────────────────────────────────────────────────
    if neck is not None:
        if hip is not None:
            m["trunk_tilt_deg"] = angle_with_horizontal(hip, neck)
        elif lh is not None and rh is not None:
            mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            m["trunk_tilt_deg"] = angle_with_horizontal(mid_hip, neck)
        else:
            m["trunk_tilt_deg"] = None
    else:
        m["trunk_tilt_deg"] = None

    return m


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_plots(rows, fps, output_path):
    frames = [r["frame"] for r in rows]
    times  = [f / fps for f in frames]

    def safe(key):
        return [r.get(key) for r in rows]

    sh_tilt   = safe("shoulder_tilt_deg")
    clav_tilt = safe("clavicle_tilt_deg")
    lh        = safe("left_shoulder_height")
    rh        = safe("right_shoulder_height")
    imbal     = safe("shoulder_imbalance")
    trunk     = safe("trunk_tilt_deg")

    def sm(series, w=SMOOTH_WINDOW):
        valid = [v if v is not None else float("nan") for v in series]
        buf, out = deque(maxlen=w), []
        for v in valid:
            if not math.isnan(v):
                buf.append(v)
            if buf:
                out.append(sum(buf) / len(buf))
            else:
                out.append(float("nan"))
        return out

    fig = plt.figure(figsize=(16, 14), facecolor="#0e1117")
    fig.suptitle("Shoulder Alignment Analysis", fontsize=20,
                 color="white", fontweight="bold", y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_cfg = dict(facecolor="#1a1d27")
    line_cfg = dict(linewidth=2, alpha=0.9)

    def style_ax(ax, title, ylabel, xlabel="Time (s)"):
        ax.set_title(title, color="white", fontsize=12, pad=6)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333344")
        ax.grid(True, color="#2a2d3a", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="#555566", linewidth=0.9, linestyle=":")

    ax1 = fig.add_subplot(gs[0, :], **ax_cfg)
    ax1.plot(times, sm(sh_tilt),   color="#4fc3f7", label="Shoulder tilt", **line_cfg)
    ax1.plot(times, sm(clav_tilt), color="#ce93d8", label="Clavicle tilt",
             linestyle="--", **line_cfg)
    ax1.fill_between(times, sm(sh_tilt), 0,
                     where=[v > 0 if v == v else False for v in sm(sh_tilt)],
                     alpha=0.15, color="#4fc3f7")
    ax1.fill_between(times, sm(sh_tilt), 0,
                     where=[v < 0 if v == v else False for v in sm(sh_tilt)],
                     alpha=0.15, color="#f48fb1")
    ax1.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=9)
    style_ax(ax1, "Shoulder & Clavicle Tilt  (+ = Anatomical Left side higher)", "Angle (°)")
    ax1.axhspan(-2.5, 2.5, alpha=0.07, color="green", label="±2.5° normal range")

    ax2 = fig.add_subplot(gs[1, 0], **ax_cfg)
    ax2.plot(times, sm(lh), color="#69f0ae", label="Left shoulder", **line_cfg)
    ax2.plot(times, sm(rh), color="#ff8a65", label="Right shoulder", **line_cfg)
    ax2.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=9)
    style_ax(ax2, "Shoulder Heights (from Neck)", "Height (px, + = higher)")

    ax3 = fig.add_subplot(gs[1, 1], **ax_cfg)
    im_sm = sm(imbal)
    colors_bar = ["#4fc3f7" if (v > 0 if v == v else False) else "#f48fb1" for v in im_sm]
    ax3.bar(times, im_sm, width=1 / fps, color=colors_bar, alpha=0.8)
    style_ax(ax3, "Shoulder Imbalance  L − R  (+ = Anatomical Left higher)", "Pixels")

    ax4 = fig.add_subplot(gs[2, 0], **ax_cfg)
    ax4.plot(times, sm(trunk), color="#ffcc02", **line_cfg)
    ax4.fill_between(times, sm(trunk), 85,
                     where=[v is not None and v == v for v in sm(trunk)],
                     alpha=0.1, color="#ffcc02")
    style_ax(ax4, "Trunk Inclination (Neck→Hip vs horizontal)", "Angle (°)")

    window = max(1, int(fps))
    raw_vals = [v if v is not None and v == v else float("nan") for v in sh_tilt]
    std_vals = []
    buf = deque(maxlen=window)
    for v in raw_vals:
        if not math.isnan(v):
            buf.append(v)
        if len(buf) > 1:
            std_vals.append(float(np.std(buf)))
        else:
            std_vals.append(float("nan"))

    ax5 = fig.add_subplot(gs[2, 1], **ax_cfg)
    ax5.plot(times, std_vals, color="#ef9a9a", **line_cfg)
    ax5.fill_between(times, std_vals, 0,
                     where=[not math.isnan(v) for v in std_vals],
                     alpha=0.2, color="#ef9a9a")
    style_ax(ax5, "Shoulder Tilt Variability  (1-s rolling std)", "Std Dev (°)")

    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Plot saved → {output_path}")


def generate_symmetry_bar_chart(rows, fps, output_path):
    import math, numpy as np

    SMOOTH_W = 7

    def smooth(series):
        from collections import deque
        buf, out = deque(maxlen=SMOOTH_W), []
        for v in series:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                buf.append(v)
            out.append(sum(buf) / len(buf) if buf else float("nan"))
        return out

    def nanmean(series):
        vals = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    tilt_series = smooth([r.get("shoulder_tilt_deg") for r in rows])
    mean_tilt   = abs(nanmean(tilt_series))

    ht_diff_pct_series = []
    for r in rows:
        lh = r.get("left_shoulder_height")
        rh = r.get("right_shoulder_height")
        sw = r.get("shoulder_width_px")
        if lh is None or rh is None:
            ht_diff_pct_series.append(float("nan"))
            continue
        diff_px = abs(lh - rh)
        if sw and sw > 0:
            ht_diff_pct_series.append(diff_px / sw * 100)
        else:
            scale = (abs(lh) + abs(rh)) / 2
            ht_diff_pct_series.append(diff_px / scale * 100 if scale > 0 else float("nan"))
    ht_diff_pct_series = smooth(ht_diff_pct_series)
    mean_ht_diff = nanmean(ht_diff_pct_series)

    lat_shift_pct_series = smooth([r.get("lateral_shift_pct") for r in rows])
    mean_lat_shift = nanmean(lat_shift_pct_series)
    if math.isnan(mean_lat_shift):
        mean_lat_shift = 0.0

    # ── Clinical thresholds (literature-based) ────────────────────────────
    # Shoulder Tilt  : normal ≤ 2.5°  (Kendall et al., 2005;
    #                                   Fortin et al., Spine 2011)
    # Height Diff    : normal ≤ 5% SW (Asher & Burton, Spine 2006;
    #                                   Scoliosis Research Society)
    # Lateral Shift  : normal ≤ 7.5% SW (Lafon et al., Eur Spine J 2009;
    #                                     Stokes & Moreland, Spine 1987)
    # Mild thresholds are set at ~1.5–2× the normal boundary, representing
    # the range between "acceptable variation" and "warrants clinical review".
    thresholds = {
        "Shoulder\nTilt (°)":    {"normal": 2.5, "mild": 5,  "val": mean_tilt},
        "Height Diff\n(% SW)":   {"normal": 5.0, "mild": 10, "val": mean_ht_diff},
        "Lateral\nShift (% SW)": {"normal": 7.5, "mild": 12, "val": mean_lat_shift},
    }

    labels  = list(thresholds.keys())
    values  = [thresholds[k]["val"]    for k in labels]
    normals = [thresholds[k]["normal"] for k in labels]
    milds   = [thresholds[k]["mild"]   for k in labels]

    bar_colors = []
    for v, n, m in zip(values, normals, milds):
        if math.isnan(v):
            bar_colors.append("#555555")
        elif v <= n:
            bar_colors.append("#2ecc71")
        elif v <= m:
            bar_colors.append("#e67e22")
        else:
            bar_colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    x = np.arange(len(labels))
    bar_width = 0.42

    bars = ax.bar(x, values, width=bar_width, color=bar_colors,
                  zorder=3, alpha=0.92, edgecolor="none")

    half = bar_width / 2 + 0.06
    for i, (n, m) in enumerate(zip(normals, milds)):
        ax.hlines(n, i - half, i + half, colors="#2ecc71", linewidth=2.5, zorder=4)
        ax.hlines(m, i - half, i + half, colors="#f39c12", linewidth=2.5,
                  linestyle="--", zorder=4)

    for bar, val in zip(bars, values):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    color="white", fontsize=13, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Normal threshold"),
        Line2D([0], [0], color="#f39c12", linewidth=2.5, linestyle="--", label="Mild threshold"),
    ]
    ax.legend(handles=legend_elements, facecolor="#1a1f2e", edgecolor="#333",
              labelcolor="white", fontsize=11, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=12)
    ax.set_ylabel("Value", color="white", fontsize=12)
    ax.tick_params(axis="y", colors="#aaaaaa")
    ax.tick_params(axis="x", length=0)
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", color="#1e2535", linewidth=0.8, zorder=0)
    ax.set_title("Shoulder Symmetry — Measured vs Reference Thresholds",
                 color="white", fontsize=15, fontweight="bold", pad=16)

    max_val = max([v for v in values if not math.isnan(v)] + milds + [1])
    ax.set_ylim(0, max_val * 1.20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Symmetry bar chart saved → {output_path}")


def write_summary(rows, fps, path, view):
    keys = ["shoulder_tilt_deg", "clavicle_tilt_deg",
            "shoulder_imbalance", "trunk_tilt_deg"]

    lines = [
        "=" * 56,
        "  SHOULDER ALIGNMENT ANALYSIS — SUMMARY",
        "=" * 56,
        f"  Total frames analysed : {len(rows)}",
        f"  Video FPS             : {fps:.2f}",
        f"  Duration              : {len(rows)/fps:.2f} s",
        f"  Camera view           : {view.upper()}",
        "",
    ]

    for k in keys:
        vals = [r[k] for r in rows
                if r.get(k) is not None and r[k] == r[k]]
        if not vals:
            lines.append(f"  {k:30s}: no data")
            continue
        arr = np.array(vals)
        lines += [
            f"  ── {k} ──",
            f"     Mean   : {arr.mean():+.2f}°",
            f"     Std    : {arr.std():.2f}",
            f"     Min    : {arr.min():+.2f}",
            f"     Max    : {arr.max():+.2f}",
            f"     |Mean| : {np.abs(arr).mean():.2f}  (severity proxy)",
            "",
        ]

    sh_vals = [r["shoulder_tilt_deg"] for r in rows
               if r.get("shoulder_tilt_deg") is not None]
    if sh_vals:
        mean_tilt = np.mean(sh_vals)
        if abs(mean_tilt) < 2.5:
            interp = "Within normal range (±2.5°). Good alignment."
        elif abs(mean_tilt) < 5:
            side = "LEFT" if mean_tilt > 0 else "RIGHT"
            interp = f"Mild tilt: anatomical {side} shoulder is higher on average."
        else:
            side = "LEFT" if mean_tilt > 0 else "RIGHT"
            interp = (f"Notable asymmetry: anatomical {side} shoulder is significantly "
                      f"higher (mean {mean_tilt:+.1f}°). Consider clinical review.")
        lines += ["  INTERPRETATION:", f"  {interp}", ""]

    lines.append("=" * 56)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[✓] Summary saved → {path}")
    print("\n".join(lines))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def create_browser_friendly_writer(output_path, fps, frame_size):
    codec_candidates = ["avc1", "H264", "mp4v"]
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if writer.isOpened():
            print(f"[â€¢] Video codec selected: {codec}")
            return writer
        writer.release()
    raise RuntimeError("Could not open a compatible video writer for MP4 export.")


def process_video(input_path, output_dir, device="cpu",
                  model_size="medium", skip_frames=1, view="front"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[•] Camera view mode  : {view.upper()}")
    print(f"[•] Loading SpinePoseEstimator (device={device}, mode={model_size}) …")
    estimator = _SpinePoseEstimator(device=device, mode=model_size)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[•] Video: {width}×{height} @ {fps:.1f} fps, ~{total} frames")

    out_video_path = os.path.join(output_dir, "annotated_video.mp4")
    writer = create_browser_friendly_writer(out_video_path, fps, (width, height))

    csv_path = os.path.join(output_dir, "shoulder_angles.csv")
    csv_fields = ["frame", "time_s",
                  "shoulder_tilt_deg", "clavicle_tilt_deg",
                  "left_shoulder_height", "right_shoulder_height",
                  "shoulder_imbalance", "shoulder_width_px",
                  "lateral_shift_pct", "trunk_tilt_deg"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    all_metrics = []
    frame_idx   = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames != 0:
            frame_idx += 1
            writer.write(frame)
            continue

        try:
            keypoints, scores = estimator(frame)
        except Exception as e:
            print(f"[!] Frame {frame_idx}: inference error ({e})")
            writer.write(frame)
            frame_idx += 1
            continue

        kp_single, sc_single = select_primary_detection(keypoints, scores)
        if kp_single is None or sc_single is None:
            annotated = frame.copy()
            cv2.putText(
                annotated,
                "No shoulder landmarks detected",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            frame_idx += 1
            continue

        # Pass view mode into metrics extraction
        m = extract_metrics(kp_single, sc_single, frame_idx, view)
        m["time_s"] = frame_idx / fps
        all_metrics.append(m)
        csv_writer.writerow(m)

        # Pass view mode into annotation so it shows in HUD
        annotated = annotate_frame(frame.copy(), kp_single, sc_single, m, view)
        writer.write(annotated)

        if frame_idx % 30 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx}/{total}  ({elapsed:.1f}s elapsed)")

        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    print(f"[✓] Annotated video → {out_video_path}")
    print(f"[✓] CSV data        → {csv_path}")

    plot_path    = os.path.join(output_dir, "angle_plot.png")
    summary_path = os.path.join(output_dir, "summary.txt")

    generate_plots(all_metrics, fps, plot_path)

    symmetry_chart_path = os.path.join(output_dir, "symmetry_bar_chart.png")
    generate_symmetry_bar_chart(all_metrics, fps, symmetry_chart_path)

    write_summary(all_metrics, fps, summary_path, view)

    return all_metrics


# ── File picker ────────────────────────────────────────────────────────────────

def pick_video_file():
    VIDEO_TYPES = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.mpeg *.mpg"),
        ("All files",   "*.*"),
    ]

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        path = filedialog.askopenfilename(
            title     = "Select a video file for shoulder analysis",
            filetypes = VIDEO_TYPES,
        )
        root.destroy()
        if path:
            return path
        print("[!] No file selected — exiting.")
        sys.exit(0)
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["zenity", "--file-selection",
             "--title=Select video for shoulder analysis",
             "--file-filter=Video files (mp4 avi mov mkv) | *.mp4 *.avi *.mov *.mkv",
             "--file-filter=All files | *"],
            capture_output=True, text=True, timeout=120,
        )
        path = result.stdout.strip()
        if path:
            return path
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["kdialog", "--getopenfilename", ".", "*.mp4 *.avi *.mov *.mkv"],
            capture_output=True, text=True, timeout=120,
        )
        path = result.stdout.strip()
        if path:
            return path
    except Exception:
        pass

    print("\n[!] Could not open a GUI file picker.")
    print("    Please type the full path to your video file:")
    path = input("  > ").strip().strip('"').strip("'")
    if path and os.path.isfile(path):
        return path

    print("[!] File not found — exiting.")
    sys.exit(1)


def ask_view_mode():
    """
    Interactively ask the user whether the video is front- or back-facing.
    Returns 'front' or 'back'.
    """
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│          Camera View Mode — please choose           │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  [1]  FRONT  — person faces the camera              │")
    print("│               (most phone selfie / webcam videos)   │")
    print("│  [2]  BACK   — person's back faces the camera       │")
    print("│               (filmed from behind)                  │")
    print("└─────────────────────────────────────────────────────┘")
    while True:
        choice = input("  Enter 1 or 2 (or type 'front' / 'back'): ").strip().lower()
        if choice in ("1", "front", "f"):
            return "front"
        if choice in ("2", "back", "b"):
            return "back"
        print("  [!] Invalid choice. Please enter 1 or 2.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shoulder Alignment Analyzer using SpinePose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_video", nargs="?", default=None,
                        help="Path to input video file  "
                             "(omit to open a file-picker dialog)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory for output files  "
                             "(default: same folder as the video, sub-folder '<name>_shoulder_output')")
    parser.add_argument("-d", "--device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="Inference device (default: cpu)")
    parser.add_argument("-m", "--model-size", default="medium",
                        choices=["small", "medium", "large", "xlarge"],
                        help="SpinePose model size (default: medium)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every N-th frame (default: 1 = all frames)")

    # ── NEW: --view argument ────────────────────────────────────────────────
    parser.add_argument(
        "--view",
        default=None,
        choices=["front", "back"],
        help=(
            "Camera view direction:\n"
            "  front  — person faces the camera (mirrored image, e.g. selfie/webcam)\n"
            "  back   — person's back faces the camera (non-mirrored)\n"
            "If omitted, you will be asked interactively."
        ),
    )

    args = parser.parse_args()

    # ── Resolve input video ───────────────────────────────────────────────
    if args.input_video:
        video_path = args.input_video
        if not os.path.isfile(video_path):
            sys.exit(f"[!] File not found: {video_path}")
    else:
        print("[•] No video path supplied — opening file picker …")
        video_path = pick_video_file()

    print(f"[•] Video selected: {video_path}")

    # ── Resolve camera view mode ─────────────────────────────────────────
    if args.view:
        view = args.view
        print(f"[•] Camera view (from argument): {view.upper()}")
    else:
        # Ask interactively if --view was not supplied
        view = ask_view_mode()
        print(f"[•] Camera view selected: {view.upper()}")

    # ── Resolve output directory ─────────────────────────────────────────
    if args.output_dir:
        output_dir = args.output_dir
    else:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, f"{video_stem}_shoulder_output")

    print(f"[•] Output directory: {output_dir}")

    process_video(
        input_path  = video_path,
        output_dir  = output_dir,
        device      = args.device,
        model_size  = args.model_size,
        skip_frames = args.skip,
        view        = view,
    )


if __name__ == "__main__":
    main()
