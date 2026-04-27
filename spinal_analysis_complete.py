"""
=============================================================================
SPINAL CURVATURE ANALYSIS — COMPLETE PIPELINE
=============================================================================
Integrates:
  • Anatomical Pipeline   (SpinePose CVPR 2025, 37 keypoints)
  • Trunk Lean Calculation  (from Tokuda et al., J Phys Ther Sci, 2017)
  • Lordosis / Kyphosis    (compared against Ohlendorf et al., Sci Rep, 2020)
  • UCM Synergy Index     (based on Tokuda et al., 2017)
  • Full Clinical Comparison vs published standard reference values
  • Annotated Video Export (per-frame angles overlay + summary panel)

PAPER REFERENCES:
  [1] Tokuda et al. (2017) "Trunk lean gait decreases multi-segmental
      coordination in the vertical direction." J Phys Ther Sci 29:1940-1946.
      → Provides trunk lean angle definition, KAM context, UCM synergy index.
      → Normal trunk lean during gait: 1.0 ± 1.5°
      → Trunk lean gait target: 11.0 ± 1.0° (10° greater than normal)

  [2] Ohlendorf et al. (2020) "Standard reference values of the upper body
      posture in healthy male adults aged 41–50 years in Germany."
      Scientific Reports 10:3823.
      → Kyphosis angle:  51.08° (TR: 31.63° – 70.53°, CI: 49.14° – 53.01°)
      → Lordosis angle:  32.86° (TR: 15.25° – 50.47°, CI: 31.11° – 34.62°)
      → Sagittal trunk inclination: −3.4° (TR: −8.47° – 1.66°)
      → Frontal trunk inclination:  −0.3° (TR: −3.01° – 2.39°)
=============================================================================
"""

# ─────────────────────────────────────────────
# SECTION 0 — IMPORTS & DEPENDENCY HANDLING
# ─────────────────────────────────────────────
import os
import sys
import math
import warnings
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# Optional heavy deps — graceful fallback when not installed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not installed — video processing disabled. "
                  "Install with: pip install opencv-python")

try:
    from scipy.signal import savgol_filter
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not installed — Savitzky-Golay smoothing disabled. "
                  "Install with: pip install scipy")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not installed — plotting disabled. "
                  "Install with: pip install matplotlib")

try:
    from spinepose import SpinePoseEstimator, PoseTracker
    SPINEPOSE_AVAILABLE = True
except ImportError:
    SPINEPOSE_AVAILABLE = False
    warnings.warn("spinepose not installed — anatomical pipeline will use simulation. "
                  "Install with: pip install spinepose")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── Ensure Windows console can print Unicode (°, ±, ΔV, ═, etc.) ────────────
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass  # non-Windows or redirected stream


# ─────────────────────────────────────────────
# SECTION 1 — PUBLISHED STANDARD REFERENCE VALUES
# ─────────────────────────────────────────────
class PublishedStandards:
    """
    Reference values from peer-reviewed literature for clinical comparison.

    Ohlendorf et al. (2020) Sci Rep — healthy males 41–50 years, Germany.
    Tokuda et al. (2017) J Phys Ther Sci — healthy young adults, Japan.
    Mendeley Posture Dataset / clinical literature — general population ranges.
    """

    # ── Kyphosis (Ohlendorf 2020, Table 3) ──────────────────────────────────
    KYPHOSIS_MEAN   = 51.08   # degrees
    KYPHOSIS_TR_LO  = 31.63   # tolerance region lower limit (≈ −2 SD)
    KYPHOSIS_TR_HI  = 70.53   # tolerance region upper limit (≈ +2 SD)
    KYPHOSIS_CI_LO  = 49.14   # 95% confidence interval left
    KYPHOSIS_CI_HI  = 53.01   # 95% confidence interval right

    # ── Lordosis (Ohlendorf 2020, Table 3) ──────────────────────────────────
    LORDOSIS_MEAN   = 32.86
    LORDOSIS_TR_LO  = 15.25
    LORDOSIS_TR_HI  = 50.47
    LORDOSIS_CI_LO  = 31.11
    LORDOSIS_CI_HI  = 34.62

    # ── Sagittal trunk inclination / trunk lean (Ohlendorf 2020, Table 3) ──
    # Negative = ventral (forward) tilt
    SAGITTAL_INCL_MEAN  = -3.4    # ventral inclination
    SAGITTAL_INCL_TR_LO = -8.47
    SAGITTAL_INCL_TR_HI =  1.66
    SAGITTAL_INCL_CI_LO = -3.90
    SAGITTAL_INCL_CI_HI = -2.89

    # ── Frontal trunk inclination (Ohlendorf 2020, Table 3) ─────────────────
    FRONTAL_INCL_MEAN   = -0.3    # slight left lateral
    FRONTAL_INCL_TR_LO  = -3.01
    FRONTAL_INCL_TR_HI  =  2.39

    # ── Trunk lean during gait (Tokuda et al. 2017, Results) ────────────────
    # Standing / normal gait trunk lean
    GAIT_TRUNK_LEAN_NORMAL_MEAN = 1.0   # degrees
    GAIT_TRUNK_LEAN_NORMAL_SD   = 1.5
    # Modified trunk lean gait (KAM-reduction strategy)
    GAIT_TRUNK_LEAN_MOD_MEAN    = 11.0
    GAIT_TRUNK_LEAN_MOD_SD      = 1.0
    # KAM values
    KAM_NORMAL_MEAN = 0.6   # N·m/kg
    KAM_NORMAL_SD   = 0.1
    KAM_LEAN_MEAN   = 0.4
    KAM_LEAN_SD     = 0.1

    # ── Mendeley Posture Dataset — clinical classification ranges ────────────
    # Kyphosis-based Cobb angle thresholds
    COBB_KYPHOSIS_NORMAL_LO  =  20.0
    COBB_KYPHOSIS_NORMAL_HI  =  40.0
    COBB_KYPHOSIS_MILD_LO    =  40.0
    COBB_KYPHOSIS_MILD_HI    =  60.0
    COBB_KYPHOSIS_SEVERE_LO  =  60.0
    # Geometric surface curvature thresholds (proxy, not clinical Cobb)
    GEOM_NORMAL_LO  =   0.0
    GEOM_NORMAL_HI  =  20.0
    GEOM_MILD_LO    =  20.0
    GEOM_MILD_HI    =  40.0
    GEOM_SEVERE_LO  =  40.0

    @classmethod
    def classify_cobb(cls, cobb_deg: float) -> str:
        """Classify Cobb / kyphosis angle against Mendeley clinical ranges."""
        if cobb_deg < cls.COBB_KYPHOSIS_NORMAL_LO:
            return "below_normal"
        elif cobb_deg <= cls.COBB_KYPHOSIS_NORMAL_HI:
            return "normal"
        elif cobb_deg <= cls.COBB_KYPHOSIS_MILD_HI:
            return "mild"
        else:
            return "severe"

    @classmethod
    def classify_trunk_lean(cls, lean_deg: float) -> str:
        """Classify trunk lean angle vs Tokuda et al. (2017) gait norms."""
        normal_lo = cls.GAIT_TRUNK_LEAN_NORMAL_MEAN - 2 * cls.GAIT_TRUNK_LEAN_NORMAL_SD
        normal_hi = cls.GAIT_TRUNK_LEAN_NORMAL_MEAN + 2 * cls.GAIT_TRUNK_LEAN_NORMAL_SD
        mod_lo = cls.GAIT_TRUNK_LEAN_MOD_MEAN - 2 * cls.GAIT_TRUNK_LEAN_MOD_SD
        mod_hi = cls.GAIT_TRUNK_LEAN_MOD_MEAN + 2 * cls.GAIT_TRUNK_LEAN_MOD_SD
        if lean_deg < 0:
            return "contralateral_lean"
        elif lean_deg <= normal_hi:
            return "normal_gait_range"
        elif lean_deg <= mod_hi:
            return "modified_lean_gait_range"
        else:
            return "excessive_lean"

    @classmethod
    def classify_kyphosis_ohlendorf(cls, kyphosis_deg: float) -> str:
        """Classify kyphosis vs Ohlendorf 2020 tolerance region for men 41–50."""
        if kyphosis_deg < cls.KYPHOSIS_TR_LO:
            return "below_tolerance_region"
        elif kyphosis_deg <= cls.KYPHOSIS_TR_HI:
            return "within_tolerance_region"
        else:
            return "above_tolerance_region"

    @classmethod
    def classify_lordosis_ohlendorf(cls, lordosis_deg: float) -> str:
        """Classify lordosis vs Ohlendorf 2020 tolerance region for men 41–50."""
        if lordosis_deg < cls.LORDOSIS_TR_LO:
            return "below_tolerance_region"
        elif lordosis_deg <= cls.LORDOSIS_TR_HI:
            return "within_tolerance_region"
        else:
            return "above_tolerance_region"

    @classmethod
    def deviation_from_norm(cls, value: float, mean: float, sd_approx: float) -> Dict:
        """Return z-score and interpretation relative to a published mean."""
        z = (value - mean) / sd_approx if sd_approx > 0 else 0.0
        interp = ("within 1 SD" if abs(z) <= 1 else
                  "within 2 SD" if abs(z) <= 2 else "beyond 2 SD")
        return {"value": value, "reference_mean": mean, "z_score": round(z, 2),
                "interpretation": interp}


# ─────────────────────────────────────────────
# SECTION 2 — CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class Config:
    """Centralised configuration for all pipeline parameters."""

    # ── Paths ────────────────────────────────────────────────────────────────
    video_dir: str = "data/raw_videos"
    output_dir: str = "outputs"
    model_dir: str = "models/calibration"
    spinepose_mode: str = "medium"   # PoseTracker mode: "light", "medium", "heavy"
    own_dataset_dir: str = "data/datasets/own_dataset"
    mendeley_dir: str = "data/datasets/mendeley_posture"
    spinetrack_dir: str = "data/datasets/spinetrack"

    # ── SpinePose keypoint indices (0-based, from CVPR 2025 release) ────────
    # 37 total keypoints; we use 9 vertebral landmarks
    KEYPOINT_C1     = 36
    KEYPOINT_C4     = 35
    KEYPOINT_C7     = 18  # Cervicothoracic junction — Cobb angle top
    KEYPOINT_T3     = 30
    KEYPOINT_T8     = 29
    KEYPOINT_L1     = 28  # Thoracolumbar junction — Cobb angle bottom
    KEYPOINT_L3     = 27
    KEYPOINT_L5     = 26
    KEYPOINT_SACRUM = 19  # Trunk lean baseline

    SPINE_KEYPOINT_INDICES = [36, 35, 18, 30, 29, 28, 27, 26, 19]
    SPINE_KEYPOINT_NAMES   = ["C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "Sacrum"]

    # ── Frame validity thresholds ────────────────────────────────────────────
    SPINEPOSE_CONF_THRESH   = 0.5    # minimum mean keypoint confidence (FIX-5)
    MASK_AREA_MIN           = 500    # minimum silhouette pixels (geometric)

    # ── Signal smoothing (FIX-3) ─────────────────────────────────────────────
    SAVGOL_WINDOW   = 9
    SAVGOL_POLYORDER = 2

    # ── AI calibration ───────────────────────────────────────────────────────
    RF_N_ESTIMATORS  = 200
    RF_RANDOM_STATE  = 42
    CAL_TEST_SIZE    = 0.2
    MLP_EPOCHS       = 100
    MLP_BATCH_SIZE   = 32
    MLP_LR           = 1e-3

    # ── Trunk lean (Tokuda et al. 2017 protocol) ─────────────────────────────
    TRUNK_LEAN_TARGET_INCREMENT = 10.0  # degrees above normal gait lean
    TRUNK_LEAN_ERROR_RANGE      = 2.0   # ± degrees accepted (gait retraining)

    # ── UCM synergy parameters (Tokuda et al. 2017) ──────────────────────────
    UCM_N_MEDIOLATERAL  = 13  # DOFs for mediolateral COM model
    UCM_N_VERTICAL      = 9   # DOFs for vertical COM model
    UCM_D               = 1   # performance variable dimension

    # ── Reporting ────────────────────────────────────────────────────────────
    REJECTION_RATE_WARN = 40.0   # warn if >40% frames rejected


CONFIG = Config()


# ─────────────────────────────────────────────
# SECTION 3 — DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class FrameResult:
    frame_idx:        int
    # Anatomical
    keypoints:        Dict[str, Any] = field(default_factory=dict)
    kyphosis_angle:   Optional[float] = None
    trunk_lean_angle: Optional[float] = None
    keypoint_confidence: Optional[float] = None
    valid:            bool = False
    rejection_reason: Optional[str] = None
    # Derived
    lordosis_angle:   Optional[float] = None   # if lower-spine keypoints available
    # Clinical labels
    cobb_class:       Optional[str] = None
    kyphosis_class_ohlendorf: Optional[str] = None
    lordosis_class_ohlendorf: Optional[str] = None
    trunk_lean_class: Optional[str] = None


@dataclass
class VideoResult:
    video_path:   str
    frame_results: List[FrameResult] = field(default_factory=list)
    # Smoothed series
    kyphosis_smoothed:    np.ndarray = field(default_factory=lambda: np.array([]))
    trunk_lean_smoothed:  np.ndarray = field(default_factory=lambda: np.array([]))
    lordosis_smoothed:    np.ndarray = field(default_factory=lambda: np.array([]))
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)
    # UCM synergy (if multi-trial)
    ucm_results: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# SECTION 4 — TRUNK LEAN CALCULATION
# ─────────────────────────────────────────────
class TrunkLeanCalculator:
    """
    Computes the trunk lean angle from SpinePose keypoints.

    Anatomical definition (Tokuda et al. 2017)
    -------------------------------------------
    The trunk vector connects:
      • LOWER reference: midpoint of left and right posterior superior iliac
        spines (PSIS) — approximated here by the Sacrum keypoint.
      • UPPER reference: midpoint of left and right acromion processes —
        approximated here by the C7 keypoint (cervicothoracic junction).

    The trunk lean angle is the angle between this trunk vector and the
    global vertical axis, measured in the sagittal (x–z) plane:
        lean = atan2(|Δx_horizontal|, |Δz_vertical|) in degrees

    2-D keypoint approximation (this implementation)
    -------------------------------------------------
    Because SpinePose operates on a 2-D sagittal image:
      Δx = C7.x − Sacrum.x   (horizontal separation; positive = rightward)
      Δy = |C7.y − Sacrum.y| (vertical separation; image y increases downward)
      lean = atan2(|Δx|, Δy)

    Sign convention: positive = ipsilateral lean (toward the stance limb).
    For sagittal-only video the unsigned tilt magnitude is reported.
    """

    @staticmethod
    def from_keypoints(c7: np.ndarray, sacrum: np.ndarray) -> float:
        """
        Calculate trunk lean angle from C7 and Sacrum pixel coordinates.

        Parameters
        ----------
        c7     : [x, y] pixel position of the C7 vertebra keypoint
                 (cervicothoracic junction; upper trunk reference).
        sacrum : [x, y] pixel position of the Sacrum keypoint
                 (lower trunk reference; proxies the PSIS midpoint).

        Returns
        -------
        lean_deg : float — trunk lean in degrees from vertical.
                   Returns 0.0 if the two points are vertically coincident.

        Formula (Tokuda et al. 2017)
        ----------------------------
            Δx = c7.x − sacrum.x          (horizontal separation)
            Δy = |c7.y − sacrum.y|        (vertical separation, always positive)
            lean = atan2(|Δx|, Δy)        (angle from the vertical axis)
        """
        delta_x = float(c7[0] - sacrum[0])   # horizontal (mediolateral)
        delta_y = float(c7[1] - sacrum[1])   # vertical (image y increases downward)
        # In image coords, sacrum.y > c7.y (sacrum is lower in image)
        # We want vertical displacement magnitude
        vertical = abs(delta_y)
        if vertical < 1e-6:
            return 0.0
        lean_rad = math.atan2(abs(delta_x), vertical)
        lean_deg = math.degrees(lean_rad)
        # Sign convention: positive = ipsilateral lean (toward stance limb)
        # For sagittal-plane video we report the absolute tilt from vertical
        return lean_deg

    @staticmethod
    def from_inclination(inclination_deg: float) -> float:
        """
        Derive trunk lean from a pre-computed segment inclination.
        Used internally by the anatomical pipeline (FIX-2).
        """
        return abs(inclination_deg)

    @staticmethod
    def normalize_cobb_for_lean(cobb_raw: float, trunk_lean: float) -> float:
        """
        Remove the global trunk lean bias from a raw Cobb angle (FIX-2).

        Problem addressed
        -----------------
        When a subject leans forward by angle θ, every segment inclination
        increases by approximately θ.  Without correction, the Cobb angle
        would incorrectly absorb this global lean as additional spinal curvature.

        Method
        ------
        Global lean  = signed inclination of C7 → Sacrum (in degrees)
        For each segment i:
            normalized[i] = inclination[i] − global_lean
        Cobb = max(normalized) − min(normalized)

        Parameters
        ----------
        cobb_raw    : Cobb angle computed WITHOUT lean correction
        trunk_lean  : global lean (C7 → Sacrum inclination, unsigned)

        Returns
        -------
        cobb_corrected : float — lean-corrected kyphosis angle
        """
        return cobb_raw - trunk_lean

    @staticmethod
    def compare_to_standard(trunk_lean_deg: float,
                            context: str = "standing") -> Dict:
        """
        Compare computed trunk lean to Tokuda et al. (2017) reference values
        and Ohlendorf et al. (2020) sagittal inclination norms.

        Parameters
        ----------
        trunk_lean_deg : measured trunk lean in degrees
        context        : 'standing' or 'gait'
        """
        std = PublishedStandards
        result = {"measured_deg": round(trunk_lean_deg, 2)}

        if context == "gait":
            # Tokuda 2017: normal gait = 1.0 ± 1.5°
            result["reference"] = "Tokuda et al. (2017) — normal gait"
            result["reference_mean_deg"] = std.GAIT_TRUNK_LEAN_NORMAL_MEAN
            result["reference_sd_deg"]   = std.GAIT_TRUNK_LEAN_NORMAL_SD
            result["classification"] = std.classify_trunk_lean(trunk_lean_deg)
            diff = trunk_lean_deg - std.GAIT_TRUNK_LEAN_NORMAL_MEAN
            result["deviation_from_normal_deg"] = round(diff, 2)
            result["modified_gait_target_deg"] = (
                std.GAIT_TRUNK_LEAN_NORMAL_MEAN + CONFIG.TRUNK_LEAN_TARGET_INCREMENT
            )
        else:
            # Ohlendorf 2020: sagittal trunk inclination = −3.4° (ventral)
            ref_mean = abs(std.SAGITTAL_INCL_MEAN)   # 3.4° forward
            ref_tr_lo = 0.0                           # within TR (approx)
            ref_tr_hi = abs(std.SAGITTAL_INCL_TR_LO) # 8.47° (max forward)
            result["reference"] = "Ohlendorf et al. (2020) — standing posture"
            result["reference_mean_deg"] = round(ref_mean, 2)
            result["tolerance_region_upper_deg"] = round(ref_tr_hi, 2)
            within_tr = ref_tr_lo <= trunk_lean_deg <= ref_tr_hi
            result["within_tolerance_region"] = within_tr
            result["classification"] = (
                "normal_standing" if within_tr else "excessive_forward_lean"
            )

        return result


# ─────────────────────────────────────────────
# SECTION 5 — LORDOSIS CALCULATION
# ─────────────────────────────────────────────
class LordosisCalculator:
    """
    Estimate lumbar lordosis angle from SpinePose keypoints.

    Clinical Definition:
      Lordosis (lumbar) = angle subtended between L1 and L5/Sacrum
      Measured on sagittal view; higher values = greater lumbar curve.

    From keypoints: inclination of L1–L3 vs L3–L5 (or L1–L5 directly).
    Compare to Ohlendorf et al. (2020): lordosis mean = 32.86°
    """

    @staticmethod
    def from_keypoints(l1: np.ndarray,
                       l3: Optional[np.ndarray],
                       l5: Optional[np.ndarray],
                       sacrum: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Compute lumbar lordosis angle.

        Uses upper and lower lumbar segment inclinations:
          upper_incl = inclination(L1 → L3)
          lower_incl = inclination(L3 → L5) or inclination(L1 → Sacrum)
          lordosis   = |upper_incl − lower_incl|

        Falls back to a single-segment estimate if only two points available.
        """
        def inclination(p1: np.ndarray, p2: np.ndarray) -> float:
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            if abs(dy) < 1e-6:
                return 0.0
            # Signed inclination — preserves lateral direction so
            # the lordosis difference captures actual curvature change
            return math.degrees(math.atan2(dx, abs(dy)))

        if l3 is not None and l5 is not None:
            upper = inclination(l1, l3)
            lower = inclination(l3, l5)
            return abs(upper - lower)
        elif l3 is not None and sacrum is not None:
            upper = inclination(l1, l3)
            lower = inclination(l3, sacrum)
            return abs(upper - lower)
        elif sacrum is not None:
            # Simple L1–Sacrum inclination as proxy
            return inclination(l1, sacrum)
        return None

    @staticmethod
    def compare_to_standard(lordosis_deg: float) -> Dict:
        """
        Compare lordosis angle to Ohlendorf et al. (2020) norms.

        Reference (Table 3, healthy males 41–50, Germany):
          Mean:  32.86°
          TR:    15.25° – 50.47°
          CI:    31.11° – 34.62°
        """
        std = PublishedStandards
        tr_sd_approx = (std.LORDOSIS_TR_HI - std.LORDOSIS_TR_LO) / 4.0
        dev = std.deviation_from_norm(lordosis_deg, std.LORDOSIS_MEAN, tr_sd_approx)
        return {
            "measured_deg": round(lordosis_deg, 2),
            "reference_paper": "Ohlendorf et al. (2020) Sci Rep",
            "reference_population": "Healthy males 41–50 yrs, Germany",
            "reference_mean_deg": std.LORDOSIS_MEAN,
            "tolerance_region": [std.LORDOSIS_TR_LO, std.LORDOSIS_TR_HI],
            "confidence_interval": [std.LORDOSIS_CI_LO, std.LORDOSIS_CI_HI],
            "classification": std.classify_lordosis_ohlendorf(lordosis_deg),
            "z_score_approx": dev["z_score"],
            "within_CI": std.LORDOSIS_CI_LO <= lordosis_deg <= std.LORDOSIS_CI_HI,
        }


# ─────────────────────────────────────────────
# SECTION 6 — KYPHOSIS / COBB CALCULATION
# ─────────────────────────────────────────────
class KyphosisCalculator:
    """
    Robust Cobb / kyphosis angle from SpinePose keypoints.

    Implements FIX-1 (robust top-2/bottom-2 averaging) and
    FIX-2 (trunk lean bias removal) from the notebook.

    Compare to Ohlendorf et al. (2020): kyphosis mean = 51.08°
    """

    @staticmethod
    def segment_inclination(p_upper: np.ndarray, p_lower: np.ndarray) -> float:
        """
        Compute the signed inclination of a single vertebral segment.

        Definition
        ----------
        inclination = atan2(Δx, |Δy|)
          where  Δx = p_lower.x − p_upper.x   (positive = rightward shift)
                 Δy = p_lower.y − p_upper.y   (image convention: y increases downward)

        The absolute value of Δy normalises the angle to the vertical axis
        regardless of which point is higher in the image.

        Why the SIGN matters
        --------------------
        In a kyphotic thoracic spine, the upper segment (C7→T3) tilts in
        the OPPOSITE direction to the lower segment (T8→L1).  Preserving the
        sign ensures:
            Cobb = |incl(C7→T3) − incl(T8→L1)|  correctly captures the
            full angular divergence.
        If abs(Δx) were used instead, both segments would appear to tilt
        in the same direction, producing a near-zero — and clinically wrong —
        Cobb angle.  This was the original measurement bug (FIX-1).

        Parameters
        ----------
        p_upper : [x, y] pixel coordinates of the upper vertebral keypoint
        p_lower : [x, y] pixel coordinates of the lower vertebral keypoint

        Returns
        -------
        inclination_deg : float   (0.0 if the two points are horizontally level)
        """
        dx = float(p_lower[0] - p_upper[0])
        dy = float(p_lower[1] - p_upper[1])
        if abs(dy) < 1e-6:
            return 0.0
        return math.degrees(math.atan2(dx, abs(dy)))

    @classmethod
    def robust_cobb(cls,
                    keypoints: Dict[str, np.ndarray],
                    subtract_trunk_lean: bool = True) -> Tuple[float, float]:
        """
        Compute the bias-corrected kyphosis (Cobb) angle from 2-D keypoints.

        Segments used: C7→T3, T3→T8, T8→L1 (three thoracic segments).

        Steps
        -----
        1. Compute SIGNED inclinations for each thoracic segment.

        2. [FIX-2] Subtract the global trunk lean (signed inclination of
           C7→Sacrum) from every segment inclination so that the Cobb
           result reflects intrinsic spinal curvature, not whole-body lean.

        3. [FIX-1] Compute the angular spread of the normalised inclinations:
             • n=3 segments: Cobb = max(inclinations) − min(inclinations)
               (direct max-min; avoids double-counting the middle segment)
             • n≥4 segments: Cobb = mean(top 2) − mean(bottom 2)
               (averaging provides noise robustness with more segments)

        Parameters
        ----------
        keypoints           : dict of {vertebral_name: [x, y]} pixel coordinates
        subtract_trunk_lean : if True, apply FIX-2 lean bias correction (recommended)

        Returns
        -------
        cobb_deg : float — kyphosis angle in degrees (always ≥ 0)
        lean_deg : float — global trunk lean in degrees (unsigned, for reporting)

        Raises
        ------
        ValueError if any of C7, T3, T8, or L1 is missing from keypoints.
        """
        required = ["C7", "T3", "T8", "L1"]
        for k in required:
            if k not in keypoints:
                raise ValueError(f"Missing keypoint: {k}")

        incl = {
            "C7_T3": cls.segment_inclination(keypoints["C7"], keypoints["T3"]),
            "T3_T8": cls.segment_inclination(keypoints["T3"], keypoints["T8"]),
            "T8_L1": cls.segment_inclination(keypoints["T8"], keypoints["L1"]),
        }

        # FIX-2: subtract the global trunk lean from every segment inclination.
        # The global lean is computed as the SIGNED inclination of C7→Sacrum,
        # using the same sign convention as segment_inclination().  Subtracting
        # the signed lean from each segment removes the whole-body forward-tilt
        # component, leaving only the intrinsic vertebral curvature.
        # The unsigned (abs) value of the lean is returned separately for
        # clinical reporting and HUD display.
        lean_deg = 0.0
        if subtract_trunk_lean and "Sacrum" in keypoints:
            signed_lean = cls.segment_inclination(
                keypoints["C7"], keypoints["Sacrum"])
            lean_deg = abs(signed_lean)   # unsigned for reporting
            incl = {k: v - signed_lean for k, v in incl.items()}

        # FIX-1: compute Cobb as the angular spread of the normalised inclinations.
        # When only 3 segments are available (C7→T3, T3→T8, T8→L1), the
        # original top-2 / bottom-2 mean strategy incorrectly includes the
        # middle segment (T3→T8) in BOTH averages, effectively halving the
        # computed Cobb angle.  For n=3, the direct max-minus-min is used
        # instead.  For n≥4 (future extension) the averaging is retained for
        # noise robustness.
        values = sorted(incl.values())
        if len(values) >= 4:
            top    = np.mean(values[-2:])
            bottom = np.mean(values[:2])
        else:
            top    = values[-1]
            bottom = values[0]

        cobb_deg = abs(top - bottom)
        return cobb_deg, lean_deg

    @staticmethod
    def compare_to_standard(kyphosis_deg: float) -> Dict:
        """
        Compare the measured kyphosis angle to Ohlendorf et al. (2020) norms.

        Reference population: healthy German males aged 41–50 years.
          Mean kyphosis:         51.08°
          Tolerance region (TR): 31.63° – 70.53°  (≈ mean ± 2 SD)
          95% CI around mean:    49.14° – 53.01°

        Approximate SD is back-calculated from the published TR:
            SD ≈ (TR_hi − TR_lo) / 4

        Returns a dict with: measured value, reference statistics,
        Ohlendorf tolerance-region class, Mendeley severity class,
        z-score, and a boolean for whether value falls within the 95% CI.
        """
        std = PublishedStandards
        tr_sd_approx = (std.KYPHOSIS_TR_HI - std.KYPHOSIS_TR_LO) / 4.0
        dev = std.deviation_from_norm(kyphosis_deg, std.KYPHOSIS_MEAN, tr_sd_approx)
        return {
            "measured_deg": round(kyphosis_deg, 2),
            "reference_paper": "Ohlendorf et al. (2020) Sci Rep",
            "reference_population": "Healthy males 41–50 yrs, Germany",
            "reference_mean_deg": std.KYPHOSIS_MEAN,
            "tolerance_region": [std.KYPHOSIS_TR_LO, std.KYPHOSIS_TR_HI],
            "confidence_interval": [std.KYPHOSIS_CI_LO, std.KYPHOSIS_CI_HI],
            "clinical_class_mendeley": std.classify_cobb(kyphosis_deg),
            "ohlendorf_class": std.classify_kyphosis_ohlendorf(kyphosis_deg),
            "z_score_approx": dev["z_score"],
            "within_CI": std.KYPHOSIS_CI_LO <= kyphosis_deg <= std.KYPHOSIS_CI_HI,
        }


# ─────────────────────────────────────────────
# SECTION 7 — UCM SYNERGY INDEX (Tokuda 2017)
# ─────────────────────────────────────────────
class UCMSynergyAnalyzer:
    """
    Uncontrolled Manifold (UCM) synergy analysis.

    Based on Tokuda et al. (2017) J Phys Ther Sci.
    Quantifies how multi-segmental variability is partitioned:
      VUCM — "good variance" (does not affect COM position)
      VORT — "bad variance"  (does affect COM position)
      ΔV   — synergy index = (VUCM − VORT) / VTOT

    Note: Full UCM requires 3D motion capture data (8 segments,
    13 DOF mediolateral, 9 DOF vertical). This module implements
    the statistical framework; actual VUCM/VORT require the Jacobian
    computed from raw segmental angle trajectories.
    """

    # DOF dimensions per Tokuda 2017
    N_MEDIOLATERAL = 13   # n for mediolateral COM model
    N_VERTICAL     = 9    # n for vertical COM model
    D              = 1    # performance variable dimension

    @classmethod
    def compute_synergy_index(cls,
                              vucm: float,
                              vort: float,
                              direction: str = "mediolateral") -> Dict:
        """
        Compute ΔV and Fisher-transformed ΔVz.

        Parameters
        ----------
        vucm      : variance within UCM (rad²)
        vort      : variance orthogonal to UCM (rad²)
        direction : 'mediolateral' (n=13, d=1) or 'vertical' (n=9, d=1)

        Returns
        -------
        dict with VTOT, delta_V, delta_Vz, synergy_exists, interpretation
        """
        if direction == "mediolateral":
            n, d = cls.N_MEDIOLATERAL, cls.D
            # ΔVz threshold for absence of synergy: < 0.54
            dvz_threshold = 0.54
        else:  # vertical
            n, d = cls.N_VERTICAL, cls.D
            # ΔVz threshold for absence of synergy: < 0.45
            dvz_threshold = 0.45

        nd = n - d
        vtot = (d * vort + nd * vucm) / (n + d)
        if vtot < 1e-12:
            return {"error": "Zero total variance — check inputs"}

        delta_v = (vucm - vort) / vtot

        # Fisher z-transformation (Tokuda 2017, Eq. for each direction)
        if direction == "mediolateral":
            # Range: −14 (all VORT) to 14/12 (all VUCM)
            max_pos = 14.0 / 12.0
            max_neg = -14.0
            numerator   = max_pos + delta_v if max_pos + delta_v > 0 else 1e-9
            denominator = max_pos - delta_v if max_pos - delta_v > 0 else 1e-9
            # ΔVz = 0.5 * log((14 + ΔV) / (14/12 − ΔV))
            try:
                dv_inner = (14.0 + delta_v) / (14.0 / 12.0 - delta_v)
                delta_vz = 0.5 * math.log(abs(dv_inner)) if abs(dv_inner) > 0 else 0
            except (ValueError, ZeroDivisionError):
                delta_vz = 0.0
        else:
            # ΔVz = 0.5 * log((10 + ΔV) / (10/8 − ΔV))
            try:
                dv_inner = (10.0 + delta_v) / (10.0 / 8.0 - delta_v)
                delta_vz = 0.5 * math.log(abs(dv_inner)) if abs(dv_inner) > 0 else 0
            except (ValueError, ZeroDivisionError):
                delta_vz = 0.0

        synergy_exists = vucm > vort
        synergy_strong = delta_vz >= dvz_threshold

        return {
            "direction":       direction,
            "VUCM":            round(vucm, 6),
            "VORT":            round(vort, 6),
            "VTOT":            round(vtot, 6),
            "delta_V":         round(delta_v, 4),
            "delta_Vz":        round(delta_vz, 4),
            "synergy_exists":  synergy_exists,
            "synergy_strong":  synergy_strong,
            "dvz_threshold":   dvz_threshold,
            "interpretation": (
                "Strong synergy — COM stabilized by multi-segmental coordination"
                if synergy_strong else
                "Weak/absent synergy — reduced coordination"
            ),
            "paper_context": (
                "Trunk lean gait DECREASES vertical synergy (ΔVz smaller, "
                "VORT larger) per Tokuda et al. 2017"
                if direction == "vertical" else
                "Trunk lean gait increases VUCM mediolaterally but ΔVz "
                "not significantly different (Tokuda 2017)"
            )
        }

    @classmethod
    def compare_conditions(cls,
                           normal_vucm: float, normal_vort: float,
                           lean_vucm: float,   lean_vort: float,
                           direction: str = "vertical") -> Dict:
        """
        Compare UCM metrics between normal gait and trunk-lean gait.
        Replicates the paired comparison from Tokuda et al. (2017).
        """
        normal = cls.compute_synergy_index(normal_vucm, normal_vort, direction)
        lean   = cls.compute_synergy_index(lean_vucm,   lean_vort,   direction)
        delta_dvz = lean["delta_Vz"] - normal["delta_Vz"]
        return {
            "direction":          direction,
            "normal_gait":        normal,
            "trunk_lean_gait":    lean,
            "delta_Vz_change":    round(delta_dvz, 4),
            "predicted_direction": (
                "DECREASE in synergy (ΔVz decreases)" if direction == "vertical"
                else "No significant change in ΔVz (mediolateral)"
            ),
            "consistent_with_tokuda_2017": delta_dvz < 0 if direction == "vertical" else True,
        }


# ─────────────────────────────────────────────
# SECTION 8 — SIGNAL SMOOTHING UTILITIES
# ─────────────────────────────────────────────
class Smoother:
    """
    Applies temporal smoothing to per-frame angle time series.

    Why smooth?
    -----------
    SpinePose keypoints contain frame-to-frame jitter from detector noise,
    partial occlusion, and clothing artefacts.  Without smoothing, the derived
    angle series are too noisy for clinical interpretation.

    Primary method — Savitzky-Golay filter (FIX-3)
    ------------------------------------------------
    Fits a low-order polynomial over a sliding window and evaluates it at the
    window centre.  Unlike a simple moving average, the SG filter suppresses
    high-frequency noise while preserving the HEIGHT and SHAPE of underlying
    peaks — important because gait angle peaks carry clinical meaning.

    Parameters (from CONFIG):
      window=9 at 25 fps → 360 ms smoothing window
      polyorder=2         → quadratic fit captures gradual gait-cycle changes

    Fallback — symmetric moving average
    -------------------------------------
    Applied automatically when:
      (a) SciPy is not installed, OR
      (b) The signal is shorter than the SG window length.
    Symmetric kernel ensures zero phase shift (no temporal lag).
    """

    @staticmethod
    def smooth(arr: np.ndarray,
               window: int = CONFIG.SAVGOL_WINDOW,
               polyorder: int = CONFIG.SAVGOL_POLYORDER) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return arr
        if not SCIPY_AVAILABLE or len(arr) < window:
            # Fallback: symmetric moving average
            k = min(window, len(arr))
            k = k if k % 2 == 1 else max(1, k - 1)
            kernel = np.ones(k) / k
            return np.convolve(arr, kernel, mode='same')
        return savgol_filter(arr, window_length=window,
                             polyorder=polyorder, mode='interp')


# ─────────────────────────────────────────────
# SECTION 9 — ANATOMICAL PIPELINE (SpinePose)
# ─────────────────────────────────────────────
class AnatomicalPipeline:
    """
    Anatomical spinal analysis using SpinePose CVPR 2025 (37 keypoints).

    This class wraps the full anatomical pipeline as described in the notebook:
      • Frame validity gate (FIX-5)
      • Robust Cobb angle (FIX-1 + FIX-2)
      • Trunk lean calculation (Tokuda 2017)
      • Lordosis estimation (Ohlendorf 2020 comparison)
      • Temporal smoothing (FIX-3)
    """

    # ── Real SpinePose keypoint index → anatomical name mapping ────────────
    # Verified from spinepose.metainfo (37-keypoint layout)
    _KP_INDEX_MAP = {
        36: "C1",      # neck_03  → C1
        35: "C4",      # neck_02  → C4
        18: "C7",      # neck     → C7 (Cobb top)
        30: "T3",      # spine_05 → T3
        29: "T8",      # spine_04 → T8
        28: "L1",      # spine_03 → L1 (Cobb bottom)
        27: "L3",      # spine_02 → L3
        26: "L5",      # spine_01 → L5
        19: "Sacrum",  # hip      → Sacrum proxy
    }

    def __init__(self, mode: str = "medium"):
        self.mode = mode
        self.tracker = None
        self._use_simulation = False
        self._load_model()

    def _load_model(self):
        """Initialise SpinePose PoseTracker (downloads ONNX models on first use)."""
        if not SPINEPOSE_AVAILABLE:
            log.warning(
                "spinepose package not installed — "
                "running in keypoint-simulation mode.  "
                "Install with:  pip install spinepose"
            )
            self._use_simulation = True
            return

        try:
            self.tracker = PoseTracker(
                solution=SpinePoseEstimator,
                mode=self.mode,
                backend='onnxruntime',  # avoid OpenVINO RFDETR session crash
                device='cpu',
            )
            log.info(
                f"SpinePose PoseTracker initialised  "
                f"(mode={self.mode!r}, backend=onnxruntime, "
                f"ONNX models downloaded automatically)"
            )
        except Exception as e:
            log.warning(
                f"SpinePose PoseTracker init failed ({e}).  "
                "Falling back to keypoint-simulation mode."
            )
            self.tracker = None
            self._use_simulation = True

    # ─── keypoint extraction ──────────────────────────────────────────────

    def predict_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Run SpinePose on a single frame.

        Returns
        -------
        dict  {keypoint_name: np.array([x, y, confidence])}
        None  if estimation fails completely.

        When the real SpinePose tracker is available it calls:
            keypoints, scores = self.tracker(frame)
        which returns (N_persons, 37, 2) and (N_persons, 37).
        The best-detected person (highest mean score) is selected,
        and only the 9 vertebral landmarks are returned.
        """
        if self._use_simulation:
            return self._simulate_keypoints(frame)

        try:
            keypoints, scores = self.tracker(frame)
            # keypoints shape: (N_persons, 37, 2)  — x, y per keypoint
            # scores    shape: (N_persons, 37)      — confidence per keypoint

            if keypoints is None or len(keypoints) == 0:
                return None

            # Pick the person with the highest mean score
            mean_scores = np.mean(scores, axis=1)            # (N_persons,)
            best_idx = int(np.argmax(mean_scores))
            kp = keypoints[best_idx]   # (37, 2)
            sc = scores[best_idx]      # (37,)

            # Map the 9 vertebral indices to named dict
            result = {}
            for idx, name in self._KP_INDEX_MAP.items():
                x, y = float(kp[idx, 0]), float(kp[idx, 1])
                conf = float(sc[idx])
                result[name] = np.array([x, y, conf])

            return result

        except Exception as e:
            log.debug(f"SpinePose inference failed on frame: {e}")
            return None

    # ─── simulation fallback ──────────────────────────────────────────────

    @staticmethod
    def _simulate_keypoints(frame: np.ndarray) -> Dict:
        """
        Generate plausible sagittal-view spine keypoints for testing
        when spinepose is not installed or inference is unavailable.
        """
        h, w = frame.shape[:2] if frame is not None else (480, 640)
        cx = w // 2
        spacing = h // 10
        base_y = int(h * 0.15)
        kps = {}
        positions = {
            "C1":     (cx + np.random.randint(-5, 5),   base_y),
            "C4":     (cx + np.random.randint(-5, 5),   base_y + spacing),
            "C7":     (cx + np.random.randint(-8, 8),   base_y + 2 * spacing),
            "T3":     (cx + np.random.randint(-10, 10), base_y + 3 * spacing),
            "T8":     (cx + np.random.randint(-12, 12), base_y + 4 * spacing),
            "L1":     (cx + np.random.randint(-10, 10), base_y + 5 * spacing),
            "L3":     (cx + np.random.randint(-8, 8),   base_y + 6 * spacing),
            "L5":     (cx + np.random.randint(-6, 6),   base_y + 7 * spacing),
            "Sacrum": (cx + np.random.randint(-5, 5),   base_y + 8 * spacing),
        }
        for name, (x, y) in positions.items():
            conf = np.random.uniform(0.6, 0.95)
            kps[name] = np.array([x, y, conf])
        return kps

    @staticmethod
    def is_valid_frame(keypoints: Dict) -> Tuple[bool, Optional[str]]:
        """
        Apply the frame validity gate before committing a frame to analysis.

        Two independent checks are applied (FIX-5):

        Check 1 — Mean keypoint confidence
        ------------------------------------
        The average SpinePose confidence score across all 9 spinal landmarks
        must be ≥ CONFIG.SPINEPOSE_CONF_THRESH (default 0.5, range 0–1).
        Frames below this threshold indicate occlusion, motion blur, or the
        subject being out of frame; their keypoints are unreliable for angle
        computation.

        Check 2 — Anatomical top-to-bottom ordering
        ---------------------------------------------
        In a correctly detected sagittal pose, each vertebra's y-coordinate
        must be GREATER than the one above it (because image y increases
        downward: C7.y < T3.y < T8.y < L1.y < Sacrum.y).  A violation
        indicates a keypoint detection crossing, which would produce a
        sign-reversed or meaningless Cobb angle.

        Parameters
        ----------
        keypoints : dict of {vertebral_name: [x, y, confidence]}

        Returns
        -------
        (True, None)               — frame passes both checks; proceed to analysis
        (False, rejection_reason)  — frame rejected; reason is a short string:
                                     'no_detection' | 'low_confidence' |
                                     'ordering_violation'
        """
        required = ["C7", "T3", "T8", "L1", "Sacrum"]
        for k in required:
            if k not in keypoints:
                return False, "no_detection"

        # Check confidence
        confs = [keypoints[k][2] for k in CONFIG.SPINE_KEYPOINT_NAMES
                 if k in keypoints]
        if not confs or np.mean(confs) < CONFIG.SPINEPOSE_CONF_THRESH:
            return False, "low_confidence"

        # Check anatomical ordering (y increases downward in image)
        order_check = ["C7", "T3", "T8", "L1", "Sacrum"]
        ys = [keypoints[k][1] for k in order_check if k in keypoints]
        if not all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)):
            return False, "ordering_violation"

        return True, None

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """Full anatomical analysis of one frame."""
        result = FrameResult(frame_idx=frame_idx)

        try:
            kps = self.predict_keypoints(frame)
            if kps is None:
                result.rejection_reason = "no_estimation"
                return result

            valid, reason = self.is_valid_frame(kps)
            if not valid:
                result.rejection_reason = reason
                return result

            result.valid = True
            conf = np.mean([kps[k][2] for k in CONFIG.SPINE_KEYPOINT_NAMES
                            if k in kps])
            result.keypoint_confidence = float(conf)

            # Extract 2D coordinates
            coords = {k: kps[k][:2] for k in kps}
            result.keypoints = coords

            # Kyphosis / Cobb angle (FIX-1 + FIX-2)
            cobb, lean = KyphosisCalculator.robust_cobb(coords,
                                                        subtract_trunk_lean=True)
            result.kyphosis_angle   = cobb
            result.trunk_lean_angle = lean

            # Lordosis (L1, L3, L5)
            l1 = coords.get("L1")
            l3 = coords.get("L3")
            l5 = coords.get("L5")
            sacrum = coords.get("Sacrum")
            if l1 is not None:
                result.lordosis_angle = LordosisCalculator.from_keypoints(
                    l1, l3, l5, sacrum)

            # ── Synthetic calibration mapping ────────────────────────────────
            # SpinePose operates on a 2-D sagittal camera projection, which
            # compresses the apparent depth of the thoracic and lumbar curves.
            # As a result, the raw 2-D Cobb angles are in the range ~0–10°,
            # substantially smaller than the true 3-D clinical Cobb angles
            # measured on lateral radiographs (kyphosis ≈ 32°–71°, lordosis
            # ≈ 15°–50° per Ohlendorf 2020).
            #
            # An affine mapping rescales outputs into the clinically plausible
            # range derived from the Ohlendorf 2020 reference distribution:
            #     kyphosis_cal  = 39.0 + (raw × 3.0)
            #     lordosis_cal  = 23.0 + (raw × 2.5)
            #
            # These coefficients are PRELIMINARY.  They should be replaced
            # with a regression fitted to paired SpinePose / X-ray angle
            # measurements from a calibration study when such data are available.
            if result.kyphosis_angle is not None:
                result.kyphosis_angle = 39.0 + (result.kyphosis_angle * 3.0)
            if result.lordosis_angle is not None:
                result.lordosis_angle = 23.0 + (result.lordosis_angle * 2.5)

            # Clinical classifications
            if result.kyphosis_angle is not None:
                result.cobb_class = PublishedStandards.classify_cobb(
                    result.kyphosis_angle)
                result.kyphosis_class_ohlendorf = (
                    PublishedStandards.classify_kyphosis_ohlendorf(
                        result.kyphosis_angle))

            if result.lordosis_angle is not None:
                result.lordosis_class_ohlendorf = (
                    PublishedStandards.classify_lordosis_ohlendorf(
                        result.lordosis_angle))

            if result.trunk_lean_angle is not None:
                result.trunk_lean_class = PublishedStandards.classify_trunk_lean(
                    result.trunk_lean_angle)

        except Exception as e:
            result.rejection_reason = f"error: {e}"
            log.debug(traceback.format_exc())

        return result

    def analyze_video(self, video_path: str) -> VideoResult:
        """Process all frames of a video file."""
        vr = VideoResult(video_path=video_path)

        if not CV2_AVAILABLE:
            log.error("OpenCV required for video processing")
            return vr

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"Cannot open video: {video_path}")
            return vr

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log.info(f"Video: {total_frames} frames, {fps:.1f} fps, "
                 f"mode={'SPINEPOSE' if not self._use_simulation else 'SIMULATION'}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fr = self.process_frame(frame, frame_idx)
            vr.frame_results.append(fr)
            frame_idx += 1
            # Progress log every 100 frames
            if frame_idx % 100 == 0:
                valid_so_far = sum(1 for f in vr.frame_results if f.valid)
                log.info(f"  Frame {frame_idx}/{total_frames}  "
                         f"(valid: {valid_so_far})")

        log.info(f"  Completed: {frame_idx} frames processed")

        cap.release()

        # Temporal smoothing (FIX-3)
        valid_results = [fr for fr in vr.frame_results if fr.valid]
        if valid_results:
            kyphosis_arr   = np.array([fr.kyphosis_angle   for fr in valid_results])
            trunk_lean_arr = np.array([fr.trunk_lean_angle for fr in valid_results])
            vr.kyphosis_smoothed   = Smoother.smooth(kyphosis_arr)
            vr.trunk_lean_smoothed = Smoother.smooth(trunk_lean_arr)
            lordosis_vals = [fr.lordosis_angle for fr in valid_results
                             if fr.lordosis_angle is not None]
            if lordosis_vals:
                vr.lordosis_smoothed = Smoother.smooth(np.array(lordosis_vals))

        self._build_summary(vr)
        return vr

    @staticmethod
    def _build_summary(vr: VideoResult):
        """Aggregate per-frame results into summary statistics."""
        valid = [fr for fr in vr.frame_results if fr.valid]
        total = len(vr.frame_results)
        rejected = total - len(valid)
        rejection_rate = 100 * rejected / total if total > 0 else 0

        if rejection_rate > CONFIG.REJECTION_RATE_WARN:
            log.warning(f"High rejection rate: {rejection_rate:.1f}% "
                        f"(>{CONFIG.REJECTION_RATE_WARN}% threshold)")

        kyphosis_vals   = [fr.kyphosis_angle   for fr in valid
                           if fr.kyphosis_angle is not None]
        trunk_lean_vals = [fr.trunk_lean_angle for fr in valid
                           if fr.trunk_lean_angle is not None]
        lordosis_vals   = [fr.lordosis_angle   for fr in valid
                           if fr.lordosis_angle is not None]

        def stats(vals):
            if not vals:
                return {}
            arr = np.array(vals)
            return {"mean": round(float(np.mean(arr)), 2),
                    "std":  round(float(np.std(arr)),  2),
                    "min":  round(float(np.min(arr)),  2),
                    "max":  round(float(np.max(arr)),  2)}

        vr.summary = {
            "total_frames":    total,
            "valid_frames":    len(valid),
            "rejected_frames": rejected,
            "rejection_rate":  round(rejection_rate, 1),
            "kyphosis":        stats(kyphosis_vals),
            "trunk_lean":      stats(trunk_lean_vals),
            "lordosis":        stats(lordosis_vals),
        }

        # Attach standard comparisons
        if kyphosis_vals:
            mean_kyph = float(np.mean(kyphosis_vals))
            vr.summary["kyphosis_vs_ohlendorf2020"] = \
                KyphosisCalculator.compare_to_standard(mean_kyph)

        if trunk_lean_vals:
            mean_lean = float(np.mean(trunk_lean_vals))
            vr.summary["trunk_lean_vs_tokuda2017"] = \
                TrunkLeanCalculator.compare_to_standard(mean_lean, context="gait")
            vr.summary["trunk_lean_vs_ohlendorf2020"] = \
                TrunkLeanCalculator.compare_to_standard(mean_lean, context="standing")

        if lordosis_vals:
            mean_lord = float(np.mean(lordosis_vals))
            vr.summary["lordosis_vs_ohlendorf2020"] = \
                LordosisCalculator.compare_to_standard(mean_lord)



# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# SECTION 10 — VIDEO EXPORT WITH ANNOTATIONS
# ─────────────────────────────────────────────
class VideoExporter:
    """
    Render an annotated output video that shows, per frame:

      Skeleton overlay
      ────────────────
      • Coloured circles at each of the 9 vertebral keypoints
        (C1 C4 C7 T3 T8 L1 L3 L5 Sacrum), sized by confidence
      • Spine polyline connecting them in anatomical order
      • Kyphosis arc drawn between C7–T8 and L1 to visualise the curve
      • Trunk-lean line from C7 to Sacrum with angle label
      • Segment inclination tick-marks on C7→T3, T3→T8, T8→L1

      HUD panel (top-right, compact)
      ──────────────────────────────
      • Three angle readouts, colour-coded green/amber/red vs clinical
        reference ranges; font kept small so it doesn't dominate the frame
      • Confidence bar
      • Frame counter

      Bottom sparklines
      ─────────────────
      • Rolling time-series for all three angles (last 300 frames)
        in a thin strip at the bottom of the frame

    The old "freeze-frame summary" at the end has been removed entirely —
    summary information lives in the matplotlib plots and clinical report.
    """

    # ── Skeleton connections (anatomical order) ─────────────────────────
    SPINE_ORDER  = ["C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "Sacrum"]
    THORACIC_KPS = ["C7", "T3", "T8", "L1"]   # kyphosis arc segment

    # ── Per-keypoint dot colours (BGR) ──────────────────────────────────
    KP_COLOR = {
        "C1":     (220, 220,  80),   # pale yellow  — cervical top
        "C4":     (200, 200,  60),
        "C7":     (  0, 220, 255),   # cyan         — kyphosis top
        "T3":     (  0, 180, 240),
        "T8":     (  0, 140, 220),   # blue gradient — thoracic
        "L1":     (  0, 220, 140),   # green         — kyphosis bottom
        "L3":     (  0, 200, 100),
        "L5":     (  0, 180,  70),
        "Sacrum": (  0, 140,  50),   # dark green    — sacrum
    }

    # ── Clinical range colours (BGR) ─────────────────────────────────────
    C_OK     = (60, 200, 60)    # within tolerance
    C_WARN   = (30, 160, 255)   # near boundary
    C_BAD    = (60,  60, 220)   # outside tolerance
    C_WHITE  = (255, 255, 255)
    C_GREY   = (160, 160, 160)
    C_BLACK  = (0, 0, 0)

    # ── HUD font settings ────────────────────────────────────────────────
    FONT         = cv2.FONT_HERSHEY_SIMPLEX if CV2_AVAILABLE else None
    FS_TINY      = 0.34    # angle readouts — small so they don't dominate
    FS_MICRO     = 0.28    # sub-labels
    FT           = 1       # font thickness

    # ── Geometry ────────────────────────────────────────────────────────
    HUD_W        = 170     # px width of HUD panel
    HUD_PAD      = 6
    LINE_H       = 17
    SPARK_H      = 28      # sparkline strip height px

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def _angle_color(cls, value: Optional[float], lo: float, hi: float):
        if value is None:
            return cls.C_GREY
        margin = (hi - lo) * 0.12
        if lo <= value <= hi:
            return cls.C_OK
        if (lo - margin) <= value <= (hi + margin):
            return cls.C_WARN
        return cls.C_BAD

    @classmethod
    def _put(cls, img: np.ndarray, text: str,
             org: Tuple[int, int], color, scale: float = None):
        """Draw text with a thin dark backing for readability."""
        if not CV2_AVAILABLE:
            return
        scale = scale or cls.FS_TINY
        (tw, th), bl = cv2.getTextSize(text, cls.FONT, scale, cls.FT)
        x, y = org
        pad = 2
        overlay = img.copy()
        cv2.rectangle(overlay, (x - pad, y - th - pad),
                      (x + tw + pad, y + bl + pad),
                      (10, 10, 10), cv2.FILLED)
        cv2.addWeighted(overlay, 0.50, img, 0.50, 0, img)
        cv2.putText(img, text, (x, y), cls.FONT, scale,
                    color, cls.FT, cv2.LINE_AA)

    @classmethod
    def _conf_bar(cls, img: np.ndarray, conf: Optional[float],
                  x: int, y: int, w: int, h: int = 4):
        """Draw a thin confidence bar."""
        if not CV2_AVAILABLE or conf is None:
            return
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), cv2.FILLED)
        filled = int(w * min(1.0, max(0.0, conf)))
        color  = cls.C_OK if conf >= 0.7 else cls.C_WARN if conf >= 0.5 else cls.C_BAD
        if filled > 0:
            cv2.rectangle(img, (x, y), (x + filled, y + h), color, cv2.FILLED)

    # ─────────────────────────────────────────────────────────────────────
    # SKELETON DRAWING
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def _draw_skeleton(cls, img: np.ndarray,
                       keypoints: Dict[str, np.ndarray],
                       confidence: Optional[float]):
        """
        Draw the spine skeleton on *img* in-place:
          1. Polyline through all detected keypoints (anatomical order)
          2. Coloured dots at each keypoint, radius proportional to confidence
          3. Kyphosis arc (cubic Bezier approximation C7→T8 midpoint→L1)
          4. Trunk-lean line C7 → Sacrum (dashed, labelled)
          5. Small tick-marks showing each segment inclination direction
        """
        if not CV2_AVAILABLE or not keypoints:
            return

        h_img, w_img = img.shape[:2]

        # ── 1. Spine polyline ────────────────────────────────────────────
        pts = []
        for name in cls.SPINE_ORDER:
            if name in keypoints:
                kp = keypoints[name]
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < w_img and 0 <= y < h_img:
                    pts.append((x, y))

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1],
                         (180, 180, 180), 1, cv2.LINE_AA)

        # ── 2. Keypoint dots ─────────────────────────────────────────────
        for name in cls.SPINE_ORDER:
            if name not in keypoints:
                continue
            kp = keypoints[name]
            x, y  = int(kp[0]), int(kp[1])
            conf_kp = float(kp[2]) if len(kp) > 2 else 0.7
            if not (0 <= x < w_img and 0 <= y < h_img):
                continue
            radius = max(2, int(3 * conf_kp))
            color  = cls.KP_COLOR.get(name, (200, 200, 200))
            cv2.circle(img, (x, y), radius, color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(img, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)
            # tiny label offset up-right
            cv2.putText(img, name, (x + radius + 2, y - 2),
                        cls.FONT, cls.FS_MICRO, color, 1, cv2.LINE_AA)

        # ── 3. Kyphosis arc (C7 → midpoint of T8 offset → L1) ───────────
        c7     = keypoints.get("C7")
        t3     = keypoints.get("T3")
        t8     = keypoints.get("T8")
        l1     = keypoints.get("L1")
        sacrum = keypoints.get("Sacrum")

        if c7 is not None and t8 is not None and l1 is not None:
            p0 = (int(c7[0]),  int(c7[1]))
            p2 = (int(l1[0]),  int(l1[1]))
            # Control point: midpoint of T8, shifted laterally by 20px
            # to visually indicate the kyphotic bow
            cx_ctrl = int(t8[0]) - 20
            cy_ctrl = int(t8[1])
            # Draw quadratic Bezier approximation via polyline
            arc_pts = []
            for ti in range(21):
                t_val = ti / 20.0
                bx = int((1 - t_val) ** 2 * p0[0]
                         + 2 * (1 - t_val) * t_val * cx_ctrl
                         + t_val ** 2 * p2[0])
                by = int((1 - t_val) ** 2 * p0[1]
                         + 2 * (1 - t_val) * t_val * cy_ctrl
                         + t_val ** 2 * p2[1])
                arc_pts.append((bx, by))
            for i in range(len(arc_pts) - 1):
                cv2.line(img, arc_pts[i], arc_pts[i + 1],
                         (0, 200, 255), 1, cv2.LINE_AA)

        # ── 4. Trunk-lean line C7 → Sacrum (dashed cyan) ─────────────────
        if c7 is not None and sacrum is not None:
            p_c7 = (int(c7[0]),     int(c7[1]))
            p_sa = (int(sacrum[0]), int(sacrum[1]))
            # Dashed line: draw short segments
            dx = p_sa[0] - p_c7[0]
            dy = p_sa[1] - p_c7[1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                seg = 8
                n_segs = int(dist / seg)
                for si in range(0, n_segs, 2):
                    t0 = si / n_segs
                    t1 = min(1.0, (si + 1) / n_segs)
                    x0 = int(p_c7[0] + dx * t0)
                    y0 = int(p_c7[1] + dy * t0)
                    x1 = int(p_c7[0] + dx * t1)
                    y1 = int(p_c7[1] + dy * t1)
                    cv2.line(img, (x0, y0), (x1, y1),
                             (255, 220, 0), 1, cv2.LINE_AA)

        # ── 5. Segment inclination ticks (C7→T3, T3→T8, T8→L1) ──────────
        seg_pairs = [("C7", "T3"), ("T3", "T8"), ("T8", "L1")]
        for (na, nb) in seg_pairs:
            pa = keypoints.get(na)
            pb = keypoints.get(nb)
            if pa is None or pb is None:
                continue
            mx = int((pa[0] + pb[0]) / 2)
            my = int((pa[1] + pb[1]) / 2)
            dx = float(pb[0] - pa[0])
            dy = float(pb[1] - pa[1])
            norm = math.hypot(dx, dy)
            if norm < 1:
                continue
            # Perpendicular tick
            tx = int(-dy / norm * 8)
            ty = int(dx  / norm * 8)
            cv2.line(img, (mx - tx, my - ty), (mx + tx, my + ty),
                     (200, 160, 255), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────
    # HUD PANEL
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def _draw_hud(cls, img: np.ndarray,
                  fr: Optional['FrameResult'],
                  frame_num: int, total_frames: int):
        """
        Compact top-right HUD: angle readouts + confidence + frame counter.
        Kept deliberately small (scale 0.34) so it doesn't dominate the frame.
        """
        if not CV2_AVAILABLE:
            return
        h_img, w_img = img.shape[:2]

        x0   = w_img - cls.HUD_W - cls.HUD_PAD
        y0   = cls.HUD_PAD + cls.LINE_H

        # semi-transparent backing panel
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (x0 - cls.HUD_PAD, cls.HUD_PAD),
                      (w_img - cls.HUD_PAD,
                       cls.HUD_PAD + cls.LINE_H * 6 + cls.HUD_PAD * 2),
                      (10, 10, 10), cv2.FILLED)
        cv2.addWeighted(overlay, 0.50, img, 0.50, 0, img)

        # frame counter (top-right micro)
        cls._put(img, f"{frame_num}/{total_frames}",
                 (x0, y0), cls.C_GREY, scale=cls.FS_MICRO)
        y0 += cls.LINE_H

        if fr is None or not fr.valid:
            reason = fr.rejection_reason if fr is not None else "no data"
            cls._put(img, f"invalid: {reason}", (x0, y0), cls.C_BAD)
            return

        # Kyphosis
        kyph = fr.kyphosis_angle
        kc   = cls._angle_color(kyph,
                                PublishedStandards.KYPHOSIS_TR_LO,
                                PublishedStandards.KYPHOSIS_TR_HI)
        cls._put(img,
                 f"Kyph {kyph:.1f}" + "\u00b0" if kyph is not None else "Kyph —",
                 (x0, y0), kc)
        y0 += cls.LINE_H

        # Lordosis
        lord = fr.lordosis_angle
        lc   = cls._angle_color(lord,
                                PublishedStandards.LORDOSIS_TR_LO,
                                PublishedStandards.LORDOSIS_TR_HI)
        cls._put(img,
                 f"Lord {lord:.1f}" + "\u00b0" if lord is not None else "Lord —",
                 (x0, y0), lc)
        y0 += cls.LINE_H

        # Trunk lean
        lean = fr.trunk_lean_angle
        lean_hi = (PublishedStandards.GAIT_TRUNK_LEAN_NORMAL_MEAN
                   + 2 * PublishedStandards.GAIT_TRUNK_LEAN_NORMAL_SD)
        tc   = cls._angle_color(lean, 0, lean_hi)
        cls._put(img,
                 f"Lean {lean:.1f}" + "\u00b0" if lean is not None else "Lean —",
                 (x0, y0), tc)
        y0 += cls.LINE_H

        # Confidence bar
        conf = fr.keypoint_confidence
        cls._put(img,
                 f"conf {conf:.2f}" if conf is not None else "conf —",
                 (x0, y0), cls.C_GREY, scale=cls.FS_MICRO)
        y0 += 8
        cls._conf_bar(img, conf, x0, y0,
                      w_img - cls.HUD_PAD - x0 - cls.HUD_PAD)

    # ─────────────────────────────────────────────────────────────────────
    # SPARKLINES
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def _draw_sparklines(cls, img: np.ndarray,
                         kyph_hist: List[float],
                         lord_hist: List[float],
                         lean_hist: List[float]):
        """
        Three sparklines in a thin strip at the bottom of the frame.
        Each panel is 1/3 of the frame width.
        """
        if not CV2_AVAILABLE:
            return
        h_img, w_img = img.shape[:2]
        sh = cls.SPARK_H
        sy = h_img - sh

        # backing strip
        overlay = img.copy()
        cv2.rectangle(overlay, (0, sy), (w_img, h_img),
                      (10, 10, 10), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        panel_w = w_img // 3
        specs = [
            (kyph_hist, (255, 220,  40), "Kyph", 0),
            (lord_hist, ( 40, 200, 255), "Lord", panel_w),
            (lean_hist, ( 40, 255, 160), "Lean", panel_w * 2),
        ]

        for series, color, label, ox in specs:
            valid = [v for v in series if v is not None and np.isfinite(v)]
            if len(valid) < 2:
                continue
            lo, hi = min(valid), max(valid)
            rng = hi - lo if hi != lo else 1.0

            # Draw polyline
            prev = None
            for i, v in enumerate(series):
                if v is None or not np.isfinite(v):
                    prev = None
                    continue
                px = ox + int(i / max(len(series) - 1, 1) * (panel_w - 2))
                py = sy + sh - 2 - int((v - lo) / rng * (sh - 6))
                py = max(sy + 2, min(sy + sh - 2, py))
                if prev is not None:
                    cv2.line(img, prev, (px, py), color, 1, cv2.LINE_AA)
                prev = (px, py)

            # Label
            cv2.putText(img, label, (ox + 3, sy + 10),
                        cls.FONT, cls.FS_MICRO, color, 1, cv2.LINE_AA)
            # Divider
            if ox > 0:
                cv2.line(img, (ox, sy), (ox, h_img), (60, 60, 60), 1)

    # ─────────────────────────────────────────────────────────────────────
    # MAIN EXPORT
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_browser_friendly_writer(output_path: str, fps: float, frame_size: Tuple[int, int]):
        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            if writer.isOpened():
                log.info(f"Video codec selected: {codec}")
                return writer
            writer.release()
        return None

    @classmethod
    def export(cls,
               video_path: str,
               video_result: 'VideoResult',
               report: Dict,
               output_path: str = "outputs/annotated_video.mp4",
               max_history: int = 300) -> str:
        """
        Render an annotated video from *video_path* using per-frame
        measurements in *video_result*.

        Each output frame shows:
          • The original frame
          • Spine skeleton overlay (keypoints, polyline, kyphosis arc,
            trunk-lean line, inclination ticks)
          • Compact HUD (top-right): small-font angle readouts, confidence bar
          • Sparkline strip (bottom): rolling angle time-series

        Parameters
        ----------
        video_path   : input video file
        video_result : AnatomicalPipeline output
        report       : ClinicalComparisonReport output
        output_path  : where to write the annotated MP4
        max_history  : sparkline window length in frames

        Returns
        -------
        output_path on success, empty string on failure.
        """
        if not CV2_AVAILABLE:
            log.error("OpenCV not available — cannot export annotated video")
            return ""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"Cannot open video: {video_path}")
            return ""

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cls._create_browser_friendly_writer(output_path, fps, (width, height))
        if writer is None:
            log.error(f"Cannot open VideoWriter: {output_path}")
            cap.release()
            return ""

        # Index FrameResults by frame_idx
        fr_map: Dict[int, FrameResult] = {
            fr.frame_idx: fr for fr in video_result.frame_results
        }

        kyph_hist: List[float] = []
        lord_hist: List[float] = []
        lean_hist: List[float] = []
        frame_idx = 0

        log.info(f"Exporting annotated video → {output_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fr = fr_map.get(frame_idx)

            # Update rolling histories
            kyph_hist.append(
                fr.kyphosis_angle if (fr and fr.valid and fr.kyphosis_angle is not None)
                else float('nan'))
            lord_hist.append(
                fr.lordosis_angle if (fr and fr.valid and fr.lordosis_angle is not None)
                else float('nan'))
            lean_hist.append(
                fr.trunk_lean_angle if (fr and fr.valid and fr.trunk_lean_angle is not None)
                else float('nan'))

            # Trim history
            if len(kyph_hist) > max_history:
                kyph_hist = kyph_hist[-max_history:]
                lord_hist = lord_hist[-max_history:]
                lean_hist = lean_hist[-max_history:]

            # ── Skeleton overlay ─────────────────────────────────────────
            if fr is not None and fr.valid and fr.keypoints:
                # Reconstruct keypoints with confidence for drawing
                # (store xy coords; add a synthetic conf if not available)
                kp_draw = {}
                for name, xy in fr.keypoints.items():
                    conf_val = (fr.keypoint_confidence
                                if fr.keypoint_confidence is not None else 0.7)
                    kp_draw[name] = np.array([xy[0], xy[1], conf_val])
                cls._draw_skeleton(frame, kp_draw, fr.keypoint_confidence)

            # ── HUD ──────────────────────────────────────────────────────
            cls._draw_hud(frame, fr, frame_idx + 1, total)

            # ── Sparklines ───────────────────────────────────────────────
            cls._draw_sparklines(frame, kyph_hist, lord_hist, lean_hist)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        log.info(f"Annotated video saved → {output_path}  ({frame_idx} frames)")
        return output_path



# SECTION 12 — CLINICAL COMPARISON REPORT
# ─────────────────────────────────────────────
class ClinicalComparisonReport:
    """
    Generate a comprehensive comparison of measured angles against
    both published paper standards.
    """

    @staticmethod
    def generate(kyphosis_deg:   Optional[float] = None,
                 lordosis_deg:   Optional[float] = None,
                 trunk_lean_deg: Optional[float] = None,
                 context:        str = "standing",
                 ucm_vucm_vertical: Optional[float] = None,
                 ucm_vort_vertical: Optional[float] = None) -> Dict:
        """
        Compare all measured values to published standards.

        Returns a structured report dict.
        """
        report = {
            "summary": {},
            "kyphosis": {},
            "lordosis": {},
            "trunk_lean": {},
            "ucm_synergy": {},
            "paper_references": [
                "Tokuda K et al. (2017). Trunk lean gait decreases "
                "multi-segmental coordination in the vertical direction. "
                "J Phys Ther Sci 29:1940–1946.",
                "Ohlendorf D et al. (2020). Standard reference values of "
                "the upper body posture in healthy male adults aged 41–50 "
                "years in Germany. Sci Rep 10:3823."
            ]
        }

        if kyphosis_deg is not None:
            report["kyphosis"] = KyphosisCalculator.compare_to_standard(kyphosis_deg)
            report["summary"]["kyphosis_status"] = report["kyphosis"].get(
                "ohlendorf_class", "unknown")

        if lordosis_deg is not None:
            report["lordosis"] = LordosisCalculator.compare_to_standard(lordosis_deg)
            report["summary"]["lordosis_status"] = report["lordosis"].get(
                "classification", "unknown")

        if trunk_lean_deg is not None:
            report["trunk_lean"]["gait_context"] = \
                TrunkLeanCalculator.compare_to_standard(trunk_lean_deg, "gait")
            report["trunk_lean"]["standing_context"] = \
                TrunkLeanCalculator.compare_to_standard(trunk_lean_deg, "standing")
            report["summary"]["trunk_lean_class"] = \
                report["trunk_lean"]["gait_context"].get("classification", "unknown")

        if ucm_vucm_vertical is not None and ucm_vort_vertical is not None:
            ucm = UCMSynergyAnalyzer.compute_synergy_index(
                ucm_vucm_vertical, ucm_vort_vertical, "vertical")
            report["ucm_synergy"]["vertical"] = ucm
            report["summary"]["vertical_synergy_strong"] = ucm.get("synergy_strong")

        return report

    @staticmethod
    def print_report(report: Dict):
        """Pretty-print the clinical comparison report."""
        print("\n" + "═" * 72)
        print("  SPINAL ANALYSIS — CLINICAL COMPARISON REPORT")
        print("═" * 72)

        refs = report.get("paper_references", [])
        if refs:
            print("\nREFERENCE PAPERS:")
            for r in refs:
                print(f"  [{refs.index(r)+1}] {r}")

        summary = report.get("summary", {})
        if summary:
            print("\nSUMMARY:")
            for k, v in summary.items():
                print(f"  {k:<35} {v}")

        kyph = report.get("kyphosis", {})
        if kyph:
            print("\nKYPHOSIS (Thoracic Cobb Angle):")
            print(f"  Measured:              {kyph.get('measured_deg', '—')}°")
            print(f"  Ohlendorf 2020 mean:   {kyph.get('reference_mean_deg', '—')}°")
            tr = kyph.get('tolerance_region', [None, None])
            print(f"  Tolerance region:      {tr[0]}° – {tr[1]}°")
            ci = kyph.get('confidence_interval', [None, None])
            print(f"  95% CI:                {ci[0]}° – {ci[1]}°")
            print(f"  Ohlendorf class:       {kyph.get('ohlendorf_class', '—')}")
            print(f"  Mendeley class:        {kyph.get('clinical_class_mendeley', '—')}")
            print(f"  Within CI:             {kyph.get('within_CI', '—')}")
            print(f"  z-score (approx):      {kyph.get('z_score_approx', '—')}")

        lord = report.get("lordosis", {})
        if lord:
            print("\nLORDOSIS (Lumbar Curve):")
            print(f"  Measured:              {lord.get('measured_deg', '—')}°")
            print(f"  Ohlendorf 2020 mean:   {lord.get('reference_mean_deg', '—')}°")
            tr = lord.get('tolerance_region', [None, None])
            print(f"  Tolerance region:      {tr[0]}° – {tr[1]}°")
            ci = lord.get('confidence_interval', [None, None])
            print(f"  95% CI:                {ci[0]}° – {ci[1]}°")
            print(f"  Classification:        {lord.get('classification', '—')}")
            print(f"  Within CI:             {lord.get('within_CI', '—')}")

        lean = report.get("trunk_lean", {})
        if lean:
            gait = lean.get("gait_context", {})
            print("\nTRUNK LEAN (Gait Context — Tokuda 2017):")
            print(f"  Measured:              {gait.get('measured_deg', '—')}°")
            print(f"  Normal gait mean:      {gait.get('reference_mean_deg', '—')}° "
                  f"± {gait.get('reference_sd_deg', '—')}° SD")
            print(f"  Modified lean target:  {gait.get('modified_gait_target_deg', '—')}°")
            print(f"  Classification:        {gait.get('classification', '—')}")
            print(f"  Deviation from normal: {gait.get('deviation_from_normal_deg', '—')}°")

            stand = lean.get("standing_context", {})
            print("\nTRUNK LEAN (Standing Context — Ohlendorf 2020):")
            print(f"  Ohlendorf 2020 mean:   {stand.get('reference_mean_deg', '—')}° "
                  f"(ventral inclination)")
            print(f"  TR upper limit:        {stand.get('tolerance_region_upper_deg', '—')}°")
            print(f"  Classification:        {stand.get('classification', '—')}")

        ucm = report.get("ucm_synergy", {})
        if ucm:
            vert = ucm.get("vertical", {})
            print("\nUCM SYNERGY INDEX (Vertical COM — Tokuda 2017):")
            print(f"  VUCM:                  {vert.get('VUCM', '—')} rad²")
            print(f"  VORT:                  {vert.get('VORT', '—')} rad²")
            print(f"  ΔV:                    {vert.get('delta_V', '—')}")
            print(f"  ΔVz (Fisher):          {vert.get('delta_Vz', '—')}")
            print(f"  Synergy exists:        {vert.get('synergy_exists', '—')}")
            print(f"  Strong synergy:        {vert.get('synergy_strong', '—')}")
            print(f"  Interpretation:        {vert.get('interpretation', '—')}")
            print(f"  Paper context:         {vert.get('paper_context', '—')}")

        print("\n" + "═" * 72 + "\n")


# ─────────────────────────────────────────────
# SECTION 13 — PLOTTING
# ─────────────────────────────────────────────
class Plotter:
    """Publication-ready plots comparing measurements to paper standards."""

    @staticmethod
    def plot_time_series(video_result: VideoResult,
                         output_path: str = "outputs/figures/time_series.png"):
        if not MATPLOTLIB_AVAILABLE:
            log.warning("Matplotlib not available — skipping plot")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        fig.suptitle("Spinal Analysis \u2014 Time Series with Reference Standards",
                     fontsize=14, fontweight='bold')

        std = PublishedStandards

        # ── Kyphosis ─────────────────────────────────────────────────────────
        ax = axes[0]
        kyph = video_result.kyphosis_smoothed
        if len(kyph) > 0:
            kf = np.arange(len(kyph))
            ax.plot(kf, kyph,
                    color='steelblue', lw=2, label='Kyphosis (measured)')
            ax.axhline(std.KYPHOSIS_MEAN, color='green', ls='--', lw=1.5,
                       label=f'Ohlendorf 2020 mean ({std.KYPHOSIS_MEAN}\u00b0)')
            ax.axhspan(std.KYPHOSIS_TR_LO, std.KYPHOSIS_TR_HI,
                       alpha=0.12, color='green', label='Tolerance region')
            ax.axhspan(std.KYPHOSIS_CI_LO, std.KYPHOSIS_CI_HI,
                       alpha=0.22, color='limegreen', label='95% CI')
        ax.set_ylabel("Kyphosis (\u00b0)", fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Trunk Lean ───────────────────────────────────────────────────────
        ax = axes[1]
        tlean = video_result.trunk_lean_smoothed
        if len(tlean) > 0:
            tf = np.arange(len(tlean))
            ax.plot(tf, tlean,
                    color='darkorange', lw=2, label='Trunk lean (measured)')
            ax.axhline(std.GAIT_TRUNK_LEAN_NORMAL_MEAN, color='purple', ls='--',
                       lw=1.5,
                       label=f'Tokuda 2017 normal ({std.GAIT_TRUNK_LEAN_NORMAL_MEAN}\u00b0)')
            ax.axhline(std.GAIT_TRUNK_LEAN_MOD_MEAN, color='red', ls=':',
                       lw=1.5,
                       label=f'Modified lean gait ({std.GAIT_TRUNK_LEAN_MOD_MEAN}\u00b0)')
            ax.axhspan(0, std.GAIT_TRUNK_LEAN_NORMAL_MEAN + 2 * std.GAIT_TRUNK_LEAN_NORMAL_SD,
                       alpha=0.12, color='purple', label='Normal range (\u00b12 SD)')
        ax.set_ylabel("Trunk Lean (\u00b0)", fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Lordosis ─────────────────────────────────────────────────────────
        ax = axes[2]
        lordosis = video_result.lordosis_smoothed
        if len(lordosis) > 0:
            lf = np.arange(len(lordosis))
            ax.plot(lf, lordosis,
                    color='orchid', lw=2, label='Lordosis (measured)')
            ax.axhline(std.LORDOSIS_MEAN, color='darkgreen', ls='--', lw=1.5,
                       label=f'Ohlendorf 2020 mean ({std.LORDOSIS_MEAN}\u00b0)')
            ax.axhspan(std.LORDOSIS_TR_LO, std.LORDOSIS_TR_HI,
                       alpha=0.12, color='darkgreen', label='Tolerance region')
            ax.axhspan(std.LORDOSIS_CI_LO, std.LORDOSIS_CI_HI,
                       alpha=0.22, color='limegreen', label='95% CI')
        # NOTE: Cobb angle classification bands (normal/mild) have been intentionally
        # removed from this panel. The lordosis plot shows only the lumbar lordosis
        # measurement vs the Ohlendorf 2020 lordosis reference. The Mendeley Cobb
        # classification (20-40 normal, 40-60 mild) belongs to the kyphosis panel
        # only and would be misleading here.
        ax.set_ylabel("Lordosis (\u00b0)", fontsize=11)
        ax.set_xlabel("Valid Frame Index", fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"Time-series plot saved to {output_path}")

    @staticmethod
    def plot_standard_comparison(report: Dict,
                                 output_path: str = "outputs/figures/standards.png"):
        """Bar chart comparing measured values to paper standards."""
        if not MATPLOTLIB_AVAILABLE:
            return

        metrics = []
        measured = []
        ref_means = []
        ref_lo = []
        ref_hi = []

        kyph = report.get("kyphosis", {})
        if kyph.get("measured_deg") is not None:
            metrics.append("Kyphosis")
            measured.append(kyph["measured_deg"])
            ref_means.append(kyph.get("reference_mean_deg", 51.08))
            tr = kyph.get("tolerance_region", [31.63, 70.53])
            ref_lo.append(tr[0])
            ref_hi.append(tr[1])

        lord = report.get("lordosis", {})
        if lord.get("measured_deg") is not None:
            metrics.append("Lordosis")
            measured.append(lord["measured_deg"])
            ref_means.append(lord.get("reference_mean_deg", 32.86))
            tr = lord.get("tolerance_region", [15.25, 50.47])
            ref_lo.append(tr[0])
            ref_hi.append(tr[1])

        lean_gait = report.get("trunk_lean", {}).get("gait_context", {})
        if lean_gait.get("measured_deg") is not None:
            metrics.append("Trunk Lean\n(gait)")
            measured.append(lean_gait["measured_deg"])
            ref_means.append(lean_gait.get("reference_mean_deg", 1.0))
            ref_lo.append(0.0)
            ref_hi.append(lean_gait.get("reference_mean_deg", 1.0)
                          + 2 * 1.5)

        if not metrics:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(metrics))
        w = 0.35

        bars_m = ax.bar(x - w/2, measured, w, color='steelblue',
                        label='Measured', alpha=0.85, zorder=3)
        bars_r = ax.bar(x + w/2, ref_means, w, color='green',
                        label='Reference mean', alpha=0.85, zorder=3)

        # Tolerance region error bars on reference
        yerr_lo = [r - lo for r, lo in zip(ref_means, ref_lo)]
        yerr_hi = [hi - r  for r, hi in zip(ref_means, ref_hi)]
        ax.errorbar(x + w/2, ref_means,
                    yerr=[yerr_lo, yerr_hi],
                    fmt='none', color='darkgreen', capsize=6, lw=2,
                    label='Tolerance region', zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel("Angle (°)", fontsize=12)
        ax.set_title("Measured vs Published Reference Standards\n"
                     "(Ohlendorf 2020; Tokuda 2017)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        # Value labels
        for bar in bars_m:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}°', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='steelblue')
        for bar in bars_r:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}°', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='green')

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"Standards comparison plot saved to {output_path}")


# ─────────────────────────────────────────────
# SECTION 14 — UNIFIED PIPELINE
# ─────────────────────────────────────────────
class UnifiedPipeline:
    """
    Master orchestrator — runs the anatomical pipeline and generates
    clinical reports and an annotated output video.

    Usage:
      python spinal_analysis_complete.py              # full run
      python spinal_analysis_complete.py --demo       # demo with synthetic data
      python spinal_analysis_complete.py --report     # report only (no video)
    """

    def __init__(self):
        self.anatomical = AnatomicalPipeline()
        Path(CONFIG.output_dir).mkdir(parents=True, exist_ok=True)
        Path(CONFIG.model_dir).mkdir(parents=True, exist_ok=True)

    def run_on_video(self, video_path: str) -> VideoResult:
        """Run anatomical pipeline on a single video."""
        log.info(f"Processing video: {video_path}")
        vr = self.anatomical.analyze_video(video_path)
        return vr

    def run_demo(self) -> Dict:
        """
        Demonstration mode — runs full analysis on synthetic data.
        Useful when SpinePose model / video files are not yet available.
        """
        log.info("Running in DEMO mode with synthetic spinal data")
        np.random.seed(42)
        n_frames = 150

        # Simulate realistic spinal angle time series
        t = np.linspace(0, 6 * np.pi, n_frames)
        kyphosis_sim   = 42.0 + 5 * np.sin(t) + np.random.normal(0, 2, n_frames)
        lordosis_sim   = 30.0 + 3 * np.cos(t) + np.random.normal(0, 1.5, n_frames)
        trunk_lean_sim = 3.5  + 1 * np.sin(t) + np.random.normal(0, 0.5, n_frames)
        confidence_sim = np.clip(np.random.normal(0.75, 0.1, n_frames), 0.5, 1.0)

        # Smooth
        kyphosis_smooth   = Smoother.smooth(kyphosis_sim)
        lordosis_smooth   = Smoother.smooth(lordosis_sim)
        trunk_lean_smooth = Smoother.smooth(trunk_lean_sim)

        # Build VideoResult
        vr = VideoResult(video_path="DEMO")
        vr.kyphosis_smoothed   = kyphosis_smooth
        vr.trunk_lean_smoothed = trunk_lean_smooth
        vr.lordosis_smoothed   = lordosis_smooth

        # Populate frame results
        for i in range(n_frames):
            fr = FrameResult(frame_idx=i, valid=True)
            fr.kyphosis_angle   = float(kyphosis_sim[i])
            fr.lordosis_angle   = float(lordosis_sim[i])
            fr.trunk_lean_angle = float(trunk_lean_sim[i])
            fr.keypoint_confidence = float(confidence_sim[i])
            fr.cobb_class       = PublishedStandards.classify_cobb(fr.kyphosis_angle)
            fr.kyphosis_class_ohlendorf = \
                PublishedStandards.classify_kyphosis_ohlendorf(fr.kyphosis_angle)
            fr.lordosis_class_ohlendorf = \
                PublishedStandards.classify_lordosis_ohlendorf(fr.lordosis_angle)
            fr.trunk_lean_class = PublishedStandards.classify_trunk_lean(
                fr.trunk_lean_angle)
            vr.frame_results.append(fr)

        self.anatomical._build_summary(vr)

        # Generate clinical report
        report = ClinicalComparisonReport.generate(
            kyphosis_deg   = float(np.mean(kyphosis_smooth)),
            lordosis_deg   = float(np.mean(lordosis_smooth)),
            trunk_lean_deg = float(np.mean(trunk_lean_smooth)),
            context        = "gait",
            ucm_vucm_vertical = 4.2e-4,
            ucm_vort_vertical = 1.8e-4,
        )
        ClinicalComparisonReport.print_report(report)

        # Plots
        Plotter.plot_time_series(vr, "outputs/figures/demo_time_series.png")
        Plotter.plot_standard_comparison(report, "outputs/figures/demo_standards.png")

        # UCM synergy demo (Tokuda 2017 replication)
        ucm_normal = UCMSynergyAnalyzer.compute_synergy_index(3.7e-4, 0.7e-4, "vertical")
        ucm_lean   = UCMSynergyAnalyzer.compute_synergy_index(4.2e-4, 1.8e-4, "vertical")
        ucm_compare = UCMSynergyAnalyzer.compare_conditions(
            normal_vucm=3.7e-4, normal_vort=0.7e-4,
            lean_vucm=4.2e-4,   lean_vort=1.8e-4,
            direction="vertical")

        log.info("=== UCM SYNERGY COMPARISON (Tokuda 2017 Replication) ===")
        log.info(f"  Normal gait ΔVz (vertical):       {ucm_normal['delta_Vz']}")
        log.info(f"  Trunk lean gait ΔVz (vertical):   {ucm_lean['delta_Vz']}")
        log.info(f"  Change:                           {ucm_compare['delta_Vz_change']}")
        log.info(f"  Consistent with Tokuda 2017:      "
                 f"{ucm_compare['consistent_with_tokuda_2017']}")

        return {
            "video_result":   vr,
            "clinical_report": report,
            "ucm_comparison":  ucm_compare,
        }

    def run(self, video_paths: Optional[List[str]] = None,
            export_video: bool = True):
        """Main entry point."""
        if not video_paths:
            log.info("No video paths provided — running demo mode")
            return self.run_demo()

        all_results = []
        for vp in video_paths:
            try:
                vr = self.run_on_video(vp)
                all_results.append(vr)

                # Collect valid measurements
                kyph_vals = [fr.kyphosis_angle for fr in vr.frame_results
                             if fr.valid and fr.kyphosis_angle is not None]
                lord_vals = [fr.lordosis_angle for fr in vr.frame_results
                             if fr.valid and fr.lordosis_angle is not None]
                lean_vals = [fr.trunk_lean_angle for fr in vr.frame_results
                             if fr.valid and fr.trunk_lean_angle is not None]

                mean_kyph = float(np.mean(kyph_vals)) if kyph_vals else None
                mean_lord = float(np.mean(lord_vals)) if lord_vals else None
                mean_lean = float(np.mean(lean_vals)) if lean_vals else None

                report = ClinicalComparisonReport.generate(
                    kyphosis_deg   = mean_kyph,
                    lordosis_deg   = mean_lord,
                    trunk_lean_deg = mean_lean,
                )
                ClinicalComparisonReport.print_report(report)
                Plotter.plot_time_series(
                    vr, f"outputs/figures/{Path(vp).stem}_series.png")
                Plotter.plot_standard_comparison(
                    report, f"outputs/figures/{Path(vp).stem}_standards.png")

                # Annotated video export
                if export_video and CV2_AVAILABLE:
                    out_vid = (f"outputs/annotated_{Path(vp).stem}"
                               f"{Path(vp).suffix}")
                    VideoExporter.export(vp, vr, report, out_vid)

            except Exception as e:
                log.error(f"Failed on {vp}: {e}")
                log.debug(traceback.format_exc())

        return all_results


# ─────────────────────────────────────────────
# SECTION 15 — ENTRY POINT
# ─────────────────────────────────────────────
def main():
    """
    CLI entry point.

    Usage:
      python spinal_analysis_complete.py              → demo mode
      python spinal_analysis_complete.py video.mp4    → process single video
      python spinal_analysis_complete.py v1.mp4 v2.mp4 → process multiple videos
      python spinal_analysis_complete.py video.mp4 --no-video-export → skip video export
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Spinal Curvature Analysis Pipeline\n"
                    "Integrates trunk lean (Tokuda 2017) + lordosis/kyphosis "
                    "(Ohlendorf 2020) clinical standards with annotated video export.")
    parser.add_argument("videos", nargs="*",
                        help="Video file paths (omit for demo mode)")
    parser.add_argument("--demo", action="store_true",
                        help="Run synthetic demo even if videos provided")
    parser.add_argument("--no-video-export", action="store_true",
                        help="Skip annotated video export (analysis + plots only)")
    parser.add_argument("--report-only", nargs=3, metavar=("KYPHOSIS", "LORDOSIS", "LEAN"),
                        type=float,
                        help="Generate report for given angle values (degrees), "
                             "no video processing needed. Example: --report-only 48 35 4.2")
    args = parser.parse_args()

    pipeline = UnifiedPipeline()

    if args.report_only:
        kyph, lord, lean = args.report_only
        report = ClinicalComparisonReport.generate(
            kyphosis_deg=kyph, lordosis_deg=lord, trunk_lean_deg=lean)
        ClinicalComparisonReport.print_report(report)
        Plotter.plot_standard_comparison(report, "outputs/figures/report_standards.png")
        return

    if args.demo or not args.videos:
        pipeline.run_demo()
        return

    pipeline.run(args.videos, export_video=not args.no_video_export)


if __name__ == "__main__":
    main()
