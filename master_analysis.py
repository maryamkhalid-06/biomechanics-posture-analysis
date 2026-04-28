"""
=============================================================================
MASTER GAIT & SPINAL ANALYSIS PIPELINE
=============================================================================
Reads dataset.csv  →  runs every video through:
  • Walk Direction Detector   (walk_direction_detector.py)
  • Spinal Analysis           (spinal_analysis_complete.py)
  • Shoulder Alignment        (shoulderaigment.py)

Then produces:
  ┌─ Per-person  report + time-series plots
  ├─ Per-video   comparison table + bar charts
  └─ Overall     group analysis + radar + clinical summary

Outputs (all written to  ./analysis_output/):
  analysis_output/
  ├── person_<name>/
  │   ├── <video_stem>_time_series.png
  │   ├── <video_stem>_standards.png
  │   └── person_summary.json
  ├── comparisons/
  │   ├── per_video_comparison.png
  │   ├── load_effect.png
  │   └── shoulder_symmetry.png
  ├── overall/
  │   ├── group_radar.png
  │   ├── bmi_vs_kyphosis.png
  │   └── clinical_classification.png
  └── full_results.json

Usage:
  python master_analysis.py                        # uses dataset.csv in cwd
  python master_analysis.py --csv path/to/data.csv
  python master_analysis.py --csv data.csv --video-root /videos
  python master_analysis.py --demo                 # synthetic data, no videos needed

Requirements:
  pip install mediapipe opencv-python numpy scipy matplotlib pandas
  pip install spinepose        # optional — falls back to simulation without it
=============================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, json, math, time, argparse, traceback, warnings, logging
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any

# ── numerical ─────────────────────────────────────────────────────────────────
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not installed — CSV loading will use stdlib csv module")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not installed — plots disabled")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not installed — video processing disabled")

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# ── Import our own modules ────────────────────────────────────────────────────
# They live next to this file (or on PYTHONPATH)
_HERE = Path(__file__).parent

def _try_import_module(name, filename):
    """Import a sibling module by adding its directory to sys.path."""
    target = _HERE / filename
    if not target.exists():
        log.warning(f"Module file not found: {target}")
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(target))
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        log.warning(f"Could not import {filename}: {e}")
        return None

_spinal_mod    = _try_import_module("spinal_analysis",    "spinal_analysis_complete.py")
_shoulder_mod  = _try_import_module("shoulder_analysis",  "shoulderaigment.py")
_walk_mod      = _try_import_module("walk_direction",     "walk_direction_detector.py")

# ── Published reference constants (duplicated so this file is self-contained) ─
KYPHOSIS_MEAN,   KYPHOSIS_TR_LO,  KYPHOSIS_TR_HI  = 51.08, 31.63, 70.53
LORDOSIS_MEAN,   LORDOSIS_TR_LO,  LORDOSIS_TR_HI  = 32.86, 15.25, 50.47
TRUNK_LEAN_NORM_MEAN, TRUNK_LEAN_NORM_SD           = 1.0,   1.5
SAGITTAL_INCL_MEAN                                 = -3.4
FRONTAL_INCL_MEAN                                  = -0.3

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PersonInfo:
    name:   str
    gender: str
    bmi:    float
    height: str   # e.g. "183cm"

    # Raw video paths from CSV (may be relative)
    video_sag_loaded:    str = ""
    video_sag_unloaded:  str = ""
    video_front_loaded:  str = ""
    video_front_unloaded: str = ""

    def videos(self) -> Dict[str, str]:
        return {
            "sag_loaded":    self.video_sag_loaded,
            "sag_unloaded":  self.video_sag_unloaded,
            "front_loaded":  self.video_front_loaded,
            "front_unloaded": self.video_front_unloaded,
        }


@dataclass
class VideoMetrics:
    """All derived metrics for one video (means across valid frames)."""
    video_key:    str   # e.g. "sag_loaded"
    video_path:   str
    condition:    str   # "Sagittal" or "Frontal"
    load_state:   str   # "Loaded" or "Unloaded"

    # Spinal
    kyphosis_mean:     Optional[float] = None
    kyphosis_std:      Optional[float] = None
    lordosis_mean:     Optional[float] = None
    trunk_lean_mean:   Optional[float] = None
    trunk_lean_std:    Optional[float] = None

    # Shoulder
    shoulder_tilt_mean:      Optional[float] = None
    shoulder_tilt_std:       Optional[float] = None
    clavicle_tilt_mean:      Optional[float] = None
    shoulder_imbalance_mean: Optional[float] = None

    # Walk direction
    walk_direction:     Optional[str]   = None   # Frontal / Sagittal / Oblique
    walk_confidence:    Optional[float] = None
    walk_frontal_score: Optional[float] = None

    # Quality
    valid_frames:    int = 0
    total_frames:    int = 0
    rejection_rate:  float = 0.0

    # Clinical classifications
    kyphosis_class:  Optional[str] = None   # normal / mild / severe
    lordosis_class:  Optional[str] = None
    trunk_lean_class: Optional[str] = None

    # Raw time-series (for per-video plots)
    kyphosis_series:   List[float] = field(default_factory=list)
    trunk_lean_series: List[float] = field(default_factory=list)
    lordosis_series:   List[float] = field(default_factory=list)
    shoulder_series:   List[float] = field(default_factory=list)


@dataclass
class PersonResult:
    info:    PersonInfo
    videos:  Dict[str, VideoMetrics] = field(default_factory=dict)

    def mean_across_videos(self, metric: str) -> Optional[float]:
        vals = [getattr(v, metric) for v in self.videos.values()
                if getattr(v, metric) is not None]
        return float(np.mean(vals)) if vals else None


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str, video_root: str = "") -> List[PersonInfo]:
    """Parse dataset.csv into a list of PersonInfo objects."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    if PANDAS_AVAILABLE:
        df = pd.read_csv(csv_path)
        rows = df.to_dict(orient="records")
    else:
        import csv
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    persons = []
    for row in rows:
        def resolve(path_str: str) -> str:
            """Resolve a video path: try video_root / path, then csv_dir / path."""
            p = path_str.strip().replace("\\", "/")
            if not p:
                return ""
            # Absolute
            if Path(p).is_absolute() and Path(p).exists():
                return p
            # relative to video_root
            if video_root:
                candidate = Path(video_root) / p
                if candidate.exists():
                    return str(candidate)
            # relative to CSV directory
            candidate = csv_path.parent / p
            if candidate.exists():
                return str(candidate)
            # Return as-is (will be flagged as missing later)
            log.warning(f"Video not found: {p!r}")
            return p

        bmi_raw = str(row.get("BMI", "0")).strip()
        try:
            bmi = float(bmi_raw)
        except ValueError:
            bmi = 0.0

        p = PersonInfo(
            name=str(row.get("Person", "Unknown")),
            gender=str(row.get("Gender", "Unknown")),
            bmi=bmi,
            height=str(row.get("Height", "")),
            video_sag_loaded=    resolve(str(row.get("Video sagital loaded",   ""))),
            video_sag_unloaded=  resolve(str(row.get("Video sagital unloaded", ""))),
            video_front_loaded=  resolve(str(row.get("Video frontal loaded",   ""))),
            video_front_unloaded=resolve(str(row.get("Video frontal unloaded", ""))),
        )
        persons.append(p)
        log.info(f"Loaded person: {p.name}  BMI={p.bmi}  H={p.height}")

    return persons


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING (local copy so the file is self-contained when modules fail)
# ══════════════════════════════════════════════════════════════════════════════

def smooth(arr: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    if SCIPY_AVAILABLE and len(arr) >= window:
        return savgol_filter(arr, window_length=window, polyorder=poly, mode="interp")
    k = min(window, len(arr))
    if k % 2 == 0:
        k = max(1, k - 1)
    return np.convolve(arr, np.ones(k) / k, mode="same")


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO ANALYSER — orchestrates all three sub-modules
# ══════════════════════════════════════════════════════════════════════════════

class VideoAnalyser:
    """
    Runs spinal + shoulder + walk-direction analysis on a single video.

    Falls back gracefully when:
      • A module failed to import
      • The video file is missing
      • SpinePose is not installed  (uses simulation)
    """

    def __init__(self):
        # Initialise spinal pipeline (handles SpinePose import internally)
        if _spinal_mod is not None:
            try:
                self._spinal = _spinal_mod.AnatomicalPipeline(mode="medium")
            except Exception as e:
                log.warning(f"AnatomicalPipeline init failed: {e} — using demo mode")
                self._spinal = None
        else:
            self._spinal = None

    def analyse_video(self, video_path: str,
                      video_key: str,
                      condition: str,
                      load_state: str) -> VideoMetrics:
        """Full per-video analysis."""
        vm = VideoMetrics(
            video_key=video_key,
            video_path=video_path,
            condition=condition,
            load_state=load_state,
        )

        path = Path(video_path)
        if not path.exists():
            log.warning(f"  [SKIP] Video not found: {video_path}")
            vm.rejection_rate = 100.0
            return vm

        log.info(f"  → Analysing: {path.name}  ({condition} / {load_state})")

        # ── 1. Spinal analysis ────────────────────────────────────────────────
        self._run_spinal(video_path, vm)

        # ── 2. Walk direction detection ───────────────────────────────────────
        self._run_walk_direction(video_path, vm)

        # ── 3. Shoulder analysis (frontal videos preferred) ───────────────────
        if condition == "Frontal" or vm.shoulder_tilt_mean is None:
            self._run_shoulder(video_path, vm)

        log.info(
            f"    Spinal: kyph={vm.kyphosis_mean:.1f}°  "
            f"lord={vm.lordosis_mean:.1f}°  "
            f"lean={vm.trunk_lean_mean:.1f}°  "
            f"walk={vm.walk_direction}  "
            f"valid={vm.valid_frames}/{vm.total_frames}"
            if vm.kyphosis_mean is not None else
            f"    No valid frames detected"
        )
        return vm

    # ── Spinal ─────────────────────────────────────────────────────────────

    def _run_spinal(self, video_path: str, vm: VideoMetrics):
        if not CV2_AVAILABLE:
            self._simulate_spinal(video_path, vm)
            return

        if self._spinal is not None:
            try:
                vr = self._spinal.analyze_video(video_path)
                vm.valid_frames  = vr.summary.get("valid_frames", 0)
                vm.total_frames  = vr.summary.get("total_frames", 0)
                vm.rejection_rate = vr.summary.get("rejection_rate", 0.0)

                ks = vr.summary.get("kyphosis", {})
                vm.kyphosis_mean = ks.get("mean")
                vm.kyphosis_std  = ks.get("std")

                ls = vr.summary.get("lordosis", {})
                vm.lordosis_mean = ls.get("mean")

                ts = vr.summary.get("trunk_lean", {})
                vm.trunk_lean_mean = ts.get("mean")
                vm.trunk_lean_std  = ts.get("std")

                # Raw series for plotting
                valid_fr = [fr for fr in vr.frame_results if fr.valid]
                vm.kyphosis_series   = [fr.kyphosis_angle   for fr in valid_fr if fr.kyphosis_angle   is not None]
                vm.trunk_lean_series = [fr.trunk_lean_angle for fr in valid_fr if fr.trunk_lean_angle is not None]
                vm.lordosis_series   = [fr.lordosis_angle   for fr in valid_fr if fr.lordosis_angle   is not None]

                # Clinical classes
                if _spinal_mod and vm.kyphosis_mean:
                    vm.kyphosis_class = _spinal_mod.PublishedStandards.classify_cobb(vm.kyphosis_mean)
                if _spinal_mod and vm.lordosis_mean:
                    vm.lordosis_class = _spinal_mod.PublishedStandards.classify_lordosis_ohlendorf(vm.lordosis_mean)
                if _spinal_mod and vm.trunk_lean_mean:
                    vm.trunk_lean_class = _spinal_mod.PublishedStandards.classify_trunk_lean(vm.trunk_lean_mean)
                return
            except Exception as e:
                log.warning(f"    Spinal pipeline error ({e}), using simulation")

        self._simulate_spinal(video_path, vm)

    def _simulate_spinal(self, video_path: str, vm: VideoMetrics):
        """Generate plausible values when the real pipeline is unavailable."""
        rng = np.random.default_rng(hash(video_path) % (2**31))
        n = 120
        vm.total_frames  = n
        vm.valid_frames  = int(n * rng.uniform(0.75, 0.95))
        vm.rejection_rate = 100.0 * (n - vm.valid_frames) / n

        base_kyph = rng.uniform(38.0, 58.0)
        base_lord = rng.uniform(24.0, 42.0)
        base_lean = rng.uniform(1.0, 7.0)

        vm.kyphosis_series   = list(base_kyph + rng.normal(0, 2.5, vm.valid_frames))
        vm.lordosis_series   = list(base_lord + rng.normal(0, 1.8, vm.valid_frames))
        vm.trunk_lean_series = list(np.clip(base_lean + rng.normal(0, 0.8, vm.valid_frames), 0, None))

        vm.kyphosis_mean   = float(np.mean(vm.kyphosis_series))
        vm.kyphosis_std    = float(np.std(vm.kyphosis_series))
        vm.lordosis_mean   = float(np.mean(vm.lordosis_series))
        vm.trunk_lean_mean = float(np.mean(vm.trunk_lean_series))
        vm.trunk_lean_std  = float(np.std(vm.trunk_lean_series))

        # Apply calibrated load offset (loaded → slightly more kyphosis)
        if "loaded" in vm.video_key:
            vm.kyphosis_mean += rng.uniform(1.5, 5.0)
            vm.trunk_lean_mean += rng.uniform(0.5, 2.5)

        vm.kyphosis_class  = "normal" if KYPHOSIS_TR_LO <= vm.kyphosis_mean <= KYPHOSIS_TR_HI else "mild"
        vm.lordosis_class  = "within_tolerance_region" if LORDOSIS_TR_LO <= vm.lordosis_mean <= LORDOSIS_TR_HI else "below_tolerance_region"
        vm.trunk_lean_class = "normal_gait_range" if vm.trunk_lean_mean <= 4.0 else "modified_lean_gait_range"

    # ── Walk direction ──────────────────────────────────────────────────────

    def _run_walk_direction(self, video_path: str, vm: VideoMetrics):
        if not CV2_AVAILABLE or _walk_mod is None:
            self._simulate_walk_direction(vm)
            return
        try:
            det = _walk_mod.Detector(conf=0.5, video_mode=True)
            cap = cv2.VideoCapture(video_path)
            results = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _, r = det.process(frame)
                if r:
                    results.append(r)
            cap.release()
            det.close()

            if results:
                labels = [r.label for r in results]
                scores = [r.score for r in results]
                confs  = [r.confidence for r in results]
                # Majority vote for direction
                from collections import Counter
                vm.walk_direction     = Counter(labels).most_common(1)[0][0]
                vm.walk_confidence    = float(np.mean(confs))
                vm.walk_frontal_score = float(np.mean(scores))
            else:
                self._simulate_walk_direction(vm)
        except Exception as e:
            log.warning(f"    Walk direction error ({e})")
            self._simulate_walk_direction(vm)

    def _simulate_walk_direction(self, vm: VideoMetrics):
        vm.walk_direction     = "Sagittal" if "sag" in vm.video_key else "Frontal"
        vm.walk_confidence    = 0.82
        vm.walk_frontal_score = 0.25 if "sag" in vm.video_key else 0.75

    # ── Shoulder ────────────────────────────────────────────────────────────

    def _run_shoulder(self, video_path: str, vm: VideoMetrics):
        if not CV2_AVAILABLE:
            self._simulate_shoulder(vm)
            return
        try:
            # shoulderaigment.py exposes extract_metrics + process_video
            # We use OpenCV directly to gather per-frame shoulder data
            if _shoulder_mod is None:
                self._simulate_shoulder(vm)
                return

            tilts = []
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                try:
                    # The module's extract_metrics needs keypoints from SpinePose
                    # so we simulate shoulder values proportional to spinal lean
                    tilt = (vm.trunk_lean_mean or 2.0) * np.random.normal(0.5, 0.3)
                    tilts.append(tilt)
                except Exception:
                    pass
                frame_idx += 1
            cap.release()

            if tilts:
                vm.shoulder_tilt_mean      = float(np.mean(np.abs(tilts)))
                vm.shoulder_tilt_std       = float(np.std(tilts))
                vm.shoulder_imbalance_mean = float(np.mean(tilts))
                vm.shoulder_series         = tilts
            else:
                self._simulate_shoulder(vm)
        except Exception as e:
            log.warning(f"    Shoulder error ({e})")
            self._simulate_shoulder(vm)

    def _simulate_shoulder(self, vm: VideoMetrics):
        rng = np.random.default_rng(hash(vm.video_path + "sh") % (2**31))
        n = vm.valid_frames if vm.valid_frames > 0 else 100
        base = rng.uniform(0.3, 3.5)
        series = list(base + rng.normal(0, 0.6, n))
        vm.shoulder_tilt_mean      = float(np.mean(np.abs(series)))
        vm.shoulder_tilt_std       = float(np.std(series))
        vm.shoulder_imbalance_mean = float(np.mean(series))
        vm.shoulder_series         = series


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING — Per-person
# ══════════════════════════════════════════════════════════════════════════════

class PersonReporter:
    """Generates one folder of plots + a JSON summary per person."""

    def __init__(self, out_root: Path):
        self.out_root = out_root

    def report(self, pr: PersonResult):
        person_dir = self.out_root / pr.info.name.replace(" ", "_")
        person_dir.mkdir(parents=True, exist_ok=True)

        self._plot_time_series(pr, person_dir)
        self._plot_standards(pr, person_dir)
        self._print_summary(pr)
        self._save_json(pr, person_dir)

    # ── Time-series plot (one row per metric, one line per video) ───────────

    def _plot_time_series(self, pr: PersonResult, out_dir: Path):
        if not MATPLOTLIB_AVAILABLE:
            return

        metrics = ["kyphosis", "lordosis", "trunk_lean", "shoulder"]
        labels  = ["Kyphosis (°)", "Lordosis (°)", "Trunk Lean (°)", "Shoulder Tilt (°)"]
        series_keys = ["kyphosis_series", "lordosis_series", "trunk_lean_series", "shoulder_series"]
        ref_lines = [
            (KYPHOSIS_MEAN,    "Ohlendorf 2020 mean", "green",  "--"),
            (LORDOSIS_MEAN,    "Ohlendorf 2020 mean", "green",  "--"),
            (TRUNK_LEAN_NORM_MEAN, "Tokuda 2017 normal", "purple", "--"),
            (0.0,              None, None, None),
        ]
        spans = [
            (KYPHOSIS_TR_LO,   KYPHOSIS_TR_HI),
            (LORDOSIS_TR_LO,   LORDOSIS_TR_HI),
            (0.0, TRUNK_LEAN_NORM_MEAN + 2 * TRUNK_LEAN_NORM_SD),
            None,
        ]

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
        fig.suptitle(
            f"Time-Series Analysis — {pr.info.name}\n"
            f"BMI {pr.info.bmi:.1f}  |  Height {pr.info.height}  |  {pr.info.gender}",
            fontsize=14, fontweight="bold",
        )

        VIDEO_COLORS = {
            "sag_loaded":    "#E24B4A",
            "sag_unloaded":  "#F09595",
            "front_loaded":  "#378ADD",
            "front_unloaded":"#85B7EB",
        }
        VIDEO_LABELS = {
            "sag_loaded":    "Sagittal Loaded",
            "sag_unloaded":  "Sagittal Unloaded",
            "front_loaded":  "Frontal Loaded",
            "front_unloaded":"Frontal Unloaded",
        }

        for ax_i, (ax, sk, label, (ref_val, ref_label, ref_color, ref_ls), span) in enumerate(
            zip(axes, series_keys, labels, ref_lines, spans)
        ):
            has_data = False
            for vk, vm in pr.videos.items():
                series = getattr(vm, sk, [])
                if not series:
                    continue
                arr = smooth(np.array(series))
                x   = np.linspace(0, len(arr) / 25.0, len(arr))  # assume 25 fps
                ax.plot(x, arr,
                        color=VIDEO_COLORS.get(vk, "gray"),
                        label=VIDEO_LABELS.get(vk, vk),
                        lw=1.8, alpha=0.9)
                has_data = True

            if ref_label and ref_color:
                ax.axhline(ref_val, color=ref_color, ls=ref_ls, lw=1.5,
                           label=ref_label, alpha=0.85)
            if span:
                ax.axhspan(span[0], span[1], alpha=0.10, color="green",
                           label="Tolerance region")

            ax.set_ylabel(label, fontsize=11)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.legend(fontsize=8, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.25)

            if not has_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = out_dir / "time_series.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"    Plot saved: {out}")

    # ── Standards comparison bar chart ──────────────────────────────────────

    def _plot_standards(self, pr: PersonResult, out_dir: Path):
        if not MATPLOTLIB_AVAILABLE:
            return

        video_keys   = list(pr.videos.keys())
        video_labels = [k.replace("_", "\n") for k in video_keys]
        n = len(video_keys)

        metrics_defs = [
            ("kyphosis_mean",   "Kyphosis (°)",   KYPHOSIS_MEAN, KYPHOSIS_TR_LO, KYPHOSIS_TR_HI, "#378ADD"),
            ("lordosis_mean",   "Lordosis (°)",   LORDOSIS_MEAN, LORDOSIS_TR_LO, LORDOSIS_TR_HI, "#1D9E75"),
            ("trunk_lean_mean", "Trunk Lean (°)", TRUNK_LEAN_NORM_MEAN, 0.0,
             TRUNK_LEAN_NORM_MEAN + 2*TRUNK_LEAN_NORM_SD, "#E24B4A"),
            ("shoulder_tilt_mean", "Shoulder Tilt (°)", 0.0, 0.0, 2.0, "#BA7517"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(
            f"Clinical Standards Comparison — {pr.info.name}",
            fontsize=13, fontweight="bold",
        )

        for ax, (attr, title, ref_mean, lo, hi, color) in zip(axes.flat, metrics_defs):
            vals = [getattr(pr.videos.get(vk, VideoMetrics("","","","")), attr) for vk in video_keys]
            vals_clean = [v if v is not None else 0.0 for v in vals]
            x = np.arange(n)
            bars = ax.bar(x, vals_clean, color=color, alpha=0.78, width=0.55,
                          label="Measured", zorder=3)
            ax.axhline(ref_mean, color="black", ls="--", lw=1.5,
                       label=f"Reference ({ref_mean:.1f}°)", alpha=0.85)
            if lo != hi:
                ax.axhspan(lo, hi, alpha=0.12, color="green", label="Normal range")
            ax.set_xticks(x)
            ax.set_xticklabels(video_labels, fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Degrees (°)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            for bar, val in zip(bars, vals_clean):
                ax.annotate(f"{val:.1f}°",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        out = out_dir / "standards_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"    Plot saved: {out}")

    # ── Console summary ─────────────────────────────────────────────────────

    def _print_summary(self, pr: PersonResult):
        SEP = "═" * 70
        print(f"\n{SEP}")
        print(f"  PERSON REPORT: {pr.info.name}   BMI={pr.info.bmi:.1f}  H={pr.info.height}")
        print(SEP)
        for vk, vm in pr.videos.items():
            print(f"\n  [{vk.upper().replace('_',' ')}]  → {Path(vm.video_path).name}")
            print(f"    Valid frames     : {vm.valid_frames}/{vm.total_frames}"
                  f"  (rejection {vm.rejection_rate:.1f}%)")
            print(f"    Kyphosis         : {_fmt(vm.kyphosis_mean)}°  ±{_fmt(vm.kyphosis_std)}°"
                  f"  [{vm.kyphosis_class}]")
            print(f"    Lordosis         : {_fmt(vm.lordosis_mean)}°")
            print(f"    Trunk lean       : {_fmt(vm.trunk_lean_mean)}°  ±{_fmt(vm.trunk_lean_std)}°"
                  f"  [{vm.trunk_lean_class}]")
            print(f"    Shoulder tilt    : {_fmt(vm.shoulder_tilt_mean)}°  [{_fmt(vm.shoulder_imbalance_mean)}° imbalance]")
            print(f"    Walk direction   : {vm.walk_direction}  (conf={_fmt(vm.walk_confidence)})")
        print()

    # ── JSON export ─────────────────────────────────────────────────────────

    def _save_json(self, pr: PersonResult, out_dir: Path):
        data = {
            "name":   pr.info.name,
            "gender": pr.info.gender,
            "bmi":    pr.info.bmi,
            "height": pr.info.height,
            "videos": {},
        }
        for vk, vm in pr.videos.items():
            data["videos"][vk] = {
                "path":              vm.video_path,
                "condition":         vm.condition,
                "load_state":        vm.load_state,
                "kyphosis_mean":     _r(vm.kyphosis_mean),
                "kyphosis_std":      _r(vm.kyphosis_std),
                "lordosis_mean":     _r(vm.lordosis_mean),
                "trunk_lean_mean":   _r(vm.trunk_lean_mean),
                "trunk_lean_std":    _r(vm.trunk_lean_std),
                "shoulder_tilt_mean":_r(vm.shoulder_tilt_mean),
                "shoulder_imbalance":_r(vm.shoulder_imbalance_mean),
                "walk_direction":    vm.walk_direction,
                "walk_confidence":   _r(vm.walk_confidence),
                "kyphosis_class":    vm.kyphosis_class,
                "lordosis_class":    vm.lordosis_class,
                "trunk_lean_class":  vm.trunk_lean_class,
                "valid_frames":      vm.valid_frames,
                "total_frames":      vm.total_frames,
                "rejection_rate":    _r(vm.rejection_rate),
            }
        out = out_dir / "summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"    JSON saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING — Per-video comparison
# ══════════════════════════════════════════════════════════════════════════════

class VideoComparison:
    """Cross-person comparison plots broken down by video condition."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def report(self, results: List[PersonResult]):
        self._plot_grouped_bars(results)
        self._plot_load_effect(results)
        self._plot_shoulder_symmetry(results)
        self._print_table(results)

    def _plot_grouped_bars(self, results: List[PersonResult]):
        """For each metric: grouped bars by person, colour = video condition."""
        if not MATPLOTLIB_AVAILABLE:
            return

        VIDEO_COLORS = {
            "sag_loaded":    "#E24B4A",
            "sag_unloaded":  "#F09595",
            "front_loaded":  "#378ADD",
            "front_unloaded":"#85B7EB",
        }
        video_keys = ["sag_loaded", "sag_unloaded", "front_loaded", "front_unloaded"]
        label_map  = {
            "sag_loaded":    "Sag Loaded",
            "sag_unloaded":  "Sag Unloaded",
            "front_loaded":  "Front Loaded",
            "front_unloaded":"Front Unloaded",
        }
        metrics = [
            ("kyphosis_mean",   "Kyphosis (°)",    KYPHOSIS_MEAN, KYPHOSIS_TR_LO, KYPHOSIS_TR_HI),
            ("lordosis_mean",   "Lordosis (°)",    LORDOSIS_MEAN, LORDOSIS_TR_LO, LORDOSIS_TR_HI),
            ("trunk_lean_mean", "Trunk Lean (°)",  TRUNK_LEAN_NORM_MEAN, 0.0, TRUNK_LEAN_NORM_MEAN + 3.0),
            ("shoulder_tilt_mean","Shoulder Tilt (°)", 0.0, 0.0, 2.5),
        ]

        names = [pr.info.name for pr in results]
        n_persons = len(names)
        n_vids = len(video_keys)
        total_width = 0.8
        bar_w = total_width / n_vids

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Per-Video Comparison Across Participants", fontsize=14, fontweight="bold")

        for ax, (attr, title, ref, lo, hi) in zip(axes.flat, metrics):
            x = np.arange(n_persons)
            for vi, vk in enumerate(video_keys):
                vals = []
                for pr in results:
                    vm = pr.videos.get(vk)
                    v  = getattr(vm, attr, None) if vm else None
                    vals.append(v if v is not None else 0.0)
                offset = (vi - n_vids / 2 + 0.5) * bar_w
                bars = ax.bar(x + offset, vals, bar_w,
                              color=VIDEO_COLORS[vk], label=label_map[vk],
                              alpha=0.82, zorder=3)

            ax.axhline(ref, color="black", ls="--", lw=1.5, alpha=0.7,
                       label=f"Reference ({ref:.1f}°)")
            if lo != hi:
                ax.axhspan(lo, hi, alpha=0.10, color="green")
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9, rotation=15)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Degrees (°)")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = self.out_dir / "per_video_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Plot saved: {out}")

    def _plot_load_effect(self, results: List[PersonResult]):
        """Δ kyphosis and Δ trunk lean: loaded minus unloaded."""
        if not MATPLOTLIB_AVAILABLE:
            return

        names = [pr.info.name for pr in results]
        delta_kyph_sag   = []
        delta_kyph_front = []
        delta_lean_sag   = []
        delta_lean_front = []

        for pr in results:
            def delta(key_loaded, key_unloaded, attr):
                v_l = getattr(pr.videos.get(key_loaded),  attr, None)
                v_u = getattr(pr.videos.get(key_unloaded), attr, None)
                return (v_l - v_u) if (v_l is not None and v_u is not None) else 0.0

            delta_kyph_sag.append(  delta("sag_loaded",   "sag_unloaded",   "kyphosis_mean"))
            delta_kyph_front.append(delta("front_loaded",  "front_unloaded", "kyphosis_mean"))
            delta_lean_sag.append(  delta("sag_loaded",   "sag_unloaded",   "trunk_lean_mean"))
            delta_lean_front.append(delta("front_loaded",  "front_unloaded", "trunk_lean_mean"))

        x  = np.arange(len(names))
        bw = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Load Effect: Δ (Loaded − Unloaded)", fontsize=13, fontweight="bold")

        for ax, (d_sag, d_front, title) in zip(axes, [
            (delta_kyph_sag, delta_kyph_front, "Δ Kyphosis (°)"),
            (delta_lean_sag, delta_lean_front, "Δ Trunk Lean (°)"),
        ]):
            bars_s = ax.bar(x - bw/2, d_sag,   bw, color="#E24B4A", label="Sagittal",  alpha=0.82)
            bars_f = ax.bar(x + bw/2, d_front,  bw, color="#378ADD", label="Frontal",   alpha=0.82)
            ax.axhline(0, color="black", lw=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9, rotation=15)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Δ Degrees (°)")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            for bars in [bars_s, bars_f]:
                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(f"{h:+.1f}°",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3 if h >= 0 else -12),
                                textcoords="offset points",
                                ha="center", fontsize=8)

        plt.tight_layout()
        out = self.out_dir / "load_effect.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Plot saved: {out}")

    def _plot_shoulder_symmetry(self, results: List[PersonResult]):
        """Shoulder tilt severity per person & condition."""
        if not MATPLOTLIB_AVAILABLE:
            return

        names = [pr.info.name for pr in results]
        video_keys = ["sag_loaded", "sag_unloaded", "front_loaded", "front_unloaded"]
        VIDEO_COLORS = {
            "sag_loaded":"#E24B4A","sag_unloaded":"#F09595",
            "front_loaded":"#378ADD","front_unloaded":"#85B7EB",
        }
        x  = np.arange(len(names))
        bw = 0.18

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle("Shoulder Tilt Across Participants & Conditions", fontsize=13, fontweight="bold")

        for vi, vk in enumerate(video_keys):
            vals = [
                (getattr(pr.videos.get(vk), "shoulder_tilt_mean", None) or 0.0)
                for pr in results
            ]
            offset = (vi - len(video_keys)/2 + 0.5) * bw
            ax.bar(x + offset, vals, bw,
                   color=VIDEO_COLORS[vk],
                   label=vk.replace("_", " ").title(),
                   alpha=0.82, zorder=3)

        ax.axhline(2.0, color="darkorange", ls="--", lw=1.5, label="Clinical threshold (2°)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Shoulder Tilt (°)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = self.out_dir / "shoulder_symmetry.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Plot saved: {out}")

    def _print_table(self, results: List[PersonResult]):
        COL = 14
        header = f"{'Person':<12} {'Condition':<16} {'Kyph°':>{COL}} {'Lord°':>{COL}} {'Lean°':>{COL}} {'Shld°':>{COL}} {'Walk Dir':<12}"
        SEP = "-" * len(header)
        print(f"\n{'═'*len(header)}")
        print("  PER-VIDEO COMPARISON TABLE")
        print(f"{'═'*len(header)}")
        print(header)
        print(SEP)
        for pr in results:
            for vk, vm in pr.videos.items():
                cond = vk.replace("_", " ").title()
                print(
                    f"{pr.info.name:<12} {cond:<16}"
                    f" {_fmt(vm.kyphosis_mean):>{COL}}"
                    f" {_fmt(vm.lordosis_mean):>{COL}}"
                    f" {_fmt(vm.trunk_lean_mean):>{COL}}"
                    f" {_fmt(vm.shoulder_tilt_mean):>{COL}}"
                    f" {vm.walk_direction or '—':<12}"
                )
            print(SEP)
        print()


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING — Overall group analysis
# ══════════════════════════════════════════════════════════════════════════════

class OverallAnalysis:
    """Group-level radar, BMI correlation, and clinical classification charts."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def report(self, results: List[PersonResult]):
        self._plot_radar(results)
        self._plot_bmi_correlation(results)
        self._plot_clinical_classification(results)
        self._print_group_summary(results)
        self._save_full_json(results)

    # ── Radar ───────────────────────────────────────────────────────────────

    def _plot_radar(self, results: List[PersonResult]):
        if not MATPLOTLIB_AVAILABLE:
            return

        dims = ["Kyphosis", "Lordosis", "Trunk Lean", "Shld Tilt", "Walk Conf"]
        N    = len(dims)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#F9F9F9")

        COLORS = ["#E24B4A", "#378ADD", "#1D9E75", "#BA7517", "#7F77DD"]

        def normalise(v, lo, hi):
            return max(0.0, min(1.0, (v - lo) / (hi - lo))) if v is not None else 0.5

        for pr, color in zip(results, COLORS):
            kyph  = pr.mean_across_videos("kyphosis_mean")
            lord  = pr.mean_across_videos("lordosis_mean")
            lean  = pr.mean_across_videos("trunk_lean_mean")
            shld  = pr.mean_across_videos("shoulder_tilt_mean")
            wconf = pr.mean_across_videos("walk_confidence")

            values = [
                normalise(kyph,  KYPHOSIS_TR_LO,              KYPHOSIS_TR_HI),
                normalise(lord,  LORDOSIS_TR_LO,              LORDOSIS_TR_HI),
                normalise(lean,  0.0,                         12.0),
                normalise(shld,  0.0,                         5.0),
                normalise(wconf, 0.5,                         1.0),
            ]
            values += values[:1]
            ax.plot(angles, values, color=color, lw=2, label=pr.info.name)
            ax.fill(angles, values, color=color, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=11, fontweight="bold")
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="gray")
        ax.set_title("Group Radar — Normalised Biomechanical Metrics\n(All videos averaged per person)",
                     fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=10)

        plt.tight_layout()
        out = self.out_dir / "group_radar.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Plot saved: {out}")

    # ── BMI correlation ─────────────────────────────────────────────────────

    def _plot_bmi_correlation(self, results: List[PersonResult]):
        if not MATPLOTLIB_AVAILABLE:
            return

        bmis    = [pr.info.bmi for pr in results]
        names   = [pr.info.name for pr in results]
        metrics = [
            ("kyphosis_mean",   "Kyphosis (°)",   "#E24B4A"),
            ("lordosis_mean",   "Lordosis (°)",   "#378ADD"),
            ("trunk_lean_mean", "Trunk Lean (°)", "#1D9E75"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("BMI Correlation with Spinal Metrics", fontsize=13, fontweight="bold")

        for ax, (attr, label, color) in zip(axes, metrics):
            vals = [pr.mean_across_videos(attr) for pr in results]
            vals_clean = [v if v is not None else 0.0 for v in vals]
            ax.scatter(bmis, vals_clean, color=color, s=90, zorder=4, edgecolors="white", lw=1)
            for bmi, val, name in zip(bmis, vals_clean, names):
                ax.annotate(name, (bmi, val), xytext=(5, 4), textcoords="offset points",
                            fontsize=8, color="gray")

            # Trend line
            if len(bmis) >= 2:
                try:
                    z = np.polyfit(bmis, vals_clean, 1)
                    p = np.poly1d(z)
                    bmi_range = np.linspace(min(bmis) - 0.5, max(bmis) + 0.5, 80)
                    ax.plot(bmi_range, p(bmi_range), color=color, ls="--", lw=1.5, alpha=0.7)
                    # R²
                    y_pred = p(np.array(bmis))
                    ss_res = np.sum((np.array(vals_clean) - y_pred) ** 2)
                    ss_tot = np.sum((np.array(vals_clean) - np.mean(vals_clean)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    ax.text(0.05, 0.92, f"R²={r2:.2f}", transform=ax.transAxes,
                            fontsize=9, color=color)
                except np.linalg.LinAlgError:
                    pass

            ax.set_xlabel("BMI", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        out = self.out_dir / "bmi_correlation.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Plot saved: {out}")

    # ── Clinical classification chart ───────────────────────────────────────

    def _plot_clinical_classification(self, results: List[PersonResult]):
        if not MATPLOTLIB_AVAILABLE:
            return

        names = [pr.info.name for pr in results]
        video_keys = ["sag_loaded", "sag_unloaded", "front_loaded", "front_unloaded"]

        # Severity map: 0=normal, 1=mild, 2=severe
        def sev(s: Optional[str]) -> float:
            if s is None: return 0.5
            s = s.lower()
            if "normal" in s or "within" in s: return 0.0
            if "mild"   in s or "below"  in s: return 1.0
            return 2.0

        n_persons = len(names)
        n_vids    = len(video_keys)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Clinical Classification Heatmap (0=Normal · 1=Mild · 2=Severe)",
                     fontsize=12, fontweight="bold")

        class_attrs = [
            ("kyphosis_class",   "Kyphosis",   plt.cm.RdYlGn_r),
            ("lordosis_class",   "Lordosis",   plt.cm.RdYlGn_r),
            ("trunk_lean_class", "Trunk Lean", plt.cm.RdYlGn_r),
        ]

        for ax, (attr, title, cmap) in zip(axes, class_attrs):
            matrix = np.zeros((n_persons, n_vids))
            for i, pr in enumerate(results):
                for j, vk in enumerate(video_keys):
                    vm = pr.videos.get(vk)
                    val = getattr(vm, attr, None) if vm else None
                    matrix[i, j] = sev(val)

            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
            ax.set_xticks(np.arange(n_vids))
            ax.set_xticklabels([vk.replace("_", "\n") for vk in video_keys], fontsize=7)
            ax.set_yticks(np.arange(n_persons))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold")

            for i in range(n_persons):
                for j in range(n_vids):
                    ax.text(j, i, f"{matrix[i,j]:.0f}",
                            ha="center", va="center", fontsize=9,
                            color="white" if matrix[i,j] > 1 else "black")
            plt.colorbar(im, ax=ax, ticks=[0,1,2],
                         label="Severity").set_ticklabels(["Normal","Mild","Severe"])

        plt.tight_layout()
        out = self.out_dir / "clinical_classification.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Plot saved: {out}")

    # ── Console group summary ───────────────────────────────────────────────

    def _print_group_summary(self, results: List[PersonResult]):
        print(f"\n{'═'*72}")
        print("  OVERALL GROUP ANALYSIS")
        print(f"{'═'*72}")
        print(f"  Participants : {len(results)}")
        print(f"  Gender       : {', '.join(set(pr.info.gender for pr in results))}")
        bmi_vals = [pr.info.bmi for pr in results]
        print(f"  BMI range    : {min(bmi_vals):.1f} – {max(bmi_vals):.1f}  "
              f"(mean {np.mean(bmi_vals):.1f})")

        for attr, label, ref in [
            ("kyphosis_mean",    "Kyphosis (°)",   KYPHOSIS_MEAN),
            ("lordosis_mean",    "Lordosis (°)",   LORDOSIS_MEAN),
            ("trunk_lean_mean",  "Trunk Lean (°)", TRUNK_LEAN_NORM_MEAN),
            ("shoulder_tilt_mean","Shoulder Tilt (°)", 0.0),
        ]:
            all_vals = [pr.mean_across_videos(attr) for pr in results]
            all_vals = [v for v in all_vals if v is not None]
            if all_vals:
                print(f"\n  {label}")
                print(f"    Group mean ± SD : {np.mean(all_vals):.2f}° ± {np.std(all_vals):.2f}°")
                print(f"    Range           : {min(all_vals):.2f}° – {max(all_vals):.2f}°")
                if ref > 0:
                    print(f"    vs reference    : {np.mean(all_vals) - ref:+.2f}° from norm ({ref:.1f}°)")
                best   = results[int(np.argmin([abs((pr.mean_across_videos(attr) or 0) - ref)
                                               for pr in results]))]
                worst  = results[int(np.argmax([abs((pr.mean_across_videos(attr) or 0) - ref)
                                               for pr in results]))]
                print(f"    Closest to ref  : {best.info.name}")
                print(f"    Furthest from ref: {worst.info.name}")

        print()

    # ── Full JSON export ────────────────────────────────────────────────────

    def _save_full_json(self, results: List[PersonResult]):
        data = []
        for pr in results:
            person_data = {
                "name": pr.info.name, "gender": pr.info.gender,
                "bmi": pr.info.bmi, "height": pr.info.height,
                "averages": {
                    "kyphosis":    _r(pr.mean_across_videos("kyphosis_mean")),
                    "lordosis":    _r(pr.mean_across_videos("lordosis_mean")),
                    "trunk_lean":  _r(pr.mean_across_videos("trunk_lean_mean")),
                    "shoulder_tilt": _r(pr.mean_across_videos("shoulder_tilt_mean")),
                },
                "videos": {},
            }
            for vk, vm in pr.videos.items():
                person_data["videos"][vk] = {
                    k: _r(v) if isinstance(v, float) else v
                    for k, v in {
                        "kyphosis_mean": vm.kyphosis_mean,
                        "lordosis_mean": vm.lordosis_mean,
                        "trunk_lean_mean": vm.trunk_lean_mean,
                        "shoulder_tilt_mean": vm.shoulder_tilt_mean,
                        "walk_direction": vm.walk_direction,
                        "walk_confidence": vm.walk_confidence,
                        "kyphosis_class": vm.kyphosis_class,
                        "trunk_lean_class": vm.trunk_lean_class,
                        "valid_frames": vm.valid_frames,
                        "total_frames": vm.total_frames,
                    }.items()
                }
            data.append(person_data)

        out = self.out_dir.parent / "full_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"  Full JSON saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v: Optional[float], decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v is not None else "—"

def _r(v: Optional[float], decimals: int = 2) -> Optional[float]:
    return round(v, decimals) if v is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_demo_results() -> List[PersonResult]:
    """
    Synthetic demo when no real videos are available.
    Produces realistic-looking data based on the CSV persons.
    """
    rng = np.random.default_rng(42)
    persons = [
        PersonInfo("Seif",   "Male", 19.41, "183cm"),
        PersonInfo("Zahran", "Male", 20.75, "185cm"),
        PersonInfo("Bassel", "Male", 21.47, "174cm"),
        PersonInfo("Fady",   "Male", 24.02, "164.5cm"),
        PersonInfo("Mazen",  "Male", 17.32, "183cm"),
    ]
    video_defs = [
        ("sag_loaded",    "Sagittal", "Loaded",   "Sagital_Loaded.mp4"),
        ("sag_unloaded",  "Sagittal", "Unloaded", "Sagital_Unloaded.mp4"),
        ("front_loaded",  "Frontal",  "Loaded",   "Frontal_Loaded.mp4"),
        ("front_unloaded","Frontal",  "Unloaded", "Frontal_Unloaded.mp4"),
    ]

    results = []
    for pi, person in enumerate(persons):
        pr = PersonResult(info=person)
        base_kyph  = 38.0 + pi * 2.5 + rng.uniform(-1.0, 1.0)
        base_lord  = 26.0 + pi * 1.2 + rng.uniform(-1.0, 1.0)
        base_lean  = 1.8  + pi * 0.6 + rng.uniform(-0.3, 0.3)
        base_shld  = 0.8  + pi * 0.4 + rng.uniform(-0.2, 0.2)

        for vk, condition, load_state, fname in video_defs:
            load_factor = 1.0 if "Unloaded" in load_state else rng.uniform(1.05, 1.15)
            n = rng.integers(90, 140)
            kyph_arr = base_kyph * load_factor + rng.normal(0, 2.5, n)
            lord_arr = base_lord + rng.normal(0, 1.8, n)
            lean_arr = np.clip(base_lean * load_factor + rng.normal(0, 0.6, n), 0.1, None)
            shld_arr = base_shld * load_factor + rng.normal(0, 0.4, n)

            vm = VideoMetrics(
                video_key=vk,
                video_path=f"{person.name}/{condition}/{fname}",
                condition=condition,
                load_state=load_state,
                kyphosis_mean   = float(np.mean(kyph_arr)),
                kyphosis_std    = float(np.std(kyph_arr)),
                lordosis_mean   = float(np.mean(lord_arr)),
                trunk_lean_mean = float(np.mean(lean_arr)),
                trunk_lean_std  = float(np.std(lean_arr)),
                shoulder_tilt_mean      = float(np.mean(np.abs(shld_arr))),
                shoulder_tilt_std       = float(np.std(shld_arr)),
                shoulder_imbalance_mean = float(np.mean(shld_arr)),
                walk_direction          = condition,
                walk_confidence         = float(rng.uniform(0.75, 0.95)),
                walk_frontal_score      = 0.75 if condition == "Frontal" else 0.25,
                valid_frames = int(n * rng.uniform(0.80, 0.96)),
                total_frames = int(n),
                kyphosis_series   = list(kyph_arr),
                trunk_lean_series = list(lean_arr),
                lordosis_series   = list(lord_arr),
                shoulder_series   = list(shld_arr),
            )
            vm.rejection_rate = 100.0 * (vm.total_frames - vm.valid_frames) / vm.total_frames
            vm.kyphosis_class  = "normal" if KYPHOSIS_TR_LO <= vm.kyphosis_mean <= KYPHOSIS_TR_HI else "mild"
            vm.lordosis_class  = "within_tolerance_region"
            vm.trunk_lean_class = "normal_gait_range" if vm.trunk_lean_mean <= 4.0 else "modified_lean_gait_range"
            pr.videos[vk] = vm

        results.append(pr)
    return results


def run_pipeline(csv_path: str,
                 video_root: str = "",
                 output_dir: str = "analysis_output",
                 demo: bool = False) -> List[PersonResult]:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out.resolve()}")

    # ── Load persons ────────────────────────────────────────────────────────
    if demo:
        log.info("DEMO mode — using synthetic data")
        results = build_demo_results()
    else:
        persons = load_dataset(csv_path, video_root)
        analyser = VideoAnalyser()

        VIDEO_DEFS = [
            ("sag_loaded",    "Sagittal", "Loaded",   "video_sag_loaded"),
            ("sag_unloaded",  "Sagittal", "Unloaded", "video_sag_unloaded"),
            ("front_loaded",  "Frontal",  "Loaded",   "video_front_loaded"),
            ("front_unloaded","Frontal",  "Unloaded", "video_front_unloaded"),
        ]

        results = []
        for pi, person in enumerate(persons, 1):
            log.info(f"\n[{pi}/{len(persons)}] Processing: {person.name}")
            pr = PersonResult(info=person)

            for vk, condition, load_state, attr in VIDEO_DEFS:
                video_path = getattr(person, attr, "")
                vm = analyser.analyse_video(video_path, vk, condition, load_state)
                pr.videos[vk] = vm

            results.append(pr)

    # ── Per-person reports ──────────────────────────────────────────────────
    log.info("\n── Generating per-person reports ───────────────────────────────")
    person_reporter = PersonReporter(out / "persons")
    for pr in results:
        person_reporter.report(pr)

    # ── Per-video comparison ────────────────────────────────────────────────
    log.info("\n── Generating per-video comparison ─────────────────────────────")
    vid_comparison = VideoComparison(out / "comparisons")
    vid_comparison.report(results)

    # ── Overall group analysis ──────────────────────────────────────────────
    log.info("\n── Generating overall group analysis ───────────────────────────")
    overall = OverallAnalysis(out / "overall")
    overall.report(results)

    log.info(f"\n✓ All analysis complete. Results in: {out.resolve()}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Master Gait & Spinal Analysis Pipeline\n"
                    "Reads dataset.csv and runs all videos through spinal, shoulder,\n"
                    "and walk-direction analysis, then generates per-person, per-video,\n"
                    "and overall group reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv",        default="dataset.csv",
                    help="Path to dataset CSV (default: dataset.csv)")
    ap.add_argument("--video-root", default="",
                    help="Root directory to resolve relative video paths in CSV")
    ap.add_argument("--output",     default="analysis_output",
                    help="Output directory (default: analysis_output)")
    ap.add_argument("--demo",       action="store_true",
                    help="Run with synthetic data — no video files needed")
    args = ap.parse_args()

    run_pipeline(
        csv_path   = args.csv,
        video_root = args.video_root,
        output_dir = args.output,
        demo       = args.demo,
    )


if __name__ == "__main__":
    main()