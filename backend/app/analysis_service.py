from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shoulderaigment as shoulder_module
import spinal_analysis_complete as spinal_module
import walk_direction_detector as walk_module

STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT_DIR", ROOT / "backend_storage"))
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
RESULTS_ROOT = STORAGE_ROOT / "results"

for folder in (UPLOAD_ROOT, RESULTS_ROOT):
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_SHOULDER_VIEWS = {"front", "back"}
ALLOWED_POSE_PLANES = {"frontal", "sagittal", "oblique"}
ALLOWED_MODEL_SIZES = {"small", "medium", "large", "xlarge"}
DETECTED_PLANE_CONFIDENCE = 0.55
DEFAULT_MODEL_SIZE = "medium"
LIVE_MAX_WIDTH = 720


class InputValidationError(Exception):
    def __init__(self, errors: dict[str, str]):
        super().__init__("Invalid request")
        self.errors = errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _mean(values: list[float | None]) -> float | None:
    cleaned = [float(v) for v in values if _is_number(v)]
    if not cleaned:
        return None
    return round(float(sum(cleaned) / len(cleaned)), 3)


def _slug(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def _to_url(path: Path) -> str:
    rel = path.resolve().relative_to(STORAGE_ROOT.resolve())
    return f"/files/{rel.as_posix()}"


def _encode_frame(frame: np.ndarray) -> str | None:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_frame(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Expected a data URL image payload.")
    _, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image frame.")
    return image


def _resize_live_frame(frame: np.ndarray, max_width: int = LIVE_MAX_WIDTH) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def _empty_section(section_name: str, guidance: str) -> dict[str, Any]:
    return {
        "active": False,
        "section": section_name,
        "guidance": guidance,
        "summary": {},
        "summary_text": guidance if section_name == "shoulder" else None,
        "clinical_report": {} if section_name == "spinal" else None,
        "assets": {
            "annotated_video": None,
            "angle_plot": None,
            "symmetry_bar_chart": None,
            "csv": None,
            "time_series": None,
            "standard_comparison": None,
        },
    }


class AnalysisService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._shoulder_estimators: dict[tuple[str, str], Any] = {}
        self._spinal_pipeline: spinal_module.AnatomicalPipeline | None = None
        self._walk_detector: walk_module.Detector | None = None
        self._jobs: dict[str, dict[str, Any]] = {}

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        errors: dict[str, str] = {}

        shoulder_view = str(config.get("shoulder_view", "")).strip().lower()
        if shoulder_view not in ALLOWED_SHOULDER_VIEWS:
            errors["shoulder_view"] = "Choose a shoulder camera view: front or back."

        pose_plane = str(config.get("pose_plane", "")).strip().lower()
        if pose_plane not in ALLOWED_POSE_PLANES:
            errors["pose_plane"] = "Choose a pose plane: frontal, sagittal, or oblique."

        model_size = str(config.get("model_size", DEFAULT_MODEL_SIZE)).strip().lower() or DEFAULT_MODEL_SIZE
        if model_size not in ALLOWED_MODEL_SIZES:
            errors["model_size"] = "Model size must be small, medium, large, or xlarge."

        skip_frames = config.get("skip_frames", 2)
        try:
            skip_frames = int(skip_frames)
            if skip_frames < 1 or skip_frames > 30:
                raise ValueError
        except Exception:
            errors["skip_frames"] = "Skip frames must be a whole number between 1 and 30."

        if errors:
            raise InputValidationError(errors)

        return {
            "shoulder_view": shoulder_view,
            "pose_plane": pose_plane,
            "model_size": model_size or DEFAULT_MODEL_SIZE,
            "skip_frames": skip_frames,
        }

    @staticmethod
    def _clinical_thresholds() -> dict[str, dict[str, Any]]:
        standards = spinal_module.PublishedStandards
        return {
            "shoulder_tilt": {
                "type": "upper_abs",
                "upper": 2.5,
                "reference": "Shoulder normal <= 2.5 deg",
            },
            "trunk_lean": {
                "type": "range_abs",
                "lower": 0.0,
                "upper": standards.GAIT_TRUNK_LEAN_NORMAL_MEAN + 2 * standards.GAIT_TRUNK_LEAN_NORMAL_SD,
                "reference": "Tokuda gait normal range",
            },
            "kyphosis": {
                "type": "range",
                "lower": standards.KYPHOSIS_TR_LO,
                "upper": standards.KYPHOSIS_TR_HI,
                "reference": "Ohlendorf tolerance region",
            },
            "lordosis": {
                "type": "range",
                "lower": standards.LORDOSIS_TR_LO,
                "upper": standards.LORDOSIS_TR_HI,
                "reference": "Ohlendorf tolerance region",
            },
        }

    def _get_shoulder_estimator(self, model_size: str) -> Any:
        key = ("cpu", model_size)
        with self._lock:
            if key not in self._shoulder_estimators:
                self._shoulder_estimators[key] = shoulder_module._SpinePoseEstimator(
                    device="cpu",
                    mode=model_size,
                )
            return self._shoulder_estimators[key]

    def _get_spinal_pipeline(self) -> spinal_module.AnatomicalPipeline:
        with self._lock:
            if self._spinal_pipeline is None:
                self._spinal_pipeline = spinal_module.AnatomicalPipeline(mode="medium")
            return self._spinal_pipeline

    def _get_walk_detector(self) -> walk_module.Detector:
        with self._lock:
            if self._walk_detector is None:
                self._walk_detector = walk_module.Detector(conf=0.5, video_mode=False)
            return self._walk_detector

    @staticmethod
    def _route_for_plane(plane: str) -> dict[str, bool]:
        plane = "sagittal" if plane == "oblique" else plane
        if plane == "frontal":
            return {"shoulder": True, "spinal": False}
        if plane == "sagittal":
            return {"shoulder": False, "spinal": True}
        return {"shoulder": False, "spinal": False}

    @staticmethod
    def _analysis_plane_for(plane: str) -> str:
        return "sagittal" if plane == "oblique" else plane

    @staticmethod
    def _metrics_for_plane(plane: str) -> list[dict[str, Any]]:
        if plane == "frontal":
            return [
                {"key": "shoulder_tilt_deg", "label": "Shoulder Tilt", "unit": "deg", "threshold_key": "shoulder_tilt"},
                {"key": "clavicle_tilt_deg", "label": "Clavicle Tilt", "unit": "deg"},
                {"key": "shoulder_imbalance", "label": "Shoulder Imbalance", "unit": "px"},
                {"key": "trunk_tilt_deg", "label": "Frontal Trunk Tilt", "unit": "deg"},
                {"key": "lateral_shift_pct", "label": "Lateral Shift", "unit": "%"},
            ]
        if plane == "sagittal":
            return [
                {"key": "trunk_lean_angle", "label": "Trunk Lean", "unit": "deg", "threshold_key": "trunk_lean"},
                {"key": "kyphosis_angle", "label": "Kyphosis", "unit": "deg", "threshold_key": "kyphosis"},
                {"key": "lordosis_angle", "label": "Lordosis", "unit": "deg", "threshold_key": "lordosis"},
                {"key": "keypoint_confidence", "label": "Keypoint Confidence", "unit": ""},
            ]
        return []

    @classmethod
    def _standby_metrics_for_plane(cls, plane: str) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        for other_plane in ("frontal", "sagittal"):
            if other_plane == plane:
                continue
            metrics.extend(cls._metrics_for_plane(other_plane))
        return metrics

    @staticmethod
    def _guidance_for_plane(plane: str, fallback_active: bool = False, source_plane: str | None = None) -> str:
        if fallback_active or source_plane == "oblique":
            return "Oblique posture was detected, so the warning stays visible while the sagittal spinal model runs as the fallback analysis."
        if plane == "frontal":
            return "Frontal view activates shoulder alignment, clavicle tilt, and frontal balance metrics."
        if plane == "sagittal":
            return "Sagittal view activates kyphosis, lordosis, trunk lean, and sagittal spinal tracking."
        return "Oblique view detected. Rotate toward frontal or sagittal to unlock the matching posture measurements."

    @staticmethod
    def _normalize_detected_plane(walk_metrics: dict[str, Any] | None) -> str | None:
        if not walk_metrics:
            return None
        label = str(walk_metrics.get("label", "")).strip().lower()
        if label in ALLOWED_POSE_PLANES:
            return label
        return None

    def _resolve_live_plane(self, selected_plane: str, walk_metrics: dict[str, Any] | None) -> str:
        detected_plane = self._normalize_detected_plane(walk_metrics)
        confidence = float(walk_metrics.get("confidence", 0.0)) if walk_metrics else 0.0
        if detected_plane and confidence >= DETECTED_PLANE_CONFIDENCE:
            return detected_plane
        return selected_plane

    def _build_routing_payload(
        self,
        selected_plane: str,
        walk_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        detected_plane = self._normalize_detected_plane(walk_metrics)
        detection_confidence = round(float(walk_metrics.get("confidence", 0.0)), 3) if walk_metrics else None
        dominant_plane = self._resolve_live_plane(selected_plane, walk_metrics)
        effective_plane = self._analysis_plane_for(dominant_plane)
        active_route = self._route_for_plane(effective_plane)
        fallback_active = dominant_plane == "oblique"
        override_active = bool(
            detected_plane
            and detected_plane != selected_plane
            and detection_confidence is not None
            and detection_confidence >= DETECTED_PLANE_CONFIDENCE
        )
        active_metrics = self._metrics_for_plane(effective_plane)
        standby_metrics = self._standby_metrics_for_plane(effective_plane)
        return {
            "selected_pose_plane": selected_plane,
            "detected_pose_plane": detected_plane,
            "detection_confidence": detection_confidence,
            "dominant_pose_plane": dominant_plane,
            "effective_pose_plane": effective_plane,
            "fallback_plane": "sagittal" if fallback_active else None,
            "fallback_active": fallback_active,
            "active_models": [name for name, enabled in active_route.items() if enabled],
            "active_metrics": active_metrics,
            "active_metric_keys": [item["key"] for item in active_metrics],
            "default_chart_metric": active_metrics[0]["key"] if active_metrics else None,
            "standby_metrics": standby_metrics,
            "override_active": override_active,
            "evaluation_source": "detected" if override_active else "selected",
            "guidance": self._guidance_for_plane(effective_plane, fallback_active=fallback_active, source_plane=dominant_plane),
        }

    def _process_shoulder_frame(
        self,
        frame: np.ndarray,
        shoulder_view: str,
        model_size: str,
        frame_idx: int = 0,
    ) -> tuple[dict[str, Any], np.ndarray]:
        estimator = self._get_shoulder_estimator(model_size)
        keypoints, scores = estimator(frame)
        kp_single, sc_single = shoulder_module.select_primary_detection(keypoints, scores)
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
            return {}, annotated

        metrics = shoulder_module.extract_metrics(kp_single, sc_single, frame_idx, shoulder_view)
        annotated = shoulder_module.annotate_frame(frame.copy(), kp_single, sc_single, metrics, shoulder_view)
        return metrics, annotated

    def _process_spinal_frame(self, frame: np.ndarray, frame_idx: int = 0) -> dict[str, Any]:
        result = self._get_spinal_pipeline().process_frame(frame, frame_idx)
        payload = {
            "valid": result.valid,
            "rejection_reason": result.rejection_reason,
            "kyphosis_angle": result.kyphosis_angle,
            "lordosis_angle": result.lordosis_angle,
            "trunk_lean_angle": result.trunk_lean_angle,
            "keypoint_confidence": result.keypoint_confidence,
            "cobb_class": result.cobb_class,
            "trunk_lean_class": result.trunk_lean_class,
            "lordosis_class_ohlendorf": result.lordosis_class_ohlendorf,
            "kyphosis_class_ohlendorf": result.kyphosis_class_ohlendorf,
        }
        return _serialize(payload)

    def _process_walk_frame(self, frame: np.ndarray) -> tuple[dict[str, Any] | None, np.ndarray]:
        detector = self._get_walk_detector()
        annotated, result = detector.process(frame.copy())
        if result is None:
            return None, annotated
        return {
            "label": result.label,
            "confidence": result.confidence,
            "score": result.score,
        }, walk_module.draw_hud(annotated, result)

    def _build_metric_states(
        self,
        live_metrics: dict[str, Any],
        active_route: dict[str, bool],
    ) -> dict[str, Any]:
        thresholds = self._clinical_thresholds()

        def metric_state(
            value: float | None,
            threshold: dict[str, Any],
            active: bool = True,
        ) -> dict[str, Any]:
            if not active:
                return {"state": "inactive", "value": None, "threshold": threshold}
            if not _is_number(value):
                return {"state": "missing", "value": None, "threshold": threshold}
            measured = float(value)
            measured_abs = abs(measured)
            if threshold["type"] == "upper_abs":
                normal = measured_abs <= threshold["upper"]
            elif threshold["type"] == "range_abs":
                normal = threshold["lower"] <= measured_abs <= threshold["upper"]
            else:
                normal = threshold["lower"] <= measured <= threshold["upper"]
            return {
                "state": "normal" if normal else "alert",
                "value": round(measured, 3),
                "threshold": threshold,
            }

        return {
            "shoulder_tilt": metric_state(
                live_metrics.get("shoulder_tilt_deg"),
                thresholds["shoulder_tilt"],
                active=active_route["shoulder"],
            ),
            "trunk_lean": metric_state(
                live_metrics.get("trunk_lean_angle"),
                thresholds["trunk_lean"],
                active=active_route["spinal"],
            ),
            "kyphosis": metric_state(
                live_metrics.get("kyphosis_angle"),
                thresholds["kyphosis"],
                active=active_route["spinal"],
            ),
            "lordosis": metric_state(
                live_metrics.get("lordosis_angle"),
                thresholds["lordosis"],
                active=active_route["spinal"],
            ),
        }

    def _draw_live_summary(
        self,
        frame: np.ndarray,
        live_metrics: dict[str, Any],
        metric_states: dict[str, Any],
        walk_result: dict[str, Any] | None,
        routing: dict[str, Any],
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        panel_x = max(w - 340, 16)
        panel_y = 84
        line_height = 30

        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x - 18, panel_y - 44), (w - 18, panel_y + 7 * line_height + 40), (5, 12, 24), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.putText(canvas, "Live Motion Router", (panel_x, panel_y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 247, 255), 2)
        selected_text = f"Selected {routing['selected_pose_plane']}"
        effective_text = f"Active {routing['effective_pose_plane']}"
        cv2.putText(canvas, selected_text, (panel_x, panel_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 176, 204), 1)
        cv2.putText(canvas, effective_text, (panel_x + 160, panel_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 222, 255), 1)

        rows = [
            ("Shoulder Tilt", live_metrics.get("shoulder_tilt_deg"), metric_states["shoulder_tilt"]),
            ("Trunk Lean", live_metrics.get("trunk_lean_angle"), metric_states["trunk_lean"]),
            ("Kyphosis", live_metrics.get("kyphosis_angle"), metric_states["kyphosis"]),
            ("Lordosis", live_metrics.get("lordosis_angle"), metric_states["lordosis"]),
        ]

        for index, (label, value, state) in enumerate(rows):
            y = panel_y + 34 + index * line_height
            color = (0, 215, 120) if state["state"] == "normal" else (48, 70, 230)
            if state["state"] == "missing":
                color = (140, 140, 140)
            if state["state"] == "inactive":
                color = (110, 128, 148)
            rendered = "--" if value is None else f"{float(value):.2f}"
            threshold = state["threshold"]
            if state["state"] == "inactive":
                threshold_text = "inactive for this view"
            elif threshold["type"] == "upper_abs":
                threshold_text = f"normal <= {threshold['upper']:.1f}"
            else:
                threshold_text = f"normal {threshold['lower']:.1f} to {threshold['upper']:.1f}"
            cv2.putText(canvas, label, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (225, 235, 245), 1)
            cv2.putText(canvas, rendered, (panel_x + 165, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            cv2.putText(canvas, threshold_text, (panel_x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 176, 204), 1)

        if walk_result:
            walk_text = f"Detected {walk_result['label']} ({walk_result['confidence'] * 100:.0f}%)"
            cv2.putText(canvas, walk_text, (panel_x, panel_y + 5 * line_height + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 214, 255), 1)
        if routing.get("fallback_active"):
            cv2.putText(
                canvas,
                "Oblique detected: sagittal fallback is active",
                (panel_x, panel_y + 6 * line_height + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 187, 134),
                1,
            )

        return canvas

    def analyze_live_frame(self, image_data: str, config: dict[str, Any]) -> dict[str, Any]:
        validated = self.validate_config(config)
        frame = _resize_live_frame(_decode_frame(image_data))
        walk_metrics, walk_frame = self._process_walk_frame(frame)
        routing = self._build_routing_payload(validated["pose_plane"], walk_metrics)
        active_route = self._route_for_plane(routing["effective_pose_plane"])

        shoulder_metrics: dict[str, Any] = {}
        shoulder_frame = frame
        if active_route["shoulder"]:
            shoulder_metrics, shoulder_frame = self._process_shoulder_frame(
                frame,
                validated["shoulder_view"],
                validated["model_size"],
            )

        spinal_metrics: dict[str, Any] = {}
        if active_route["spinal"]:
            spinal_metrics = self._process_spinal_frame(frame)

        merged = {
            **_serialize(shoulder_metrics),
            **spinal_metrics,
            "pose_plane_detected": walk_metrics,
            "selected_pose_plane": validated["pose_plane"],
            "effective_pose_plane": routing["effective_pose_plane"],
        }
        states = self._build_metric_states(merged, active_route)
        routing["clinical_thresholds"] = self._clinical_thresholds()

        preferred_frame = shoulder_frame if active_route["shoulder"] else frame
        if walk_frame is not None:
            preferred_frame = walk_frame
        annotated = self._draw_live_summary(preferred_frame, merged, states, walk_metrics, routing)

        return {
            "metrics": merged,
            "threshold_states": states,
            "routing": routing,
            "annotated_frame": _encode_frame(annotated),
        }

    def _summarize_shoulder_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "frames": len(rows),
            "shoulder_tilt_mean": _mean([row.get("shoulder_tilt_deg") for row in rows]),
            "clavicle_tilt_mean": _mean([row.get("clavicle_tilt_deg") for row in rows]),
            "shoulder_imbalance_mean": _mean([row.get("shoulder_imbalance") for row in rows]),
            "trunk_tilt_mean": _mean([row.get("trunk_tilt_deg") for row in rows]),
            "lateral_shift_mean": _mean([row.get("lateral_shift_pct") for row in rows]),
        }

    def _detect_pose_plane_from_video(self, video_path: str) -> dict[str, Any] | None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        detector = self._get_walk_detector()
        labels: list[str] = []
        confidences: list[float] = []
        checked = 0

        try:
            while checked < 18:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, result = detector.process(frame)
                if result is not None:
                    labels.append(result.label.lower())
                    confidences.append(result.confidence)
                checked += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, checked * 10)
        finally:
            cap.release()

        if not labels:
            return None

        label, votes = Counter(labels).most_common(1)[0]
        return {
            "label": label,
            "confidence": round(float(sum(confidences) / len(confidences)), 3),
            "votes": votes,
            "samples": len(labels),
        }

    def analyze_video(self, video_path: str, config: dict[str, Any], include_exports: bool = True) -> dict[str, Any]:
        validated = self.validate_config(config)
        result_id = _slug("analysis")
        result_dir = RESULTS_ROOT / result_id
        shoulder_dir = result_dir / "shoulder"
        spinal_dir = result_dir / "spinal"
        shoulder_dir.mkdir(parents=True, exist_ok=True)
        spinal_dir.mkdir(parents=True, exist_ok=True)
        detected_plane = self._detect_pose_plane_from_video(video_path)
        routing = self._build_routing_payload(validated["pose_plane"], detected_plane)
        active_route = self._route_for_plane(routing["effective_pose_plane"])

        warnings: list[str] = []
        if detected_plane and detected_plane["label"] == "oblique" and routing.get("fallback_active"):
            warnings.append(
                "Oblique posture was detected. The app kept the oblique warning visible and ran the sagittal spinal workflow as the fallback analysis."
            )
        elif validated["pose_plane"] == "oblique":
            warnings.append(
                "Oblique was selected manually, so the sagittal spinal workflow was used as the fallback analysis."
            )
        elif detected_plane and validated["pose_plane"] != detected_plane["label"]:
            if routing["override_active"]:
                warnings.append(
                    f"Detector confidence is {routing['detection_confidence'] * 100:.0f}%, so routing switched from '{validated['pose_plane']}' to '{routing['effective_pose_plane']}'."
                )
            else:
                warnings.append(
                    f"The selected pose plane is '{validated['pose_plane']}', while detection leans toward '{detected_plane['label']}'. Confidence stayed below the auto-switch threshold, so the selected workflow was kept."
                )

        shoulder_manifest = _empty_section(
            "shoulder",
            "Shoulder alignment runs on frontal captures.",
        )
        if active_route["shoulder"]:
            shoulder_rows = shoulder_module.process_video(
                input_path=video_path,
                output_dir=str(shoulder_dir),
                device="cpu",
                model_size=validated["model_size"],
                skip_frames=validated["skip_frames"],
                view=validated["shoulder_view"],
            )
            shoulder_manifest = {
                "active": True,
                "summary": self._summarize_shoulder_rows(shoulder_rows),
                "summary_text": _read_text(shoulder_dir / "summary.txt"),
                "guidance": "Frontal workflow active: shoulder alignment, clavicle tilt, and symmetry outputs generated.",
                "assets": {
                    "annotated_video": _to_url(shoulder_dir / "annotated_video.mp4"),
                    "angle_plot": _to_url(shoulder_dir / "angle_plot.png"),
                    "symmetry_bar_chart": _to_url(shoulder_dir / "symmetry_bar_chart.png"),
                    "csv": _to_url(shoulder_dir / "shoulder_angles.csv"),
                },
            }

        spinal_manifest = _empty_section(
            "spinal",
            "Spinal curvature runs on sagittal captures.",
        )
        if active_route["spinal"]:
            spinal_result = self._get_spinal_pipeline().analyze_video(video_path)
            report = spinal_module.ClinicalComparisonReport.generate(
                kyphosis_deg=spinal_result.summary.get("kyphosis", {}).get("mean"),
                lordosis_deg=spinal_result.summary.get("lordosis", {}).get("mean"),
                trunk_lean_deg=spinal_result.summary.get("trunk_lean", {}).get("mean"),
                context="standing",
            )

            spinal_plot = spinal_dir / "time_series.png"
            spinal_compare = spinal_dir / "standard_comparison.png"
            spinal_module.Plotter.plot_time_series(spinal_result, str(spinal_plot))
            spinal_module.Plotter.plot_standard_comparison(report, str(spinal_compare))

            spinal_video_path = ""
            if include_exports:
                spinal_video_path = spinal_module.VideoExporter.export(
                    video_path,
                    spinal_result,
                    report,
                    str(spinal_dir / "annotated_spinal.mp4"),
                )

            spinal_manifest = {
                "active": True,
                "summary": _serialize(spinal_result.summary),
                "clinical_report": _serialize(report),
                "guidance": "Sagittal workflow active: kyphosis, lordosis, and trunk lean outputs generated.",
                "assets": {
                    "time_series": _to_url(spinal_plot),
                    "standard_comparison": _to_url(spinal_compare),
                    "annotated_video": _to_url(Path(spinal_video_path)) if spinal_video_path else None,
                    "angle_plot": None,
                    "symmetry_bar_chart": None,
                    "csv": None,
                },
            }

        manifest = {
            "analysis_id": result_id,
            "video_name": Path(video_path).name,
            "config": validated,
            "warnings": warnings,
            "detected_pose_plane": detected_plane,
            "routing": routing,
            "shoulder": shoulder_manifest,
            "spinal": spinal_manifest,
        }

        (result_dir / "manifest.json").write_text(
            json.dumps(_serialize(manifest), indent=2),
            encoding="utf-8",
        )
        return _serialize(manifest)

    def _resolve_video_path(self, raw_path: str, csv_dir: Path) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        uploads_candidate = (UPLOAD_ROOT / raw_path).resolve()
        if uploads_candidate.exists():
            return uploads_candidate
        return (csv_dir / raw_path).resolve()

    def start_research_job(self, csv_path: Path, config: dict[str, Any]) -> str:
        validated = self.validate_config(config)
        job_id = _slug("research")
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued for processing.",
            "created_at": time.time(),
            "result": None,
        }
        worker = threading.Thread(
            target=self._run_research_job,
            args=(job_id, csv_path, validated),
            daemon=True,
        )
        worker.start()
        return job_id

    def _flatten_research_row(self, row: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
        shoulder_summary = analysis["shoulder"]["summary"]
        spinal_summary = analysis["spinal"]["summary"]
        routing = analysis.get("routing", {})
        return {
            **row,
            "selected_pose_plane": routing.get("selected_pose_plane"),
            "detected_pose_plane": routing.get("detected_pose_plane"),
            "effective_pose_plane": routing.get("effective_pose_plane"),
            "active_models": ", ".join(routing.get("active_models") or []),
            "shoulder_tilt_mean": shoulder_summary.get("shoulder_tilt_mean"),
            "clavicle_tilt_mean": shoulder_summary.get("clavicle_tilt_mean"),
            "shoulder_imbalance_mean": shoulder_summary.get("shoulder_imbalance_mean"),
            "trunk_tilt_mean": shoulder_summary.get("trunk_tilt_mean"),
            "kyphosis_mean": spinal_summary.get("kyphosis", {}).get("mean"),
            "lordosis_mean": spinal_summary.get("lordosis", {}).get("mean"),
            "trunk_lean_mean": spinal_summary.get("trunk_lean", {}).get("mean"),
            "rejection_rate": spinal_summary.get("rejection_rate"),
            "analysis_id": analysis.get("analysis_id"),
        }

    def _aggregate_people(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["person_id"]][row["state"]].append(row)

        result: list[dict[str, Any]] = []
        for person_id, states in grouped.items():
            entry = {"person_id": person_id, "states": {}}
            for state, state_rows in states.items():
                entry["states"][state] = {
                    "video_count": len(state_rows),
                    "shoulder_tilt_mean": _mean([row.get("shoulder_tilt_mean") for row in state_rows]),
                    "kyphosis_mean": _mean([row.get("kyphosis_mean") for row in state_rows]),
                    "lordosis_mean": _mean([row.get("lordosis_mean") for row in state_rows]),
                    "trunk_lean_mean": _mean([row.get("trunk_lean_mean") for row in state_rows]),
                }
            result.append(entry)
        return result

    def _build_state_comparison(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        per_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            per_state[row["state"]].append(row)

        comparison = []
        for state, state_rows in per_state.items():
            comparison.append(
                {
                    "state": state,
                    "shoulder_tilt_mean": _mean([row.get("shoulder_tilt_mean") for row in state_rows]),
                    "kyphosis_mean": _mean([row.get("kyphosis_mean") for row in state_rows]),
                    "lordosis_mean": _mean([row.get("lordosis_mean") for row in state_rows]),
                    "trunk_lean_mean": _mean([row.get("trunk_lean_mean") for row in state_rows]),
                }
            )
        return comparison

    def _write_research_csv(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _run_research_job(self, job_id: str, csv_path: Path, config: dict[str, Any]) -> None:
        job = self._jobs[job_id]
        job["status"] = "running"
        job["message"] = "Reading dataset."

        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                required = {"person_id", "video_path", "state", "label", "notes"}
                if reader.fieldnames is None or not {"person_id", "video_path", "state"}.issubset(set(reader.fieldnames)):
                    raise InputValidationError(
                        {
                            "csv": "CSV must include person_id, video_path, and state columns."
                        }
                    )

            total = len(rows)
            output_dir = RESULTS_ROOT / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            results: list[dict[str, Any]] = []
            skipped: list[str] = []

            for index, raw_row in enumerate(rows, start=1):
                row = {key: (value or "").strip() for key, value in raw_row.items()}
                job["progress"] = round(((index - 1) / max(total, 1)) * 100)
                job["message"] = f"Processing row {index} of {total}."

                if not row.get("video_path"):
                    skipped.append(f"Row {index}: missing video_path.")
                    continue

                resolved = self._resolve_video_path(row["video_path"], csv_path.parent)
                if not resolved.exists():
                    skipped.append(f"Row {index}: file not found at {resolved}.")
                    continue

                analysis = self.analyze_video(str(resolved), config, include_exports=False)
                results.append(self._flatten_research_row(row, analysis))

            people = self._aggregate_people(results)
            state_comparison = self._build_state_comparison(results)
            export_path = output_dir / "research_results.csv"
            self._write_research_csv(export_path, results)

            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = "Researcher mode analysis is ready."
            job["result"] = {
                "rows": results,
                "people": people,
                "state_comparison": state_comparison,
                "skipped": skipped,
                "export_csv": _to_url(export_path) if export_path.exists() else None,
            }
        except InputValidationError as exc:
            job["status"] = "failed"
            job["message"] = "CSV validation failed."
            job["result"] = {"errors": exc.errors}
        except Exception as exc:
            job["status"] = "failed"
            job["message"] = str(exc)
            job["result"] = {"errors": {"server": str(exc)}}

    def get_research_job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return _serialize(job)
