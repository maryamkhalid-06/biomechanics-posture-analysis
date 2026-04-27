from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .analysis_service import ALLOWED_MODEL_SIZES, ALLOWED_POSE_PLANES, ALLOWED_SHOULDER_VIEWS
from .analysis_service import AnalysisService, InputValidationError, STORAGE_ROOT, UPLOAD_ROOT

app = FastAPI(title="Biomechanics Analysis API", version="1.0.0")
service = AnalysisService()


def _cors_origins() -> list[str]:
    configured = [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()]
    if configured:
        return configured
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(STORAGE_ROOT)), name="files")
FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "1").strip().lower() not in {"0", "false", "no"}


def _parse_config(
    shoulder_view: str,
    pose_plane: str,
    model_size: str,
    skip_frames: int,
) -> dict:
    return {
        "shoulder_view": shoulder_view,
        "pose_plane": pose_plane,
        "model_size": model_size,
        "skip_frames": skip_frames,
    }


@app.exception_handler(InputValidationError)
async def validation_error_handler(_, exc: InputValidationError):
    return JSONResponse(status_code=422, content={"errors": exc.errors})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "options": {
            "shoulder_views": sorted(ALLOWED_SHOULDER_VIEWS),
            "pose_planes": sorted(ALLOWED_POSE_PLANES),
            "model_sizes": sorted(ALLOWED_MODEL_SIZES),
        },
    }


if not (SERVE_FRONTEND and FRONTEND_ROOT.exists()):
    @app.get("/")
    def root():
        return {
            "status": "ok",
            "message": "Biomechanics backend is running.",
            "frontend_served_here": False,
        }


def _merge_live_config(payload: dict) -> dict:
    base = {
        "shoulder_view": "front",
        "pose_plane": "frontal",
        "model_size": "medium",
        "skip_frames": 2,
    }
    base.update(payload or {})
    return service.validate_config(base)


@app.post("/api/analyze/video")
async def analyze_video(
    video: UploadFile = File(...),
    shoulder_view: str = Form(...),
    pose_plane: str = Form(...),
    model_size: str = Form("medium"),
    skip_frames: int = Form(2),
):
    if not video.filename:
        raise InputValidationError({"video": "Please upload a video file."})

    destination_dir = UPLOAD_ROOT / "single"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / video.filename
    destination.write_bytes(await video.read())

    config = _parse_config(
        shoulder_view,
        pose_plane,
        model_size,
        skip_frames,
    )
    return service.analyze_video(str(destination), config, include_exports=True)


@app.post("/api/research/jobs")
async def create_research_job(
    csv_file: UploadFile = File(...),
    shoulder_view: str = Form(...),
    pose_plane: str = Form(...),
    model_size: str = Form("medium"),
    skip_frames: int = Form(2),
):
    if not csv_file.filename:
        raise InputValidationError({"csv": "Please upload a CSV file."})

    destination_dir = UPLOAD_ROOT / "research"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / csv_file.filename
    destination.write_bytes(await csv_file.read())

    config = _parse_config(
        shoulder_view,
        pose_plane,
        model_size,
        skip_frames,
    )
    job_id = service.start_research_job(destination, config)
    return {"job_id": job_id}


@app.get("/api/research/jobs/{job_id}")
def get_research_job(job_id: str):
    job = service.get_research_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/api/live/frame")
async def analyze_live_frame(payload: Annotated[dict, Body(...)]):
    image = payload.get("image")
    if not image:
        raise InputValidationError({"frame": "Live frame is missing."})

    config = _merge_live_config(payload.get("config") or {})
    try:
        return service.analyze_live_frame(str(image), config)
    except InputValidationError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/ws/live-analysis")
async def live_analysis(websocket: WebSocket):
    await websocket.accept()
    config = _merge_live_config({})

    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)
            message_type = payload.get("type")

            if message_type == "config":
                config = _merge_live_config(payload.get("payload", {}))
                await websocket.send_json({"type": "config-ack", "payload": config})
                continue

            if message_type == "frame":
                try:
                    result = service.analyze_live_frame(payload["payload"]["image"], config)
                    await websocket.send_json({"type": "analysis", "payload": result})
                except InputValidationError as exc:
                    await websocket.send_json({"type": "error", "payload": {"errors": exc.errors}})
                except Exception as exc:
                    await websocket.send_json({"type": "error", "payload": {"errors": {"frame": str(exc)}}})
                continue

            await websocket.send_json({"type": "error", "payload": {"errors": {"message": "Unknown message type."}}})
    except WebSocketDisconnect:
        return


if SERVE_FRONTEND and FRONTEND_ROOT.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_ROOT), html=True), name="frontend")
