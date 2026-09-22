"""
app.py — Neuro Fusion v2 FastAPI Application
=============================================
Routes:
  GET  /                 → home page (module selection)
  GET  /alzheimers       → AD video upload page
  POST /upload/alzheimers→ analyze video for AD
  GET  /dementia         → Dementia audio upload page
  POST /upload/dementia  → analyze audio for Dementia
  GET  /chat             → chatbot page
  POST /chat             → ask question
  GET  /history          → past results (both modules)
  GET  /api/health        → health check
"""

import os
import uuid
import json
import time
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from utils import ensure_dirs, save_upload, extract_audio_ffmpeg, extract_frames_ffmpeg
from model import predict_from_audio_frames
from model_dementia import predict_dementia
from chatbot import ask_chatbot, rebuild_knowledge_base

APP_TITLE   = "Neuro Fusion — Alzheimer's & Dementia Screening"
APP_VERSION = "2.0.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR  = "uploads"
AUDIO_DIR   = os.path.join("outputs", "audio")
FRAMES_DIR  = os.path.join("outputs", "frames")
RESULTS_DIR = os.path.join("outputs", "results")

ensure_dirs([UPLOAD_DIR, AUDIO_DIR, FRAMES_DIR, RESULTS_DIR,
             "knowledge_base", "checkpoints"])


# ─────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="home.html",
        context={"title": APP_TITLE},
    )


@app.get("/alzheimers", response_class=HTMLResponse)
def alzheimers_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="alzheimers.html",
        context={"title": APP_TITLE},
    )


@app.get("/dementia", response_class=HTMLResponse)
def dementia_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="dementia.html",
        context={"title": APP_TITLE},
    )


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="chat.html",
        context={"title": APP_TITLE},
    )


@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    records = _load_recent_results(20)
    return templates.TemplateResponse(
        request=request, name="history.html",
        context={"title": APP_TITLE, "records": records},
    )


# ─────────────────────────────────────────────────────────────
# Alzheimer's — video upload + analysis
# ─────────────────────────────────────────────────────────────

@app.post("/upload/alzheimers", response_class=HTMLResponse)
async def upload_alzheimers(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    filename = (video.filename or "").strip()
    allowed  = (".mp4", ".mov", ".webm", ".mkv", ".avi")

    if not filename.lower().endswith(allowed):
        return templates.TemplateResponse(
            request=request, name="results_ad.html",
            context=dict(title=APP_TITLE,
                         error=f"Unsupported file. Use: {', '.join(allowed).upper()}",
                         error_detail=None),
            status_code=400,
        )

    job_id     = str(uuid.uuid4())[:8]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename.replace(' ','_')}")
    await save_upload(video, video_path)

    audio_path     = os.path.join(AUDIO_DIR, f"{job_id}.wav")
    frames_out_dir = os.path.join(FRAMES_DIR, job_id)
    started_at     = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    t0             = time.time()

    try:
        extract_audio_ffmpeg(video_path, audio_path)
        extract_frames_ffmpeg(video_path, frames_out_dir, fps=5)

        frames_extracted = (
            len([f for f in os.listdir(frames_out_dir) if f.endswith(".jpg")])
            if os.path.exists(frames_out_dir) else 0
        )

        ad_result = predict_from_audio_frames(audio_path, frames_out_dir)
        elapsed   = round(time.time() - t0, 2)

        record = dict(
            job_id=job_id,
            module="alzheimers",
            video_name=filename,
            file_size_mb=round(os.path.getsize(video_path)/(1024*1024), 2),
            frames_extracted=frames_extracted,
            started_at=started_at,
            elapsed_sec=elapsed,
            prediction=ad_result["prediction"],
            risk_score_percent=ad_result["risk_score_percent"],
            risk_level=ad_result["risk_level"],
            speech_features=ad_result.get("speech_features", {}),
            visual_features=ad_result.get("visual_features", {}),
            explainability=ad_result.get("explainability", {}),
            notes=ad_result.get("notes", ""),
        )

        background_tasks.add_task(_save_result, job_id, record)

        return templates.TemplateResponse(
            request=request, name="results_ad.html",
            context=dict(title=APP_TITLE, error=None, **record),
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request=request, name="results_ad.html",
            context=dict(title=APP_TITLE,
                         error=f"Processing failed: {exc}",
                         error_detail="Make sure ffmpeg is installed."),
            status_code=500,
        )


# ─────────────────────────────────────────────────────────────
# Dementia — audio upload + analysis
# ─────────────────────────────────────────────────────────────

@app.post("/upload/dementia", response_class=HTMLResponse)
async def upload_dementia(
    request: Request,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
):
    filename = (audio.filename or "").strip()
    allowed  = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

    if not filename.lower().endswith(allowed):
        return templates.TemplateResponse(
            request=request, name="results_dementia.html",
            context=dict(title=APP_TITLE,
                         error=f"Unsupported file. Use: {', '.join(allowed).upper()}",
                         error_detail=None),
            status_code=400,
        )

    job_id    = str(uuid.uuid4())[:8]
    raw_path  = os.path.join(UPLOAD_DIR, f"{job_id}_{filename.replace(' ','_')}")
    await save_upload(audio, raw_path)

    wav_path   = os.path.join(AUDIO_DIR, f"{job_id}.wav")
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    t0         = time.time()

    try:
        # Convert any audio format to 16kHz mono WAV
        extract_audio_ffmpeg(raw_path, wav_path)

        # Run AD heuristic audio features (for feature table) + dementia model
        from model import _extract_heuristic_audio_features
        audio_feats = _extract_heuristic_audio_features(wav_path)

        dem_result = predict_dementia(audio_path=wav_path,
                                      audio_feats=audio_feats)
        elapsed = round(time.time() - t0, 2)

        speech_table = dict(
            Duration=f"{audio_feats['duration_sec']}s",
            PauseRatio=f"{round(audio_feats['pause_ratio']*100,1)}%",
            SpeechRate=f"{audio_feats['speech_rate_proxy']} bursts/s",
            MFCCVariance=str(audio_feats["mfcc_variance"]),
            EnergyStd=str(audio_feats["energy_std"]),
            PitchVariability=str(audio_feats["pitch_variability"]),
            ZCR=str(audio_feats["zero_crossing_rate"]),
        )

        record = dict(
            job_id=job_id,
            module="dementia",
            audio_name=filename,
            elapsed_sec=elapsed,
            started_at=started_at,
            dementia_prediction=dem_result["dementia_prediction"],
            dementia_risk_percent=dem_result["dementia_risk_percent"],
            dementia_level=dem_result["dementia_level"],
            dementia_confidence=dem_result["dementia_confidence"],
            dementia_method=dem_result["dementia_method"],
            dementia_contributions=dem_result.get("dementia_contributions", {}),
            attention_highlights=dem_result.get("attention_highlights", []),
            speech_features=speech_table,
        )

        background_tasks.add_task(_save_result, job_id, record)

        return templates.TemplateResponse(
            request=request, name="results_dementia.html",
            context=dict(title=APP_TITLE, error=None, **record),
        )

    except Exception as exc:
        return templates.TemplateResponse(
            request=request, name="results_dementia.html",
            context=dict(title=APP_TITLE,
                         error=f"Processing failed: {exc}",
                         error_detail="Make sure ffmpeg is installed."),
            status_code=500,
        )


# ─────────────────────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(question: str = Form(...)):
    result = ask_chatbot(question)
    return JSONResponse(result)


@app.post("/api/rebuild-kb")
def rebuild_kb():
    return JSONResponse(rebuild_knowledge_base())


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    import shutil
    return JSONResponse(dict(
        status="ok",
        version=APP_VERSION,
        ffmpeg=shutil.which("ffmpeg") is not None,
    ))


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    with open(path) as f:
        return JSONResponse(json.load(f))


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _save_result(job_id: str, data: dict):
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    path  = os.path.join(RESULTS_DIR, f"{job_id}.json")
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def _load_recent_results(limit: int = 20) -> list:
    results = []
    if not os.path.exists(RESULTS_DIR):
        return results
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(RESULTS_DIR, fname)) as f:
                    results.append(json.load(f))
            except Exception:
                pass
        if len(results) >= limit:
            break
    return results
