import os
import shutil
import subprocess
from fastapi import UploadFile


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


async def save_upload(upload: UploadFile, out_path: str):
    with open(out_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found.\n"
            "  Windows : choco install ffmpeg   OR  https://ffmpeg.org/download.html\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg"
        )


def extract_audio_ffmpeg(video_path: str, audio_out_path: str):
    _check_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",          # no video
        "-ac", "1",     # mono
        "-ar", "16000", # 16 kHz — standard for speech models
        "-sample_fmt", "s16",
        audio_out_path
    ]
    _run(cmd, "Audio extraction failed")


def extract_frames_ffmpeg(video_path: str, frames_out_dir: str, fps: int = 5):
    _check_ffmpeg()
    os.makedirs(frames_out_dir, exist_ok=True)
    out_pattern = os.path.join(frames_out_dir, "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps},scale=320:-1",  # resize for speed
        "-q:v", "3",   # JPEG quality 1-31 (lower = better)
        out_pattern
    ]
    _run(cmd, "Frame extraction failed")


def _run(cmd: list, err_msg: str):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{err_msg}\n\nffmpeg output:\n{p.stderr[-1500:].strip()}")
