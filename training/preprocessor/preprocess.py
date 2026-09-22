"""
training/preprocessor/preprocess.py
=====================================
Preprocess raw video/audio files into model-ready feature files (.npz).

For each subject file, extracts:
  - speech_emb   : (T_s, 768) Wav2Vec2 embeddings
  - visual_emb   : (T_v, 1404) MediaPipe FaceMesh landmarks
  - audio_feats  : dict of 9 heuristic audio features
  - visual_feats : dict of 8 heuristic visual features
  - label        : 0 (HC) or 1 (AD)

Output: training/data/processed/<LABEL>_<ID>.npz

Usage:
  cd project_final
  python training/preprocessor/preprocess.py
  python training/preprocessor/preprocess.py --limit 10   # process first 10 only
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import (
    _extract_wav2vec2_embeddings,
    _extract_landmarks_mediapipe,
    _extract_heuristic_audio_features,
    _extract_visual_features_mediapipe,
    _extract_heuristic_visual_features,
    MP_AVAILABLE,
    CV2_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
)

RAW_AD_DIR   = Path("training/data/raw/AD")
RAW_HC_DIR   = Path("training/data/raw/HC")
PROC_DIR     = Path("training/data/processed")
VIDEO_EXTS   = (".mp4", ".mov", ".avi", ".mkv", ".webm")
AUDIO_EXTS   = (".wav", ".mp3", ".m4a")


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found!\n"
            "  Colab: !apt-get install -y ffmpeg\n"
            "  Windows: choco install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


def _extract_audio(video_path: str, out_wav: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        out_wav
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{r.stderr[-800:]}")


def _extract_frames(video_path: str, out_dir: str, fps: int = 2):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale=320:-1",
        "-q:v", "3",
        os.path.join(out_dir, "frame_%05d.jpg")
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{r.stderr[-800:]}")


def preprocess_file(src_path: Path, label: int, subject_id: str,
                    out_dir: Path) -> bool:
    """
    Process one raw file → save .npz to out_dir.
    Returns True on success, False on failure.
    """
    out_path = out_dir / f"{'AD' if label == 1 else 'HC'}_{subject_id}.npz"
    if out_path.exists():
        print(f"  [SKIP] {out_path.name} already processed")
        return True

    suffix = src_path.suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav_path    = str(tmp / "audio.wav")
        frames_dir  = str(tmp / "frames")

        # ── Extract audio ──
        if suffix in VIDEO_EXTS:
            _extract_audio(str(src_path), wav_path)
        elif suffix in AUDIO_EXTS:
            cmd = ["ffmpeg", "-y", "-i", str(src_path),
                   "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav_path]
            subprocess.run(cmd, capture_output=True)
        else:
            print(f"  [SKIP] Unsupported format: {src_path.suffix}")
            return False

        if not os.path.exists(wav_path):
            print(f"  [FAIL] Audio extraction failed for {src_path.name}")
            return False

        # ── Extract frames (video only) ──
        if suffix in VIDEO_EXTS:
            _extract_frames(str(src_path), frames_dir, fps=2)
        else:
            os.makedirs(frames_dir, exist_ok=True)

        # ── Extract features ──
        speech_emb   = _extract_wav2vec2_embeddings(wav_path)
        visual_emb   = _extract_landmarks_mediapipe(frames_dir)
        audio_feats  = _extract_heuristic_audio_features(wav_path)
        visual_feats = (_extract_visual_features_mediapipe(frames_dir)
                        if MP_AVAILABLE and CV2_AVAILABLE
                        else _extract_heuristic_visual_features(frames_dir))

        # ── Convert to numpy for storage ──
        s_np = speech_emb.squeeze(0).cpu().numpy() if speech_emb is not None else None
        v_np = visual_emb.squeeze(0).cpu().numpy() if visual_emb is not None else None

        # ── Build heuristic feature vector (14 dims) ──
        feat_vector = np.array([
            audio_feats.get("pause_ratio", 0.0),
            audio_feats.get("speech_rate_proxy", 0.0),
            audio_feats.get("energy_mean", 0.0),
            audio_feats.get("energy_std", 0.0),
            audio_feats.get("mfcc_variance", 0.0),
            audio_feats.get("pitch_variability", 0.0),
            audio_feats.get("zero_crossing_rate", 0.0),
            audio_feats.get("spectral_centroid", 0.0),
            audio_feats.get("duration_sec", 0.0),
            visual_feats.get("face_detected_ratio", 0.0),
            visual_feats.get("expressiveness_score", 0.0),
            visual_feats.get("inter_frame_motion", 0.0),
            visual_feats.get("visual_entropy", 0.0),
            visual_feats.get("au_variance", 0.0),
        ], dtype=np.float32)

        # ── Save .npz ──
        save_dict = {
            "label":        np.array(label, dtype=np.int64),
            "feat_vector":  feat_vector,
        }
        if s_np is not None:
            save_dict["speech_emb"] = s_np
        if v_np is not None:
            save_dict["visual_emb"] = v_np

        np.savez_compressed(str(out_path), **save_dict)

    inferred = "transformer" if speech_emb is not None else "heuristic"
    print(f"  [OK] {out_path.name}  "
          f"(speech={speech_emb is not None}, "
          f"visual={visual_emb is not None}, "
          f"method={inferred})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Neuro Fusion v2 — Preprocessor")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to process (for testing)")
    args = parser.parse_args()

    _check_ffmpeg()
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all source files
    ad_files = sorted([
        f for f in RAW_AD_DIR.iterdir()
        if f.suffix.lower() in VIDEO_EXTS + AUDIO_EXTS
    ]) if RAW_AD_DIR.exists() else []

    hc_files = sorted([
        f for f in RAW_HC_DIR.iterdir()
        if f.suffix.lower() in VIDEO_EXTS + AUDIO_EXTS
    ]) if RAW_HC_DIR.exists() else []

    if not ad_files and not hc_files:
        print("No raw files found.")
        print("Add videos to training/data/raw/AD/ and training/data/raw/HC/")
        return

    total = len(ad_files) + len(hc_files)
    print(f"\nPreprocessing {len(ad_files)} AD + {len(hc_files)} HC files")
    print(f"  Wav2Vec2: {'✓' if TRANSFORMERS_AVAILABLE else '✗ (install transformers)'}")
    print(f"  FaceMesh: {'✓' if (MP_AVAILABLE and CV2_AVAILABLE) else '✗ (install mediapipe + opencv)'}")
    print(f"  Output:   {PROC_DIR}\n")

    ok = err = 0
    count = 0
    for label, files in [(1, ad_files), (0, hc_files)]:
        lname = "AD" if label == 1 else "HC"
        print(f"── {lname} ({len(files)} files) ──")
        for f in files:
            if args.limit and count >= args.limit:
                break
            sid = f.stem
            try:
                success = preprocess_file(f, label, sid, PROC_DIR)
                if success:
                    ok += 1
                else:
                    err += 1
            except Exception as e:
                print(f"  [ERROR] {f.name}: {e}")
                err += 1
            count += 1
        if args.limit and count >= args.limit:
            break

    proc_files = list(PROC_DIR.glob("*.npz"))
    ad_proc    = sum(1 for f in proc_files if f.name.startswith("AD_"))
    hc_proc    = sum(1 for f in proc_files if f.name.startswith("HC_"))

    print(f"\n{'='*50}")
    print(f"  Preprocessing complete")
    print(f"  Processed:  {ok} ok | {err} errors")
    print(f"  Saved:      AD={ad_proc} | HC={hc_proc} | Total={len(proc_files)}")
    if len(proc_files) >= 20:
        print(f"  ✓ Ready for training. Run:")
        print(f"    python training/trainer/train.py --stage A")
        if len(proc_files) >= 40:
            print(f"    python training/trainer/train.py --stage B")
    else:
        print(f"  ⚠ Need at least 20 total — collect more data first.")
    print("="*50)


if __name__ == "__main__":
    main()
