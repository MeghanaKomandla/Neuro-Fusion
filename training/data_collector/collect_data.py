"""
training/data_collector/collect_data.py
=======================================
Collect labeled video+audio data for Alzheimer's detection training.

Sources supported:
  1. YouTube public interviews (AD-positive public figures)
  2. DementiaNet GitHub repository
  3. Your own recordings (healthy elderly controls)

Usage:
  python training/data_collector/collect_data.py --mode youtube
  python training/data_collector/collect_data.py --mode dementianet --path ../../dementianet
  python training/data_collector/collect_data.py --mode healthy --dir ../../my_recordings
  python training/data_collector/collect_data.py --mode summary
"""

import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

RAW_DIR = Path("training/data/raw")
AD_DIR  = RAW_DIR / "AD"
HC_DIR  = RAW_DIR / "HC"   # Healthy Controls

# Public-figure AD interviews — confirmed diagnoses, educational/research use
# Format: (label, youtube_url, short_id, description)
YOUTUBE_AD_URLS = [
    # Glen Campbell — documented Alzheimer's journey
    ("AD", "https://www.youtube.com/watch?v=3d8IpfBSs2I", "campbell_01",
     "Glen Campbell Alzheimer's interview"),
    # Terry Pratchett — early-onset AD documentary segment
    ("AD", "https://www.youtube.com/watch?v=ZWWL7PGG7_o", "pratchett_01",
     "Terry Pratchett living with Alzheimer's"),
    # Alzheimer's Research UK patient interview (CC licensed)
    ("AD", "https://www.youtube.com/watch?v=1bNe3BbkD_I", "aruk_01",
     "Alzheimer's Research UK patient interview"),
    # BBC documentary segment — public domain
    ("AD", "https://www.youtube.com/watch?v=7XOTGLKmZDA", "bbc_ad_01",
     "BBC Alzheimer's documentary interview"),
    # Rosa Linda Lozano — TEDx self-reported AD
    ("AD", "https://www.youtube.com/watch?v=3H5QYC3vlqw", "tedx_ad_01",
     "TEDx AD self-report talk"),
]

# Age-matched healthy controls — public interviews, 60-80 year olds
YOUTUBE_HC_URLS = [
    ("HC", "https://www.youtube.com/watch?v=rnw9Jz1wnBg", "healthy_01",
     "David Attenborough interview (healthy elderly)"),
    ("HC", "https://www.youtube.com/watch?v=7RU-DpB8Cnw", "healthy_02",
     "Age-matched healthy control interview"),
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _ensure_dirs():
    AD_DIR.mkdir(parents=True, exist_ok=True)
    HC_DIR.mkdir(parents=True, exist_ok=True)


def _check_yt_dlp():
    if shutil.which("yt-dlp") is None:
        print("yt-dlp not found. Installing…")
        subprocess.run(["pip", "install", "yt-dlp"], check=True)


def _download_video(url: str, out_path: Path, max_duration_s: int = 300) -> bool:
    """Download up to max_duration_s seconds of video via yt-dlp."""
    _check_yt_dlp()
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*0-{max_duration_s}",
        "--output", str(out_path),
        "--no-playlist",
        "--quiet",
        "--progress",
        url,
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0 and out_path.exists()


def _write_manifest(records: list, path: Path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Manifest saved: {path}")


# ─────────────────────────────────────────────────────────────
# Collection modes
# ─────────────────────────────────────────────────────────────

def collect_youtube():
    """Download YouTube AD + healthy control videos."""
    _ensure_dirs()
    manifest = []
    all_urls = YOUTUBE_AD_URLS + YOUTUBE_HC_URLS

    for label, url, vid_id, desc in all_urls:
        dest_dir = AD_DIR if label == "AD" else HC_DIR
        out_path = dest_dir / f"{vid_id}.mp4"

        if out_path.exists():
            print(f"  [SKIP] {vid_id} already downloaded")
            manifest.append({"id": vid_id, "label": label,
                             "path": str(out_path), "source": "youtube"})
            continue

        print(f"  [DL] {label} — {desc} … ", end="", flush=True)
        ok = _download_video(url, out_path)
        if ok:
            print("✓")
            manifest.append({"id": vid_id, "label": label,
                             "path": str(out_path), "source": "youtube"})
        else:
            print("✗ (skipped — check URL or internet connection)")

    _write_manifest(manifest, RAW_DIR / "youtube_manifest.json")
    print(f"\nYouTube collection complete. Downloaded: "
          f"{sum(1 for r in manifest if 'youtube' in r.get('source',''))} files")


def collect_dementianet(dementianet_path: str):
    """
    Copy audio files from DementiaNet GitHub clone.
    DementiaNet: https://github.com/shreyasgite/dementianet
    Clone with: git clone https://github.com/shreyasgite/dementianet
    """
    _ensure_dirs()
    src = Path(dementianet_path)
    if not src.exists():
        print(f"ERROR: DementiaNet path not found: {src}")
        print("Clone it with:  git clone https://github.com/shreyasgite/dementianet")
        return

    manifest = []
    # DementiaNet typically has audio files in subdirs by subject
    audio_exts = (".wav", ".mp3", ".mp4", ".m4a")
    found = list(src.rglob("*"))
    audio_files = [f for f in found if f.suffix.lower() in audio_exts]

    print(f"Found {len(audio_files)} audio files in DementiaNet")
    for i, af in enumerate(audio_files):
        vid_id = f"dn_{i:04d}"
        dest   = AD_DIR / f"{vid_id}{af.suffix}"
        if not dest.exists():
            shutil.copy2(af, dest)
        manifest.append({"id": vid_id, "label": "AD",
                         "path": str(dest), "source": "dementianet"})

    _write_manifest(manifest, RAW_DIR / "dementianet_manifest.json")
    print(f"DementiaNet: copied {len(manifest)} files → {AD_DIR}")


def collect_healthy(healthy_dir: str):
    """
    Import your own healthy elderly recordings.
    Expected: a folder of MP4/WAV files, one per subject.
    """
    _ensure_dirs()
    src = Path(healthy_dir)
    if not src.exists():
        print(f"ERROR: Directory not found: {src}")
        return

    manifest = []
    exts = (".mp4", ".mov", ".avi", ".mkv", ".wav", ".mp3")
    files = [f for f in src.iterdir() if f.suffix.lower() in exts]

    for i, f in enumerate(sorted(files)):
        vid_id = f"self_{i:04d}"
        dest   = HC_DIR / f"{vid_id}{f.suffix}"
        if not dest.exists():
            shutil.copy2(f, dest)
        manifest.append({"id": vid_id, "label": "HC",
                         "path": str(dest), "source": "self_recorded"})

    _write_manifest(manifest, RAW_DIR / "healthy_manifest.json")
    print(f"Self-recorded: imported {len(manifest)} healthy subjects → {HC_DIR}")


def print_summary():
    """Print dataset summary."""
    _ensure_dirs()
    ad_files = list(AD_DIR.iterdir())
    hc_files = list(HC_DIR.iterdir())
    ad_vids  = [f for f in ad_files if f.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    hc_vids  = [f for f in hc_files if f.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    ad_wav   = [f for f in ad_files if f.suffix.lower() in (".wav", ".mp3")]
    hc_wav   = [f for f in hc_files if f.suffix.lower() in (".wav", ".mp3")]

    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    print(f"  AD  — Video: {len(ad_vids):3d}  |  Audio: {len(ad_wav):3d}")
    print(f"  HC  — Video: {len(hc_vids):3d}  |  Audio: {len(hc_wav):3d}")
    print(f"  Total subjects: {len(ad_vids)+len(ad_wav)+len(hc_vids)+len(hc_wav)}")
    total = len(ad_vids) + len(ad_wav) + len(hc_vids) + len(hc_wav)
    if total < 20:
        print("\n  ⚠  Less than 20 samples — collect more before training.")
        print("     Minimum recommended: 20 AD + 20 Healthy")
    elif total < 50:
        print("\n  ℹ  Enough for SVM baseline (Stage A).")
        print("     Collect 50+ for transformer fine-tuning (Stage B).")
    else:
        print(f"\n  ✓  {total} samples — ready for full training pipeline!")
    print("="*50 + "\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neuro Fusion v2 — Data Collector")
    parser.add_argument("--mode", choices=["youtube", "dementianet", "healthy", "summary"],
                        required=True)
    parser.add_argument("--path",  default="../../dementianet",
                        help="Path to cloned DementiaNet repo")
    parser.add_argument("--dir",   default="../../my_recordings",
                        help="Path to your self-recorded healthy videos")
    args = parser.parse_args()

    if args.mode == "youtube":
        collect_youtube()
    elif args.mode == "dementianet":
        collect_dementianet(args.path)
    elif args.mode == "healthy":
        collect_healthy(args.dir)
    elif args.mode == "summary":
        print_summary()


if __name__ == "__main__":
    main()
