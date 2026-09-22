# ================================================================
# NEURO FUSION v2 — COMPLETE COLAB TRAINING SCRIPT
# ================================================================
# Run each cell one by one in Google Colab
# Runtime: T4 GPU (Runtime → Change runtime type → T4 GPU)
# ================================================================


# ── CELL 1 — Check GPU ──────────────────────────────────────
import torch
print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ── CELL 2 — Upload your project zip ────────────────────────
from google.colab import files
print("Select your zip file:")
files.upload()


# ── CELL 3 — Extract zip ────────────────────────────────────
import zipfile, os

# Find the uploaded zip
zips = [f for f in os.listdir('.') if f.endswith('.zip')]
print("Found:", zips)
z = zips[0]

with zipfile.ZipFile(z, 'r') as zf:
    zf.extractall('.')

# Find project folder
folders = [f for f in os.listdir('.') if os.path.isdir(f)
           and 'neuro' in f.lower()]
print("Project folders:", folders)

# Change into project folder
if folders:
    os.chdir(folders[0])
    print("Working directory:", os.getcwd())


# ── CELL 4 — Install packages ───────────────────────────────
import subprocess
subprocess.run(["pip", "install", "transformers==4.40.0",
                "mediapipe==0.10.35", "librosa", "soundfile",
                "opencv-python-headless", "scikit-learn", "yt-dlp",
                "-q"])
print("Packages installed")


# ── CELL 5 — Connect Google Drive (RECOMMENDED) ─────────────
# Saves your checkpoints permanently — never lost between sessions
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/neuro_fusion', exist_ok=True)
print("Google Drive connected. Checkpoints will save to:")
print("/content/drive/MyDrive/neuro_fusion/")


# ── CELL 6 — Upload your healthy elderly videos ─────────────
from google.colab import files
import shutil

print("Upload your healthy elderly recordings (MP4/MOV files):")
uploaded = files.upload()

os.makedirs("training/data/raw/HC", exist_ok=True)
for fname in uploaded:
    shutil.move(fname, f"training/data/raw/HC/{fname}")
    print(f"Moved: {fname} → training/data/raw/HC/")

print(f"\nHealthy videos ready: {len(uploaded)}")


# ── CELL 7 — Get DementiaNet data ───────────────────────────
# Clone DementiaNet GitHub repository
subprocess.run(["git", "clone",
                "https://github.com/shreyasgite/dementianet",
                "../dementianet"])

# Import AD audio files
subprocess.run(["python", "training/data_collector/collect_data.py",
                "--mode", "dementianet",
                "--path", "../dementianet"])

print("DementiaNet imported")


# ── CELL 8 — Download YouTube AD interviews ─────────────────
subprocess.run(["python", "training/data_collector/collect_data.py",
                "--mode", "youtube"])


# ── CELL 9 — Check dataset summary ─────────────────────────
subprocess.run(["python", "training/data_collector/collect_data.py",
                "--mode", "summary"])


# ── CELL 10 — Preprocess all data ───────────────────────────
# This extracts Wav2Vec2 embeddings + MediaPipe landmarks
# Takes ~1-2 min per video on Colab GPU
print("Starting preprocessing — this may take 1-2 hours...")
print("Do NOT close this tab.\n")

subprocess.run(["python", "training/preprocessor/preprocess.py"])


# ── CELL 11 — Train AD model Stage A (SVM, 5 min) ──────────
subprocess.run(["python", "training/trainer/train.py",
                "--stage", "A", "--augment"])


# ── CELL 12 — Train AD model Stage B (Transformer, 30 min) ─
subprocess.run(["python", "training/trainer/train.py",
                "--stage", "B", "--epochs", "30", "--augment"])


# ── CELL 13 — Train Dementia model (30 min) ─────────────────
subprocess.run(["python", "train_dementia.py", "--epochs", "40"])


# ── CELL 14 — Evaluate both models ──────────────────────────
subprocess.run(["python", "training/evaluator/evaluate.py"])


# ── CELL 15 — Save to Google Drive ──────────────────────────
import shutil

checkpoints = [
    "checkpoints/cross_modal_v2.pt",
    "checkpoints/stage_a_ensemble.pkl",
    "checkpoints/dementia_transformer.pt",
]

for ckpt in checkpoints:
    if os.path.exists(ckpt):
        dest = f"/content/drive/MyDrive/neuro_fusion/{os.path.basename(ckpt)}"
        shutil.copy(ckpt, dest)
        print(f"Saved to Drive: {dest}")
    else:
        print(f"NOT FOUND: {ckpt}")

print("\nAll checkpoints saved to Google Drive!")


# ── CELL 16 — Download to your PC ───────────────────────────
from google.colab import files

for ckpt in checkpoints:
    if os.path.exists(ckpt):
        files.download(ckpt)
        print(f"Downloaded: {ckpt}")

print("\nCopy these 3 files to your PC:")
print("  checkpoints/cross_modal_v2.pt")
print("  checkpoints/stage_a_ensemble.pkl")
print("  checkpoints/dementia_transformer.pt")
print("\nThen restart your server:")
print("  python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000")
