"""
preprocess_audio.py - Audio Preprocessing for Dementia Detection
Run BEFORE train_dementia.py

Usage:
  python preprocess_audio.py
"""

import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DEMENTIA_DIR   = Path("data/dementia")
NODEMENTIA_DIR = Path("data/nodementia")
OUTPUT_DIR     = Path("training/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE  = 16000
MAX_DURATION = 30
MODEL_NAME   = "facebook/wav2vec2-base"


def load_audio(path: Path, target_sr: int = SAMPLE_RATE):
    try:
        import librosa
        waveform, _ = librosa.load(str(path), sr=target_sr, mono=True)
        max_samples = MAX_DURATION * target_sr
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        if len(waveform) < target_sr:
            return None
        return waveform.astype("float32")
    except Exception as e:
        print(f"    Error loading {path.name}: {e}")
        return None


def extract_embeddings(waveform_np, processor, model, device):
    import torch
    try:
        inputs = processor(
            waveform_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            emb = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return emb.astype(np.float32)
    except Exception as e:
        print(f"    Embedding error: {e}")
        return None


def process_folder(folder, label, processor, model, device):
    wav_files = list(Path(folder).rglob("*.wav"))
    tag = "DEMENTIA" if label == 1 else "NO-DEMENTIA"
    print(f"\n  Processing {tag}: {len(wav_files)} files")
    saved = skipped = 0
    for i, wav_path in enumerate(wav_files):
        out_path = OUTPUT_DIR / f"{wav_path.stem}_label{label}.npz"
        if out_path.exists():
            saved += 1
            continue
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    [{i+1}/{len(wav_files)}] {wav_path.name}")
        waveform = load_audio(wav_path)
        if waveform is None:
            skipped += 1
            continue
        emb = extract_embeddings(waveform, processor, model, device)
        if emb is None:
            skipped += 1
            continue
        np.savez_compressed(
            str(out_path),
            speech_emb=emb,
            label=np.array(label, dtype=np.int64),
            filename=str(wav_path.name)
        )
        saved += 1
    print(f"    Saved: {saved}  |  Skipped: {skipped}")
    return saved


def main():
    import torch

    print("=" * 55)
    print("  DEMENTIA AUDIO PREPROCESSING")
    print("=" * 55)

    dem_files = list(DEMENTIA_DIR.rglob("*.wav"))
    nod_files = list(NODEMENTIA_DIR.rglob("*.wav"))
    print(f"\n  Dementia files:    {len(dem_files)}")
    print(f"  No-dementia files: {len(nod_files)}")

    if not dem_files or not nod_files:
        print("\n  ERROR: Audio files not found!")
        return

    try:
        import librosa
        print(f"  librosa: OK ({librosa.__version__})")
    except ImportError:
        print("\n  ERROR: pip install librosa soundfile")
        return

    print(f"\n  Loading Wav2Vec2: {MODEL_NAME}")
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model     = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model     = model.to(device).eval()
        print(f"  Device: {device}  Model: OK")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("\n  Fix: pip install transformers==4.40.0 librosa soundfile")
        return

    print("\n  Sanity check on 1 file...")
    wf = load_audio(dem_files[0])
    if wf is None:
        print("  FAILED to load audio")
        return
    emb = extract_embeddings(wf, processor, model, device)
    if emb is None:
        print("  FAILED to extract embeddings")
        return
    print(f"  OK — shape: {emb.shape}")

    process_folder(DEMENTIA_DIR,   label=1,
                   processor=processor, model=model, device=device)
    process_folder(NODEMENTIA_DIR, label=0,
                   processor=processor, model=model, device=device)

    total = len(list(OUTPUT_DIR.glob("*.npz")))
    print(f"\n{'='*55}")
    print(f"  DONE — {total} .npz files saved in {OUTPUT_DIR}")
    print(f"  Next: python train_dementia.py --epochs 40")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
    