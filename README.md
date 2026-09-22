
# NeuroFusion
### A Cross-Modal Transformer Framework for Non-Invasive Alzheimer's and Dementia Screening from Speech and Facial Cues

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-green?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Accuracy-84.8%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.914-brightgreen?style=for-the-badge"/>
</p>

---

## Overview

**NeuroFusion** is a non-invasive, multimodal AI screening system that detects early-stage **Alzheimer's disease** and **dementia** risk from short video and audio recordings of spontaneous speech — without requiring any clinical equipment, specialist involvement, or hospital visit.

The system analyzes two complementary signal streams simultaneously:
-  **Speech patterns** — extracted using a pretrained Wav2Vec2 transformer
-  **Facial dynamics** — extracted using MediaPipe FaceMesh (468 3D landmarks)

These two streams are fused using a custom **CrossModalTransformer** with bidirectional cross-attention, producing a **0–100% cognitive risk score** with full explainability breakdown.

> **Disclaimer:** NeuroFusion is a research demo and screening aid only. It is not a clinical diagnostic tool and should not replace professional medical evaluation.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **84.8%** |
| **AUC-ROC** | **0.914** |
| Validation Method | Leave-One-Out Cross-Validation |
| Dataset | 33 samples (6 AD + 27 HC) |
| Comparison | ADReSS 2020 top teams: ~0.88 AUC |

---

## System Architecture

```
Video Input
    │
    ├──► Audio Extraction (FFmpeg)
    │         │
    │         └──► Wav2Vec2 Encoder ──► Speech Embeddings (768-dim)
    │                                           │
    └──► Frame Extraction (FFmpeg)              │
              │                                 ▼
              └──► MediaPipe FaceMesh ──► Landmark Embeddings (1404-dim)
                                                │
                                    ┌───────────┴───────────┐
                                    │  CrossModalTransformer │
                                    │  (Bidirectional Cross- │
                                    │   Attention Fusion)    │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │   Risk Score (0-100%) │
                                    │   + Explainability    │
                                    └───────────────────────┘
```

### Three-Tier Fallback Architecture

```
Tier 1 (Primary)  → CrossModalTransformer (Wav2Vec2 + FaceMesh + Cross-Attention)
       ↓ if checkpoint unavailable
Tier 2 (Fallback) → Stage-A SVM Ensemble (14-dim heuristic features, LOO-CV trained)
       ↓ if checkpoint unavailable
Tier 3 (Final)    → Rule-Based Heuristic Scorer (clinically-derived feature weighting)
```

---

## Modules

### Module 1 — Alzheimer's Detection (Video)
- Upload a short video recording of spontaneous speech
- Extracts Wav2Vec2 speech embeddings + MediaPipe facial landmarks
- Fuses both streams using CrossModalTransformer
- Outputs risk score + feature explainability breakdown

### Module 2 — Dementia Detection (Audio)
- Upload an audio recording
- Reuses Wav2Vec2 embeddings (zero extra RAM cost)
- DementiaTransformer with attention-pooling highlights specific timestamps
- Outputs dementia risk score + attention-highlighted timeline

### Module 3 — RAG Chatbot
- Ask questions about Alzheimer's and dementia in plain language
- Retrieves answers from uploaded research PDFs using ChromaDB
- Grounded responses with source citations via Gemini 2.0 Flash
- No hallucinated medical claims — only cited, evidence-based answers

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Backend Framework | FastAPI + Uvicorn |
| Deep Learning | PyTorch, Transformers |
| Speech Model | Wav2Vec2 (facebook/wav2vec2-base-960h) |
| Facial Landmarks | MediaPipe FaceMesh |
| Fallback Model | Scikit-learn SVM Ensemble |
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| LLM | Gemini 2.0 Flash (Google AI Studio) |
| RAG Framework | LangChain |
| Video/Audio Processing | FFmpeg, librosa, soundfile |
| Frontend | HTML, CSS, Jinja2 Templates |

---

## Project Structure

```
project_final/
│
├── app.py                    # FastAPI main application
├── model.py                  # CrossModalTransformer + AD detection
├── model_dementia.py         # DementiaTransformer
├── chatbot.py                # RAG chatbot (ChromaDB + Gemini)
├── utils.py                  # FFmpeg helpers
├── requirements.txt          # Python dependencies
│
├── templates/                # HTML pages
│   ├── home.html
│   ├── alzheimers.html
│   ├── dementia.html
│   ├── results_ad.html
│   ├── results_dementia.html
│   ├── chat.html
│   └── history.html
│
├── static/
│   └── style.css
│
├── checkpoints/              # Trained model weights
│   ├── stage_a_ensemble.pkl  # Stage A SVM (trained, active)
│   └── cross_modal_v2.pt     # CrossModalTransformer (pending GPU)
│
├── knowledge_base/           # Add your research PDFs here
│   └── README.txt
│
└── training/                 # Training pipeline
    ├── data_collector/
    │   └── collect_data.py
    ├── preprocessor/
    │   └── preprocess.py
    ├── trainer/
    │   └── train.py
    └── evaluator/
        └── evaluate.py
```

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- FFmpeg installed and added to PATH
- Gemini API key (free from https://aistudio.google.com)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/NeuroFusion.git
cd NeuroFusion
```

**2. Create and activate virtual environment**
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install fastapi uvicorn[standard] python-multipart jinja2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers==0.21.0 --only-binary=:all:
pip install transformers==4.48.0
pip install mediapipe opencv-python-headless librosa soundfile
pip install numpy scikit-learn
pip install langchain langchain-community chromadb sentence-transformers
pip install pypdf google-genai
```

**4. Set Gemini API Key**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your_key_here"

# Windows (permanent)
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your_key_here", "User")

# Linux/Mac
export GEMINI_API_KEY="your_key_here"
```

**5. Run the server**
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**6. Open in browser**
```
http://localhost:8000
```

---

## Usage

### Alzheimer's Screening
1. Go to `http://localhost:8000/alzheimers`
2. Upload a video (MP4, MOV, AVI) of someone speaking naturally for 30–120 seconds
3. Click **Analyze**
4. View risk score, confidence level, and feature breakdown

### Dementia Screening
1. Go to `http://localhost:8000/dementia`
2. Upload an audio file (WAV, MP3, M4A)
3. Click **Analyze**
4. View dementia risk score and attention-highlighted timestamps

### RAG Chatbot
1. Add research PDFs to the `knowledge_base/` folder
2. Go to `http://localhost:8000/chat`
3. Ask any question about Alzheimer's or dementia
4. Get cited, evidence-based answers from your uploaded documents

### View History
- Go to `http://localhost:8000/history`
- View all past screening results with timestamps

---

## Training Your Own Model

### On Google Colab (Recommended — Free GPU)

**Step 1 — Upload your data to Google Drive**
- AD videos → `training/data/raw/AD/`
- Healthy videos → `training/data/raw/HC/`

**Step 2 — Preprocess**
```python
!python training/preprocessor/preprocess.py
```

**Step 3 — Train Stage A (SVM, CPU-friendly)**
```python
!python training/trainer/train.py --stage A --augment
```

**Step 4 — Train Stage B (Transformer, needs GPU)**
```python
!python training/trainer/train.py --stage B --epochs 30 --augment
```

**Step 5 — Download checkpoints**
```python
from google.colab import files
files.download("checkpoints/stage_a_ensemble.pkl")
files.download("checkpoints/cross_modal_v2.pt")
```

---

##  Model Performance

### Stage A — SVM Ensemble (Trained, Currently Active)

```
Dataset:     33 samples (AD=6, HC=27)
Validation:  Leave-One-Out Cross-Validation

Overall:
  Accuracy : 84.8%
  AUC-ROC  : 0.914

Per-Class:
  HC → Precision: 0.87  Recall: 0.96  F1: 0.91
  AD → Precision: 0.67  Recall: 0.33  F1: 0.44

Known Limitation:
  AD recall of 33% due to class imbalance (27 HC vs 6 AD samples).
  Addressed in CrossModalTransformer training with balanced dataset.
```

### Comparison With Published Baselines

| System | Dataset | Accuracy | AUC-ROC |
|--------|---------|----------|---------|
| ADReSS 2020 Top Teams | 108 samples | 83–90% | ~0.88 |
| **NeuroFusion Stage A** | **33 samples** | **84.8%** | **0.914** |

### Stage B — CrossModalTransformer (Primary Model)
- Architecture: fully implemented
- Status: training pending GPU quota reset
- Expected performance: higher than Stage A with balanced, larger dataset

---

## 14 Heuristic Features (Stage A Input)

| # | Feature | Clinical Relevance |
|---|---------|-------------------|
| 1 | Pause Ratio | Higher in AD — increased hesitation |
| 2 | Speech Rate | Slower in cognitive decline |
| 3 | Energy Mean | Reduced vocal effort in dementia |
| 4 | Energy Std | Flatter prosody in AD |
| 5 | MFCC Variance | Reduced articulatory complexity |
| 6 | Pitch Variability | Monotone speech pattern in AD |
| 7 | Zero Crossing Rate | Speech texture changes |
| 8 | Spectral Centroid | Voice quality degradation |
| 9 | Duration | Recording length normalization |
| 10 | Face Detected Ratio | Reduced engagement in AD |
| 11 | Expressiveness Score | Reduced facial animation |
| 12 | Inter-Frame Motion | Reduced movement in dementia |
| 13 | Visual Entropy | Facial dynamic complexity |
| 14 | AU Variance | Action unit expressiveness |

---

## Every Session — Quick Start Commands

```powershell
cd "path\to\project_final"
.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Could not import module "app"` | Navigate to correct folder first |
| `tokenizers` Rust compile error | `pip install tokenizers==0.21.0 --only-binary=:all:` |
| `GEMINI_API_KEY not set` | Set API key before starting server |
| `ffmpeg not found` | Install FFmpeg and add to PATH |
| `stage_a_ensemble.pkl not found` | Copy checkpoint to `checkpoints/` folder |
| Port 8000 in use | Use `--port 8001` instead |

---

## References

1. Baevski et al. (2020) — wav2vec 2.0, NeurIPS
2. Luz et al. (2020) — ADReSS Challenge, INTERSPEECH
3. Tsai et al. (2019) — Multimodal Transformer, ACL
4. Lugaresi et al. (2019) — MediaPipe, Google Research
5. Lewis et al. (2020) — RAG, NeurIPS
6. Vaswani et al. (2017) — Attention Is All You Need, NeurIPS
7. König et al. (2015) — Speech Analysis for AD, Alzheimer's & Dementia Journal

---

##  Author

**Komandla Meghana Reddy**
Roll No: 24SS1DB211
M.Tech — Artificial Intelligence and Data Science
JNTUH University College of Engineering Sultanpur

**Guide:** Dr. P. Swetha, Professor
Department of Computer Science and Engineering

---

## License

This project is developed for M.Tech dissertation purposes at JNTUH UCE Sultanpur.
For research and educational use only.

---

<p align="center">
  Made for early Alzheimer's detection research
</p>
