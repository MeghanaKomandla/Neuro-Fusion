# Neuro Fusion v2 — Complete Step by Step Guide

## Your PC Specs (Confirmed)
- CPU: Intel i3-1005G1, 3.8GB RAM, no GPU
- All inference runs locally
- All training runs on Google Colab (free)

---

## PHASE 1 — Setup on Your PC

### Step 1 — Extract the zip
Extract `neuro_fusion_v2_final.zip` to your project folder.

### Step 2 — Open terminal in VS Code
Press Ctrl+` in VS Code

### Step 3 — Navigate to project
```
cd "C:\Users\Pooja\Downloads\M.Tech\Major Project\neuro_fusion_v2_final"
```

### Step 4 — Create and activate venv
```
py -m venv venv
venv\Scripts\activate
```
You should see (venv) at the start of the line.

### Step 5 — Install packages
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
This takes 5-10 minutes. Wait for it.

### Step 6 — Set Gemini API key (for chatbot)
1. Go to aistudio.google.com
2. Sign in with Google account
3. Click "Get API Key" → Create API key → Copy it
4. In your terminal:
```
set GEMINI_API_KEY=paste_your_key_here
```

### Step 7 — Add research PDFs (for chatbot citations)
Create folder `knowledge_base/` and add these free PDFs:
- WHO Dementia report: who.int/news-room/fact-sheets/detail/dementia
- Alzheimer's Assoc report: alz.org (search "facts and figures PDF")
- Any Alzheimer's/Dementia research papers from PubMed

### Step 8 — Run the app
```
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Step 9 — Open browser
```
http://localhost:8000
```

The app works immediately with heuristic scoring.
For trained model weights, complete Phase 2 below.

---

## PHASE 2 — Train Models on Google Colab

### Step 1 — Open Colab
Go to colab.research.google.com
Runtime → Change runtime type → T4 GPU → Save

### Step 2 — Create a new notebook
Click "+ New notebook"

### Step 3 — Run COLAB_TRAINING.py
Copy each cell from COLAB_TRAINING.py and run them one by one.
The file is included in your project folder.

### Step 4 — What Colab will do
- Upload your healthy elderly videos
- Download DementiaNet data from GitHub
- Download YouTube AD interview videos
- Preprocess everything (extract Wav2Vec2 + MediaPipe features)
- Train AD model: Stage A (SVM) + Stage B (CrossModalTransformer)
- Train Dementia model (DementiaTransformer)
- Save checkpoints to Google Drive

### Step 5 — Download checkpoints
After training, Colab will download 3 files:
- cross_modal_v2.pt (AD transformer)
- stage_a_ensemble.pkl (AD SVM)
- dementia_transformer.pt (Dementia model)

### Step 6 — Paste checkpoints on your PC
Copy all 3 files into:
```
neuro_fusion_v2_final/checkpoints/
```

### Step 7 — Restart server
```
venv\Scripts\activate
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The app now loads trained weights automatically.
Confidence scores change from ~10% to ~60-85%.

---

## PHASE 3 — Using the App

### Upload a video
1. Open http://localhost:8000
2. Click "Upload Video" or drag and drop
3. Wait 5-15 minutes (Wav2Vec2 on CPU)
4. See results: AD score + Dementia score

### Use the AI chatbot
1. Click "AI Assistant" in the nav bar
2. Ask any question about Alzheimer's or Dementia
3. Get answers with citations from your PDFs

### View history
1. Click "History" in nav bar
2. See all past analysis results

---

## PHASE 4 — Collect Training Data

### Your existing data (already have)
- YouTube AD interview videos → training/data/raw/AD/
- Your healthy elderly recordings → training/data/raw/HC/

### DementiaNet data (free, instant)
```
git clone https://github.com/shreyasgite/dementianet
python training/data_collector/collect_data.py --mode dementianet --path ../dementianet
```

### Check your dataset
```
python training/data_collector/collect_data.py --mode summary
```

---

## Every Time You Open VS Code

```
cd "C:\Users\Pooja\Downloads\M.Tech\Major Project\neuro_fusion_v2_final"
venv\Scripts\activate
set GEMINI_API_KEY=your_key_here
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open browser: http://localhost:8000

---

## Project File Structure

```
neuro_fusion_v2_final/
├── app.py                    ← FastAPI routes (AD + Dementia + Chatbot)
├── model.py                  ← Alzheimer's detection (Wav2Vec2 + CMT)
├── model_dementia.py         ← Dementia detection (DementiaTransformer)
├── chatbot.py                ← RAG + Gemini chatbot
├── utils.py                  ← ffmpeg helpers
├── requirements.txt          ← all packages
├── train_dementia.py         ← dementia training script (run in Colab)
├── COLAB_TRAINING.py         ← complete Colab training guide
├── checkpoints/              ← paste trained weights here
│   ├── cross_modal_v2.pt
│   ├── stage_a_ensemble.pkl
│   └── dementia_transformer.pt
├── knowledge_base/           ← add research PDFs here
│   └── (add WHO, Alz PDFs)
├── training/
│   ├── data_collector/
│   │   └── collect_data.py
│   ├── preprocessor/
│   │   └── preprocess.py
│   ├── trainer/
│   │   └── train.py
│   └── evaluator/
│       └── evaluate.py
└── templates/
    ├── index.html            ← upload page
    ├── results.html          ← AD + Dementia results
    ├── chat.html             ← AI chatbot page
    └── history.html          ← past results
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| python not recognized | Use `py` instead of `python` |
| pip not recognized | Run `venv\Scripts\activate` first |
| MemoryError | Close Chrome, VS Code, WhatsApp before starting |
| Site cannot be reached | Server not running — run uvicorn command first |
| Internal server error | Paste error from terminal here |
| Chatbot says key not set | Run `set GEMINI_API_KEY=your_key` |
| Low confidence (10%) | Models not trained yet — complete Phase 2 |

---

## RAM Management

Close these before starting server:
- Chrome/Edge browser (free 500MB)
- WhatsApp desktop (free 200MB)
- Other VS Code windows (free 300MB)

After closing: ~1.5GB free → enough for Wav2Vec2 + app
