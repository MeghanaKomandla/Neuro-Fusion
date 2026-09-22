"""
model.py — Alzheimer's Detection Engine
========================================
Wav2Vec2 + MediaPipe FaceMesh + CrossModalTransformer

Three-tier fallback:
  Tier 1: torch + transformers + mediapipe → full transformer
  Tier 2: torch + transformers only        → Wav2Vec2 + heuristic
  Tier 3: numpy only                       → heuristic scoring
"""

import os
import glob
import math
import wave
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TORCH_AVAILABLE        = False
TRANSFORMERS_AVAILABLE = False
LIBROSA_AVAILABLE      = False
MP_AVAILABLE           = False
CV2_AVAILABLE          = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch — device: {DEVICE}")
except ImportError:
    DEVICE = None

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    pass

try:
    import mediapipe as mp
    _ = mp.solutions.face_mesh
    MP_AVAILABLE = True
except (ImportError, AttributeError):
    pass

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass

_wav2vec2_processor = None
_wav2vec2_model     = None
_mp_face_mesh       = None
_cross_modal_model  = None


if TORCH_AVAILABLE:
    class CrossModalTransformer(nn.Module):
        def __init__(self, speech_dim=768, visual_dim=1404,
                     d_model=256, nhead=8, dropout=0.1):
            super().__init__()
            self.speech_proj = nn.Sequential(
                nn.Linear(speech_dim, d_model), nn.LayerNorm(d_model), nn.ReLU())
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_dim, d_model), nn.LayerNorm(d_model), nn.ReLU())
            self.cross_attn_s2v = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_attn_v2s = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.self_attn_s = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            self.self_attn_v = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)
            fused = d_model * 2
            self.ffn  = nn.Sequential(
                nn.Linear(fused, fused * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(fused * 4, fused))
            self.norm = nn.LayerNorm(fused)
            self.classifier = nn.Sequential(
                nn.Linear(fused, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 2))
            self.risk_head = nn.Sequential(
                nn.Linear(fused, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid())

        def forward(self, s, v):
            s = self.speech_proj(s)
            v = self.visual_proj(v)
            sf, _ = self.cross_attn_s2v(s, v, v)
            vf, _ = self.cross_attn_v2s(v, s, s)
            sf = s + sf
            vf = v + vf
            sf, _ = self.self_attn_s(sf, sf, sf)
            vf, _ = self.self_attn_v(vf, vf, vf)
            fused = torch.cat([sf.mean(1), vf.mean(1)], dim=-1)
            fused = self.norm(fused + self.ffn(fused))
            return self.classifier(fused), self.risk_head(fused)


def _get_wav2vec2():
    global _wav2vec2_processor, _wav2vec2_model
    if _wav2vec2_processor is None and TRANSFORMERS_AVAILABLE:
        logger.info("Loading Wav2Vec2…")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")
        _wav2vec2_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h").to(DEVICE).eval()
        logger.info(f"Wav2Vec2 loaded on {DEVICE}")
    return _wav2vec2_processor, _wav2vec2_model


def _get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None and MP_AVAILABLE:
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.4)
    return _mp_face_mesh


def _get_cross_modal_model():
    global _cross_modal_model
    if _cross_modal_model is None and TORCH_AVAILABLE:
        _cross_modal_model = CrossModalTransformer().to(DEVICE)
        ckpt = Path("checkpoints/cross_modal_v2.pt")
        if ckpt.exists():
            _cross_modal_model.load_state_dict(
                torch.load(str(ckpt), map_location=DEVICE))
            logger.info("AD model: loaded trained weights")
        else:
            logger.warning("AD model: no checkpoint — using random weights")
        _cross_modal_model.eval()
    return _cross_modal_model


def _load_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "r") as wf:
        n, sr, ch, sw = (wf.getnframes(), wf.getframerate(),
                         wf.getnchannels(), wf.getsampwidth())
        raw = wf.readframes(n)
    if sw == 2:
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        y = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        y = y[::ch]
    return y, sr


def _extract_wav2vec2_embeddings(audio_path: str) -> Optional["torch.Tensor"]:
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        processor, model = _get_wav2vec2()
        y, sr = _load_wav(audio_path)
        if sr != 16000:
            if LIBROSA_AVAILABLE:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                idx = np.round(np.arange(0, len(y), sr/16000)).astype(int)
                y   = y[idx[idx < len(y)]]
        y = y[:30 * 16000]
        inputs = processor(y, sampling_rate=16000,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(inputs.input_values.to(DEVICE))
        return out.last_hidden_state
    except Exception as e:
        logger.warning(f"Wav2Vec2 failed: {e}")
        return None


def _extract_landmarks_mediapipe(frames_dir: str) -> Optional["torch.Tensor"]:
    if not MP_AVAILABLE or not CV2_AVAILABLE:
        return None
    try:
        fm    = _get_face_mesh()
        files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not files:
            return None
        step    = max(1, len(files) // 40)
        sampled = files[::step][:40]
        seq     = []
        for fp in sampled:
            img = cv2.imread(fp)
            if img is None:
                continue
            res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lms    = res.multi_face_landmarks[0].landmark
                coords = np.array([[l.x, l.y, l.z] for l in lms],
                                   dtype=np.float32).flatten()
                seq.append(coords)
        if len(seq) < 2:
            return None
        arr = np.stack(seq)
        T, D = arr.shape
        if D < 1404:
            arr = np.pad(arr, ((0,0),(0,1404-D)))
        else:
            arr = arr[:, :1404]
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return t.to(DEVICE) if DEVICE else t
    except Exception as e:
        logger.warning(f"FaceMesh failed: {e}")
        return None


def _frame_signal(y, fsz, hop):
    n = (len(y) - fsz) // hop + 1
    if n <= 0:
        return np.zeros((1, fsz))
    idx = np.arange(fsz)[None,:] + np.arange(n)[:,None]*hop
    return y[idx]


def _extract_heuristic_audio_features(audio_path: str) -> dict:
    default = dict(duration_sec=0.0, pause_ratio=0.0, speech_rate_proxy=0.0,
                   energy_mean=0.0, energy_std=0.0, pitch_variability=0.0,
                   mfcc_variance=0.0, zero_crossing_rate=0.0, spectral_centroid=0.0)
    if not os.path.exists(audio_path):
        return default
    try:
        if LIBROSA_AVAILABLE:
            y, sr    = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(y) / sr
            rms      = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
            thresh   = 0.04 * (rms.max() if rms.max() > 0 else 1e-4)
            silent   = rms < thresh
            pause_r  = float(silent.mean())
            bursts   = np.where(np.diff(silent.astype(int)) == -1)[0]
            srate    = len(bursts) / max(duration, 1.0)
            e_mean   = float(rms.mean())
            e_std    = float(rms.std())
            mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
            try:
                f0    = librosa.yin(y, fmin=60, fmax=400,
                                    frame_length=2048, hop_length=512)
                voiced = f0[f0 > 0]
                pvar   = float(np.std(voiced)) if len(voiced) > 1 else 0.0
            except Exception:
                pvar = 0.0
            zcr      = float(librosa.feature.zero_crossing_rate(y).mean())
            sc       = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        else:
            y, sr    = _load_wav(audio_path)
            duration = len(y) / max(sr, 1)
            amp      = np.abs(y)
            thresh   = 0.04 * amp.max() if amp.max() > 0 else 1e-4
            silent   = amp < thresh
            pause_r  = float(silent.mean())
            bursts   = np.where(np.diff(silent.astype(int)) == -1)[0]
            srate    = len(bursts) / max(duration, 1.0)
            fsz      = int(0.025 * sr)
            hop      = int(0.010 * sr)
            frames   = _frame_signal(y, fsz, hop)
            energies = np.mean(frames ** 2, axis=1)
            e_mean   = float(energies.mean())
            e_std    = float(energies.std())
            pvar     = float(np.std(np.diff(energies))) if len(energies)>2 else 0.0
            fft_s    = max(512, 2**math.ceil(math.log2(max(fsz,2))))
            ls       = np.log(np.abs(np.fft.rfft(frames, n=fft_s))**2 + 1e-10)
            edges    = np.linspace(0, ls.shape[1], 14, dtype=int)
            mfcc_np  = np.array([ls[:,edges[i]:edges[i+1]].mean(1)
                                  for i in range(13)]).T
            mfcc_var = float(np.mean(np.var(mfcc_np, axis=0)))
            zcr      = float(np.mean(np.abs(np.diff(np.sign(y)))))
            chunk    = y[:1024] if len(y)>=1024 else y
            spec     = np.abs(np.fft.rfft(chunk, n=1024))
            freqs    = np.fft.rfftfreq(1024, 1.0/sr)
            sc       = float(np.sum(freqs*spec)/(spec.sum()+1e-10))
        return dict(duration_sec=round(float(duration),2),
                    pause_ratio=round(float(pause_r),3),
                    speech_rate_proxy=round(float(srate),2),
                    energy_mean=round(float(e_mean),6),
                    energy_std=round(float(e_std),6),
                    pitch_variability=round(float(pvar),4),
                    mfcc_variance=round(float(mfcc_var),4),
                    zero_crossing_rate=round(float(zcr),4),
                    spectral_centroid=round(float(sc),1))
    except Exception as e:
        default["_error"] = str(e)
        return default


def _extract_heuristic_visual_features(frames_dir: str) -> dict:
    default = dict(n_frames=0, face_detected_ratio=0.0,
                   mean_brightness=0.0, brightness_std=0.0,
                   inter_frame_motion=0.0, expressiveness_score=0.0,
                   visual_entropy=0.0, au_variance=0.0)
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    n = len(files)
    if n == 0:
        return default
    default["n_frames"] = n
    step   = max(1, n // 30)
    sample = files[::step][:30]
    if CV2_AVAILABLE:
        br = []
        fc = 0
        for fp in sample:
            img = cv2.imread(fp)
            if img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            br.append(float(g.mean()))
            if g.std() > 15:
                fc += 1
        if not br:
            return default
        b = np.array(br)
        mean_b = float(b.mean())
        motion = float(np.mean(np.abs(np.diff(b))) / (mean_b + 1.0))
        expr   = min(float(b.std()) / (mean_b + 1.0), 1.0)
        h, _   = np.histogram(b, bins=32, range=(0,256))
        hn     = h / (h.sum() + 1e-10)
        ent    = float(-np.sum(hn * np.log2(hn + 1e-10)))
        default.update(face_detected_ratio=round(fc/max(len(br),1),3),
                       mean_brightness=round(mean_b,1),
                       brightness_std=round(float(b.std()),1),
                       inter_frame_motion=round(motion,4),
                       expressiveness_score=round(expr,4),
                       visual_entropy=round(ent,3),
                       au_variance=round(float(np.var(b)),3))
    else:
        sizes = []
        for fp in sample:
            try:
                sizes.append(os.path.getsize(fp))
            except Exception:
                pass
        if not sizes:
            return default
        s = np.array(sizes, dtype=float)
        ms = float(s.mean())
        motion = float(np.mean(np.abs(np.diff(s))) / (ms + 1.0))
        expr   = min(float(s.std()) / (ms + 1.0), 1.0)
        h, _   = np.histogram(s, bins=32)
        hn     = h / (h.sum() + 1e-10)
        ent    = float(-np.sum(hn * np.log2(hn + 1e-10)))
        default.update(face_detected_ratio=round(float(np.mean(s>8000)),3),
                       mean_brightness=round(ms,1),
                       brightness_std=round(float(s.std()),1),
                       inter_frame_motion=round(motion,4),
                       expressiveness_score=round(expr,4),
                       visual_entropy=round(ent,3),
                       au_variance=round(float(np.var(s)),3))
    return default


def _extract_visual_features_mediapipe(frames_dir: str) -> dict:
    base = _extract_heuristic_visual_features(frames_dir)
    if not MP_AVAILABLE or not CV2_AVAILABLE:
        return base
    try:
        fm    = _get_face_mesh()
        files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not files:
            return base
        step    = max(1, len(files) // 40)
        sampled = files[::step][:40]
        seq     = []
        for fp in sampled:
            img = cv2.imread(fp)
            if img is None:
                continue
            res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lms    = res.multi_face_landmarks[0].landmark
                coords = np.array([[l.x,l.y,l.z] for l in lms],
                                   dtype=np.float32).flatten()
                seq.append(coords)
        if len(seq) < 2:
            return base
        arr = np.stack(seq)
        T   = arr.shape[0]
        n_lm = arr.shape[1] // 3
        diffs  = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        motion = float(diffs.mean())
        expr   = float(np.std(diffs))
        pv     = np.var(arr, axis=1)
        h, _   = np.histogram(pv, bins=32)
        hn     = h / (h.sum() + 1e-10)
        ent    = float(-np.sum(hn * np.log2(hn + 1e-10)))
        au_raw = [70,105,61,291,152,33,263,1,13,14]
        ki     = [i for i in au_raw if i < n_lm]
        kc     = arr.reshape(T, n_lm, 3)[:, ki, :2]
        base.update(inter_frame_motion=round(motion,5),
                    expressiveness_score=round(expr,5),
                    visual_entropy=round(ent,3),
                    au_variance=round(float(np.var(kc)),5),
                    face_detected_ratio=round(len(seq)/max(len(sampled),1),3))
        return base
    except Exception as e:
        logger.warning(f"MediaPipe heuristic failed: {e}")
        return base


def _compute_risk_score_heuristic(audio: dict, visual: dict) -> Tuple[float, dict]:
    c = {}
    c["pause_ratio"]       = min(audio.get("pause_ratio",0)/0.55, 1.0)
    e = audio.get("energy_std",0)
    c["energy_dynamics"]   = (1.0-min(math.log1p(e*1e5)/12.0,1.0)) if e>0 else 0.75
    mv = audio.get("mfcc_variance",0)
    c["mfcc_richness"]     = (1.0-min(mv/8.0,1.0)) if mv>0 else 0.7
    c["speech_rate"]       = 1.0-min(audio.get("speech_rate_proxy",0)/6.0,1.0)
    c["zcr_articulation"]  = 1.0-min(audio.get("zero_crossing_rate",0)/0.25,1.0)
    c["pitch_monotonicity"]= 1.0-min(audio.get("pitch_variability",0)/50.0,1.0)
    ss = (c["pause_ratio"]*0.30 + c["energy_dynamics"]*0.18 +
          c["mfcc_richness"]*0.18 + c["speech_rate"]*0.14 +
          c["zcr_articulation"]*0.10 + c["pitch_monotonicity"]*0.10)
    c["expressiveness"]  = 1.0-min(visual.get("expressiveness_score",0)*4.0,1.0)
    c["face_visibility"] = 1.0-visual.get("face_detected_ratio",0)*0.5
    c["visual_entropy"]  = 1.0-min(visual.get("visual_entropy",0)/5.0,1.0)
    c["au_variance"]     = 1.0-min(visual.get("au_variance",0)*100.0,1.0)
    vs = (c["expressiveness"]*0.40 + c["face_visibility"]*0.20 +
          c["visual_entropy"]*0.20 + c["au_variance"]*0.20)
    p = ss*0.65 + vs*0.35
    return float(np.clip(p,0.03,0.97)), {k: round(v,3) for k,v in c.items()}


def _fuse_with_transformer(speech_emb, visual_emb) -> Tuple[float, dict]:
    model = _get_cross_modal_model()
    if model is None:
        return 0.5, {}
    try:
        with torch.no_grad():
            logits, risk = model(speech_emb, visual_emb)
            probs = torch.softmax(logits, dim=-1)
            p_cls  = float(probs[0,1].cpu())
            p_risk = float(risk[0,0].cpu())
        p = float(np.clip(0.5*p_cls+0.5*p_risk, 0.03, 0.97))
        return p, {"transformer_cls": round(p_cls,3),
                   "transformer_risk": round(p_risk,3)}
    except Exception as e:
        logger.warning(f"Transformer fusion failed: {e}")
        return 0.5, {}


def _compute_final_risk(audio_feats, visual_feats, speech_emb, visual_emb):
    h_score, h_contrib = _compute_risk_score_heuristic(audio_feats, visual_feats)
    if TORCH_AVAILABLE and speech_emb is not None and visual_emb is not None:
        t_score, t_contrib = _fuse_with_transformer(speech_emb, visual_emb)
        p_ad   = float(np.clip(0.60*t_score+0.40*h_score, 0.03, 0.97))
        contrib = {**h_contrib, **t_contrib}
        method  = "cross_modal_transformer"
    elif TORCH_AVAILABLE and speech_emb is not None:
        with torch.no_grad():
            pooled = speech_emb.mean(1).squeeze(0).cpu().numpy()
        proxy  = float(np.clip(np.mean(np.abs(pooled[:8]))*2, 0, 1))
        p_ad   = float(np.clip(0.50*proxy+0.50*h_score, 0.03, 0.97))
        contrib = {**h_contrib, "wav2vec2_proxy": round(proxy,3)}
        method  = "wav2vec2_heuristic"
    else:
        p_ad, contrib, method = h_score, h_contrib, "heuristic_only"
    return p_ad, contrib, method


def _risk_level(s: float) -> str:
    if s <= 30: return "Low"
    if s <= 55: return "Moderate"
    if s <= 75: return "High"
    return "Very High"


# ─────────────────────────────────────────────────────────────
# Stage A SVM Ensemble (trained baseline, lightweight reference)
# ─────────────────────────────────────────────────────────────

_stage_a_pipeline = None

def _get_stage_a_model():
    """Load the trained SVM ensemble checkpoint, if present."""
    global _stage_a_pipeline
    if _stage_a_pipeline is None:
        ckpt = Path("checkpoints/stage_a_ensemble.pkl")
        if ckpt.exists():
            try:
                import pickle
                with open(ckpt, "rb") as f:
                    data = pickle.load(f)
                _stage_a_pipeline = data["pipeline"]
                logger.info(
                    f"Stage A SVM loaded — trained on {data.get('n_samples','?')} "
                    f"samples, LOO accuracy {data.get('loo_accuracy',0)*100:.1f}%"
                )
            except Exception as e:
                logger.warning(f"Stage A SVM load failed: {e}")
                _stage_a_pipeline = False
        else:
            _stage_a_pipeline = False
    return _stage_a_pipeline if _stage_a_pipeline is not False else None


def _predict_stage_a(audio_feats: dict, visual_feats: dict) -> Optional[dict]:
    """
    Run the trained Stage A SVM ensemble on the 14-dim heuristic
    feature vector. Returns None if no checkpoint is available.
    """
    pipe = _get_stage_a_model()
    if pipe is None:
        return None
    try:
        feat_vector = np.array([[
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
        ]], dtype=np.float32)
        proba = float(pipe.predict_proba(feat_vector)[0][1])
        return {
            "stage_a_risk_percent": round(proba * 100, 1),
            "stage_a_level":        _risk_level(proba * 100),
        }
    except Exception as e:
        logger.warning(f"Stage A SVM prediction failed: {e}")
        return None


def predict_from_audio_frames(audio_path: str, frames_dir: str) -> dict:
    speech_emb   = _extract_wav2vec2_embeddings(audio_path)
    visual_emb   = _extract_landmarks_mediapipe(frames_dir)
    audio_feats  = _extract_heuristic_audio_features(audio_path)
    visual_feats = (_extract_visual_features_mediapipe(frames_dir)
                    if MP_AVAILABLE and CV2_AVAILABLE
                    else _extract_heuristic_visual_features(frames_dir))
    p_ad, contrib, method = _compute_final_risk(
        audio_feats, visual_feats, speech_emb, visual_emb)

    # Stage A SVM — trained baseline reference (if checkpoint exists)
    stage_a = _predict_stage_a(audio_feats, visual_feats)
    if stage_a is not None:
        # If the cross-modal transformer has no trained checkpoint yet,
        # prefer the trained Stage A SVM result as the primary score —
        # it is a real trained model, vs. random transformer weights.
        ckpt_exists = Path("checkpoints/cross_modal_v2.pt").exists()
        if not ckpt_exists:
            p_ad   = stage_a["stage_a_risk_percent"] / 100.0
            method = "stage_a_svm_ensemble"

    risk_pct   = round(p_ad * 100, 1)
    confidence = round(abs(p_ad - 0.5) * 200, 1)
    notes = []
    if audio_feats["duration_sec"] < 5:
        notes.append("Audio very short — results may be unreliable")
    if audio_feats["pause_ratio"] > 0.45:
        notes.append("High pause ratio detected")
    if visual_feats["face_detected_ratio"] < 0.4:
        notes.append("Low face visibility")
    notes.append(f"Method: {method}")
    notes.append("Research demo — not a clinical diagnosis")
    return dict(
        prediction=("Alzheimer's Indicators Detected"
                    if p_ad >= 0.50 else "No Significant Risk Detected"),
        risk_score_percent=risk_pct,
        risk_level=_risk_level(risk_pct),
        confidence_percent=confidence,
        inference_method=method,
        stage_a_risk_percent=stage_a["stage_a_risk_percent"] if stage_a else None,
        stage_a_level=stage_a["stage_a_level"] if stage_a else None,
        speech_features=dict(
            Duration=f"{audio_feats['duration_sec']}s",
            PauseRatio=f"{round(audio_feats['pause_ratio']*100,1)}%",
            SpeechRate=f"{audio_feats['speech_rate_proxy']} bursts/s",
            MFCCVariance=str(audio_feats["mfcc_variance"]),
            EnergyStd=str(audio_feats["energy_std"]),
            PitchVariability=str(audio_feats["pitch_variability"]),
            ZCR=str(audio_feats["zero_crossing_rate"]),
        ),
        visual_features=dict(
            FramesAnalyzed=str(visual_feats["n_frames"]),
            FaceDetection=f"{round(visual_feats['face_detected_ratio']*100,1)}%",
            Expressiveness=str(visual_feats["expressiveness_score"]),
            InterFrameMotion=str(visual_feats["inter_frame_motion"]),
            VisualEntropy=str(visual_feats["visual_entropy"]),
        ),
        explainability={k: round(v*100,1) for k,v in contrib.items()},
        notes=" | ".join(notes),
        _audio_feats=audio_feats,
        _visual_feats=visual_feats,
        _speech_emb=speech_emb,
    )
