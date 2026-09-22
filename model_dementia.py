"""
model_dementia.py — Dementia Detection Engine
===============================================
Reuses Wav2Vec2 already loaded by model.py (zero extra RAM).
DementiaTransformer: 4-layer encoder + attention pooling.

Fallback chain:
  1. DementiaTransformer with trained checkpoint
  2. Wav2Vec2 proxy + heuristic blend
  3. Pure heuristic scoring
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

_dementia_model = None
DEMENTIA_CKPT   = Path("checkpoints/dementia_transformer.pt")


if TORCH_AVAILABLE:
    class AttentionPooling(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.attn = nn.Linear(d_model, 1)

        def forward(self, x):
            w = torch.softmax(self.attn(x), dim=1)
            return (w * x).sum(dim=1), w.squeeze(-1)


    class DementiaTransformer(nn.Module):
        """
        Input:  Wav2Vec2 embeddings (B, T, 768)
        Output: logits (B,2), risk (B,1), attn_weights (B,T)
        """
        def __init__(self, d_in=768, d_model=256, nhead=8,
                     n_layers=4, dropout=0.1):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True,
                norm_first=True)
            self.transformer = nn.TransformerEncoder(enc, n_layers)
            self.attn_pool   = AttentionPooling(d_model)
            self.classifier  = nn.Sequential(
                nn.Linear(d_model, 128), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2))
            self.risk_head   = nn.Sequential(
                nn.Linear(d_model, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid())

        def forward(self, x):
            x = self.proj(x)
            x = self.transformer(x)
            pooled, attn_w = self.attn_pool(x)
            return self.classifier(pooled), self.risk_head(pooled), attn_w


def _get_dementia_model():
    global _dementia_model
    if _dementia_model is None and TORCH_AVAILABLE:
        _dementia_model = DementiaTransformer().to(DEVICE)
        if DEMENTIA_CKPT.exists():
            _dementia_model.load_state_dict(
                torch.load(str(DEMENTIA_CKPT), map_location=DEVICE))
            logger.info("Dementia model: loaded trained weights")
        else:
            logger.warning(
                "Dementia model: no checkpoint found. "
                "Train with: python train_dementia.py (in Colab)")
        _dementia_model.eval()
    return _dementia_model


def _heuristic_dementia_score(audio_feats: dict) -> Tuple[float, dict]:
    """Literature-calibrated heuristic dementia score."""
    pause  = min(audio_feats.get("pause_ratio", 0) / 0.55, 1.0)
    pitch  = 1.0 - min(audio_feats.get("pitch_variability", 0) / 50.0, 1.0)
    mfcc   = 1.0 - min(audio_feats.get("mfcc_variance", 0) / 8.0, 1.0)
    srate  = 1.0 - min(audio_feats.get("speech_rate_proxy", 0) / 6.0, 1.0)
    e      = audio_feats.get("energy_std", 0)
    energy = (1.0 - min(e * 1e5 / 12.0, 1.0)) if e > 0 else 0.75
    score  = (pause*0.35 + pitch*0.25 + mfcc*0.20 +
              srate*0.10 + energy*0.10)
    contrib = dict(pause_ratio=round(pause,3),
                   pitch_monotonicity=round(pitch,3),
                   mfcc_richness=round(mfcc,3),
                   speech_rate=round(srate,3),
                   energy_dynamics=round(energy,3))
    return float(np.clip(score, 0.03, 0.97)), contrib


def predict_dementia(audio_path: str,
                     speech_emb=None,
                     audio_feats: dict = None) -> dict:
    """
    Dementia prediction reusing Wav2Vec2 embeddings
    already computed by predict_from_audio_frames().

    Args:
        audio_path:  path to WAV file
        speech_emb:  (1,T,768) tensor from model.py — pass directly
                     to avoid loading Wav2Vec2 twice
        audio_feats: heuristic feature dict from model.py
    """
    # Load audio feats if not provided
    if audio_feats is None:
        from model import _extract_heuristic_audio_features
        audio_feats = _extract_heuristic_audio_features(audio_path)

    # Load speech embedding if not provided
    if speech_emb is None and TORCH_AVAILABLE:
        from model import _extract_wav2vec2_embeddings
        speech_emb = _extract_wav2vec2_embeddings(audio_path)

    method     = "heuristic"
    top_frames = []
    contrib    = {}

    if TORCH_AVAILABLE and speech_emb is not None:
        model = _get_dementia_model()
        if model is not None:
            try:
                with torch.no_grad():
                    logits, risk, attn_w = model(speech_emb)
                    probs  = torch.softmax(logits, dim=-1)
                    p_cls  = float(probs[0, 1].cpu())
                    p_risk = float(risk[0, 0].cpu())
                    attn   = attn_w[0].cpu().numpy()
                p_dem = float(np.clip(0.5*p_cls + 0.5*p_risk, 0.03, 0.97))
                top_frames = [
                    f"{round(float(t)*0.02,1)}s"
                    for t in np.argsort(attn)[-3:][::-1]
                ]
                contrib = dict(dementia_cls=round(p_cls,3),
                               dementia_risk=round(p_risk,3))
                method  = "dementia_transformer"
            except Exception as e:
                logger.warning(f"DementiaTransformer failed: {e}")
                p_dem, contrib = _heuristic_dementia_score(audio_feats)
        else:
            # No checkpoint — use Wav2Vec2 proxy
            with torch.no_grad():
                pooled = speech_emb.mean(1).squeeze(0).cpu().numpy()
            proxy   = float(np.clip(np.mean(np.abs(pooled[:8]))*2, 0, 1))
            h, hc   = _heuristic_dementia_score(audio_feats)
            p_dem   = float(np.clip(0.5*proxy + 0.5*h, 0.03, 0.97))
            contrib = {**hc, "wav2vec2_proxy": round(proxy,3)}
            method  = "wav2vec2_heuristic"
    else:
        p_dem, contrib = _heuristic_dementia_score(audio_feats)

    risk_pct = round(float(p_dem) * 100, 1)
    if risk_pct <= 30:   level = "Low"
    elif risk_pct <= 55: level = "Moderate"
    elif risk_pct <= 75: level = "High"
    else:                level = "Very High"

    return dict(
        dementia_prediction=(
            "Dementia Indicators Detected"
            if p_dem >= 0.50 else "No Dementia Indicators Detected"),
        dementia_risk_percent=risk_pct,
        dementia_level=level,
        dementia_confidence=round(abs(p_dem - 0.5) * 200, 1),
        dementia_method=method,
        dementia_contributions={
            k: round(v*100,1)
            for k,v in contrib.items() if isinstance(v, float)},
        attention_highlights=top_frames,
    )
