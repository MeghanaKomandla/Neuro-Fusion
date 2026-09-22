"""
train_dementia.py — Dementia Model Training
=============================================
Run this in Google Colab (T4 GPU).

Usage:
  python train_dementia.py
  python train_dementia.py --epochs 50

After training, download:
  checkpoints/dementia_transformer.pt
and paste it into your local checkpoints/ folder.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

PROC_DIR = Path("training/data/processed")
CKPT_DIR = Path("checkpoints")


def load_data(max_t: int = 150):
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required")

    S, y = [], []
    skipped = 0
    for f in sorted(PROC_DIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if "speech_emb" not in d:
            skipped += 1
            continue
        s = d["speech_emb"].astype(np.float32)
        # Pad/truncate time dimension
        if s.shape[0] >= max_t:
            s = s[:max_t]
        else:
            s = np.pad(s, ((0, max_t - s.shape[0]), (0, 0)))
        # Ensure 768 dim (Wav2Vec2 output)
        if s.shape[1] < 768:
            s = np.pad(s, ((0, 0), (0, 768 - s.shape[1])))
        else:
            s = s[:, :768]
        S.append(s)
        y.append(int(d["label"]))
    if skipped:
        print(f"  Skipped {skipped} files (no speech embeddings)")
    return np.array(S, dtype=np.float32), np.array(y, dtype=np.int64)


def train(epochs: int = 40):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from model_dementia import DementiaTransformer

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print("  DEMENTIA — DementiaTransformer Training")
    print(f"{'='*55}")
    print(f"  Device: {DEVICE}")

    S_np, y_np = load_data()
    n    = len(y_np)
    n_ad = int(y_np.sum())
    n_hc = n - n_ad

    if n < 10:
        print(f"  Need at least 10 samples. Found {n}.")
        print("  Collect more data and rerun preprocessing.")
        return

    print(f"  Loaded: {n} samples  (dementia={n_ad}, healthy={n_hc})")

    S = torch.tensor(S_np).to(DEVICE)
    y = torch.tensor(y_np).to(DEVICE)

    # Class weights
    w_pos = n / (2 * n_ad) if n_ad > 0 else 1.0
    w_neg = n / (2 * n_hc) if n_hc > 0 else 1.0
    weights = torch.tensor([w_neg, w_pos]).float().to(DEVICE)
    ce_fn   = nn.CrossEntropyLoss(weight=weights)

    # 80/20 split
    idx    = torch.randperm(n)
    n_val  = max(1, int(n * 0.2))
    tr_idx = idx[n_val:]
    vl_idx = idx[:n_val]
    S_tr, y_tr = S[tr_idx], y[tr_idx]
    S_vl, y_vl = S[vl_idx], y[vl_idx]
    print(f"  Train: {len(y_tr)}  |  Val: {len(y_vl)}")

    model = DementiaTransformer().to(DEVICE)

    # Phase 1 — heads only
    ph1 = min(10, epochs // 4)
    for p in model.parameters():
        p.requires_grad = False
    for p in list(model.classifier.parameters()) + \
             list(model.risk_head.parameters()) + \
             list(model.attn_pool.parameters()):
        p.requires_grad = True

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4)

    print(f"\n  Phase 1: {ph1} epochs, lr=1e-3 (heads only)…")
    best = _train_loop(model, opt1, ce_fn, S_tr, y_tr, S_vl, y_vl,
                       ph1, 8, "P1")

    # Phase 2 — all layers
    for p in model.parameters():
        p.requires_grad = True
    ph2   = epochs - ph1
    opt2  = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=ph2)

    print(f"\n  Phase 2: {ph2} epochs, lr=1e-4 (all layers)…")
    best = _train_loop(model, opt2, ce_fn, S_tr, y_tr, S_vl, y_vl,
                       ph2, 8, "P2", sched)

    CKPT_DIR.mkdir(exist_ok=True)
    out = CKPT_DIR / "dementia_transformer.pt"
    torch.save(model.state_dict(), str(out))
    print(f"\n  Checkpoint saved: {out}")
    print(f"  Best val accuracy: {best*100:.1f}%")
    print(f"{'='*55}")
    print("\nDownload and paste into your local checkpoints/ folder.")


def _train_loop(model, opt, ce_fn, S_tr, y_tr, S_vl, y_vl,
                epochs, batch, label, sched=None):
    import torch
    import torch.nn.functional as F
    best = 0.0
    for ep in range(epochs):
        model.train()
        idx  = torch.randperm(len(y_tr))
        Se   = S_tr[idx] + torch.randn_like(S_tr[idx]) * 0.02
        ye   = y_tr[idx]
        nb   = max(1, len(ye) // batch)
        tot  = 0.0
        for i in range(nb):
            s  = Se[i*batch:(i+1)*batch]
            yb = ye[i*batch:(i+1)*batch]
            if len(yb) == 0:
                continue
            opt.zero_grad()
            logits, risk, _ = model(s)
            ce   = ce_fn(logits, yb)
            mse  = F.mse_loss(risk, yb.float().unsqueeze(1))
            loss = 0.7 * ce + 0.3 * mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if sched:
            sched.step()
        model.eval()
        with torch.no_grad():
            lv, _, _ = model(S_vl)
            acc = (lv.argmax(1) == y_vl).float().mean().item()
        if acc > best:
            best = acc
        if (ep+1) % 5 == 0 or ep == 0 or ep == epochs-1:
            print(f"  [{label}] Ep {ep+1:3d}/{epochs}  "
                  f"loss={tot/nb:.4f}  val_acc={acc*100:.1f}%")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train(args.epochs)
