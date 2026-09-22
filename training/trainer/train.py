"""
training/trainer/train.py
==========================
Two-stage training pipeline for Neuro Fusion v2.

Stage A — SVM Ensemble (works with any dataset size ≥ 10)
  • Loads 14-dim heuristic feature vectors from .npz files
  • StandardScaler + SelectKBest feature selection
  • Voting ensemble: SVM (weight=2) + RandomForest + GradientBoosting
  • Leave-One-Out CV for honest small-dataset evaluation
  • Augments training data with Gaussian noise
  • Saves: checkpoints/stage_a_ensemble.pkl

Stage B — CrossModalTransformer fine-tuning (recommended ≥ 40 samples)
  • Loads full speech_emb + visual_emb tensors
  • Two-phase training:
      Phase 1 (epochs 1-10): only classifier/risk_head trainable
      Phase 2 (epochs 11-N): all layers, lower LR
  • Combined loss: 0.7 × CrossEntropy + 0.3 × MSE
  • Class-weighted loss for imbalanced datasets
  • Saves: checkpoints/cross_modal_v2.pt

Usage:
  cd project_final
  python training/trainer/train.py --stage A
  python training/trainer/train.py --stage B --epochs 30 --augment
  python training/trainer/train.py --stage AB   # run both sequentially
"""

import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROC_DIR  = Path("training/data/processed")
CKPT_DIR  = Path("checkpoints")

# ─────────────────────────────────────────────────────────────
# Data loading utilities
# ─────────────────────────────────────────────────────────────

def load_heuristic_features(proc_dir: Path):
    """
    Load 14-dim heuristic feature vectors and labels.
    Returns (X, y, ids) where X.shape = (N, 14).
    """
    X, y, ids = [], [], []
    for f in sorted(proc_dir.glob("*.npz")):
        d     = np.load(f, allow_pickle=True)
        label = int(d["label"])
        feats = d["feat_vector"].astype(np.float32)
        if feats.shape[0] < 14:
            feats = np.pad(feats, (0, 14 - feats.shape[0]))
        X.append(feats[:14])
        y.append(label)
        ids.append(f.stem)
    return np.array(X), np.array(y), ids


def load_embedding_features(proc_dir: Path, max_s=100, max_v=20):
    """
    Load speech_emb + visual_emb tensors for transformer training.
    Pads/truncates to (max_s, 768) and (max_v, 1404).
    Returns (speech_tensors, visual_tensors, labels) as numpy arrays.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required for Stage B. Install: pip install torch")

    S_list, V_list, y_list = [], [], []
    skipped = 0
    for f in sorted(proc_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if "speech_emb" not in d or "visual_emb" not in d:
            skipped += 1
            continue
        s = d["speech_emb"].astype(np.float32)   # (T_s, 768)
        v = d["visual_emb"].astype(np.float32)   # (T_v, 1404)

        # Pad / truncate speech
        if s.shape[0] >= max_s:
            s = s[:max_s]
        else:
            s = np.pad(s, ((0, max_s - s.shape[0]), (0, 0)))

        # Pad / truncate visual
        if v.shape[1] < 1404:
            v = np.pad(v, ((0, 0), (0, 1404 - v.shape[1])))
        elif v.shape[1] > 1404:
            v = v[:, :1404]
        if v.shape[0] >= max_v:
            v = v[:max_v]
        else:
            v = np.pad(v, ((0, max_v - v.shape[0]), (0, 0)))

        S_list.append(s)
        V_list.append(v)
        y_list.append(int(d["label"]))

    if skipped:
        print(f"  ⚠ Skipped {skipped} files (no speech/visual embeddings — "
              f"preprocess with transformers + mediapipe installed)")

    return (np.array(S_list, dtype=np.float32),
            np.array(V_list, dtype=np.float32),
            np.array(y_list, dtype=np.int64))


# ─────────────────────────────────────────────────────────────
# Stage A — SVM Ensemble
# ─────────────────────────────────────────────────────────────

def augment_features(X: np.ndarray, y: np.ndarray, n_aug: int = 4):
    """Add Gaussian noise augmentation to training features."""
    stds  = X.std(axis=0, keepdims=True) + 1e-8
    X_aug = [X]
    y_aug = [y]
    for _ in range(n_aug):
        noise = np.random.randn(*X.shape) * stds * 0.05
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


def train_stage_a(augment: bool = True):
    print("\n" + "="*55)
    print("  STAGE A — SVM Ensemble Training")
    print("="*55)

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        from sklearn.pipeline import Pipeline
    except ImportError:
        raise RuntimeError("scikit-learn required. Install: pip install scikit-learn")

    X, y, ids = load_heuristic_features(PROC_DIR)
    n_samples  = len(y)
    n_ad       = int(y.sum())
    n_hc       = n_samples - n_ad
    print(f"  Loaded: {n_samples} samples  (AD={n_ad}, HC={n_hc})")

    if n_samples < 6:
        print(f"  ✗ Need at least 6 samples. Found {n_samples}.")
        return

    feature_names = [
        "pause_ratio", "speech_rate", "energy_mean", "energy_std",
        "mfcc_variance", "pitch_variability", "zcr", "spec_centroid",
        "duration", "face_ratio", "expressiveness", "inter_frame_motion",
        "visual_entropy", "au_variance"
    ]

    w_ad = n_samples / (2 * n_ad) if n_ad > 0 else 1.0
    w_hc = n_samples / (2 * n_hc) if n_hc > 0 else 1.0
    class_weight = {1: w_ad, 0: w_hc}
    print(f"  Class weights: AD={w_ad:.2f}  HC={w_hc:.2f}")

    print(f"\n  Running Leave-One-Out CV ({n_samples} folds)…")
    loo   = LeaveOneOut()
    preds, trues, probas = [], [], []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if augment:
            X_tr, y_tr = augment_features(X_tr, y_tr, n_aug=4)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(10, X.shape[1]))),
            ("clf", VotingClassifier(
                estimators=[
                    ("svm", SVC(kernel="rbf", probability=True,
                                class_weight=class_weight, C=1.0, gamma="scale")),
                    ("rf",  RandomForestClassifier(n_estimators=50,
                                class_weight=class_weight, random_state=42)),
                    ("gb",  GradientBoostingClassifier(n_estimators=50,
                                learning_rate=0.1, random_state=42)),
                ],
                voting="soft",
                weights=[2, 1, 1],
            )),
        ])
        pipe.fit(X_tr, y_tr)
        pred  = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        preds.append(pred[0])
        trues.append(y_te[0])
        probas.append(proba[0])

        if fold % 10 == 0 or fold == n_samples - 1:
            acc_so_far = accuracy_score(trues, preds)
            print(f"  Fold {fold+1:3d}/{n_samples}  acc={acc_so_far*100:.1f}%")

    acc = accuracy_score(trues, preds)
    try:
        auc = roc_auc_score(trues, probas)
    except Exception:
        auc = 0.0

    print(f"\n  LOO-CV Results:")
    print(f"    Accuracy : {acc*100:.1f}%")
    print(f"    AUC-ROC  : {auc:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(trues, preds, target_names=["HC", "AD"],
                                 zero_division=0))

    print("  Training final model on full dataset…")
    X_full, y_full = augment_features(X, y, n_aug=4) if augment else (X, y)
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=min(10, X.shape[1]))),
        ("clf", VotingClassifier(
            estimators=[
                ("svm", SVC(kernel="rbf", probability=True,
                            class_weight=class_weight, C=1.0, gamma="scale")),
                ("rf",  RandomForestClassifier(n_estimators=100,
                            class_weight=class_weight, random_state=42)),
                ("gb",  GradientBoostingClassifier(n_estimators=100,
                            learning_rate=0.1, random_state=42)),
            ],
            voting="soft",
            weights=[2, 1, 1],
        )),
    ])
    final_pipe.fit(X_full, y_full)

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / "stage_a_ensemble.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({
            "pipeline":      final_pipe,
            "feature_names": feature_names,
            "n_samples":     n_samples,
            "loo_accuracy":  acc,
            "loo_auc":       auc,
        }, f)

    print(f"\n  ✓ Saved: {ckpt_path}")
    print("="*55)
    return acc, auc


# ─────────────────────────────────────────────────────────────
# Stage B — CrossModalTransformer fine-tuning
# ─────────────────────────────────────────────────────────────

def train_stage_b(epochs: int = 30, augment: bool = True):
    print("\n" + "="*55)
    print("  STAGE B — CrossModalTransformer Training")
    print("="*55)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise RuntimeError("PyTorch required for Stage B. Install: pip install torch")

    from model import CrossModalTransformer, DEVICE

    S_np, V_np, y_np = load_embedding_features(PROC_DIR)
    n = len(y_np)
    if n < 10:
        print(f"  ✗ Need at least 10 samples with embeddings. Found {n}.")
        print("    Preprocess with transformers + mediapipe installed, then retry.")
        return

    n_ad = int(y_np.sum())
    n_hc = n - n_ad
    print(f"  Loaded: {n} samples with full embeddings (AD={n_ad}, HC={n_hc})")

    S = torch.tensor(S_np).to(DEVICE)
    V = torch.tensor(V_np).to(DEVICE)
    y = torch.tensor(y_np).to(DEVICE)

    w_ad = n / (2 * n_ad) if n_ad > 0 else 1.0
    w_hc = n / (2 * n_hc) if n_hc > 0 else 1.0
    weights = torch.tensor([w_hc, w_ad], dtype=torch.float32).to(DEVICE)
    ce_fn   = nn.CrossEntropyLoss(weight=weights)

    idx      = torch.randperm(n)
    n_val    = max(1, int(n * 0.2))
    n_train  = n - n_val
    tr_idx   = idx[:n_train]
    val_idx  = idx[n_train:]

    S_tr, V_tr, y_tr = S[tr_idx], V[tr_idx], y[tr_idx]
    S_val, V_val, y_val = S[val_idx], V[val_idx], y[val_idx]
    print(f"  Train: {n_train}  |  Val: {n_val}")

    model = CrossModalTransformer().to(DEVICE)
    for name, param in model.named_parameters():
        if "classifier" not in name and "risk_head" not in name:
            param.requires_grad = False

    phase1_epochs = min(10, epochs // 3)
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    print(f"\n  Phase 1: training heads only ({phase1_epochs} epochs, lr=1e-3)…")
    _run_training(model, opt1, ce_fn, S_tr, V_tr, y_tr, S_val, V_val, y_val,
                  epochs=phase1_epochs, batch_size=8, augment=augment,
                  phase_label="Phase1")

    for param in model.parameters():
        param.requires_grad = True

    phase2_epochs = epochs - phase1_epochs
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs)

    print(f"\n  Phase 2: fine-tuning all layers ({phase2_epochs} epochs, lr=1e-4)…")
    best_val_acc = _run_training(
        model, opt2, ce_fn, S_tr, V_tr, y_tr, S_val, V_val, y_val,
        epochs=phase2_epochs, batch_size=8, augment=augment,
        phase_label="Phase2", scheduler=sched
    )

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / "cross_modal_v2.pt"
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"\n  ✓ Saved: {ckpt_path}")
    print(f"  Best val accuracy: {best_val_acc*100:.1f}%")
    print("="*55)
    return best_val_acc


def _augment_embeddings(S, V, y, noise_std=0.02):
    import torch
    S_aug = S + torch.randn_like(S) * noise_std
    V_aug = V + torch.randn_like(V) * noise_std
    return (torch.cat([S, S_aug]), torch.cat([V, V_aug]),
            torch.cat([y, y]))


def _run_training(model, optimizer, ce_fn, S_tr, V_tr, y_tr,
                  S_val, V_val, y_val, epochs, batch_size,
                  augment, phase_label, scheduler=None):
    import torch
    import torch.nn.functional as F
    best_val_acc = 0.0

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(y_tr))
        S_e, V_e, y_e = S_tr[idx], V_tr[idx], y_tr[idx]
        if augment:
            S_e, V_e, y_e = _augment_embeddings(S_e, V_e, y_e)

        total_loss = 0.0
        n_batches  = max(1, len(y_e) // batch_size)
        for i in range(n_batches):
            s = S_e[i*batch_size:(i+1)*batch_size]
            v = V_e[i*batch_size:(i+1)*batch_size]
            yb = y_e[i*batch_size:(i+1)*batch_size]
            if len(yb) == 0:
                continue
            optimizer.zero_grad()
            logits, risk = model(s, v)
            ce   = ce_fn(logits, yb)
            mse  = F.mse_loss(risk, yb.float().unsqueeze(1))
            loss = 0.7 * ce + 0.3 * mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_v, _ = model(S_val, V_val)
            preds_v = logits_v.argmax(dim=1)
            val_acc = (preds_v == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        avg_loss = total_loss / n_batches
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            print(f"  [{phase_label}] Ep {ep+1:3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc*100:.1f}%")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Neuro Fusion v2 — Trainer")
    parser.add_argument("--stage", choices=["A", "B", "AB"], default="AB",
                        help="A=SVM, B=Transformer, AB=both")
    parser.add_argument("--epochs",  type=int, default=30,
                        help="Epochs for Stage B (default: 30)")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable data augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    args = parser.parse_args()

    if not any(PROC_DIR.glob("*.npz")):
        print("No preprocessed data found.")
        print("Run:  python training/preprocessor/preprocess.py  first.")
        return

    if args.stage in ("A", "AB"):
        train_stage_a(augment=args.augment)

    if args.stage in ("B", "AB"):
        train_stage_b(epochs=args.epochs, augment=args.augment)

    print("\n✓ Training complete!")
    print("Restart your server to load new weights:")
    print("  uvicorn app:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
