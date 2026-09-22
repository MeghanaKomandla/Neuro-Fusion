"""
training/evaluator/evaluate.py
================================
Evaluate trained model performance and check checkpoint integration.

Usage:
  cd project_final
  python training/evaluator/evaluate.py
  python training/evaluator/evaluate.py --integrate
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROC_DIR = Path("training/data/processed")
CKPT_DIR = Path("checkpoints")


def evaluate_stage_a():
    print("\n── Stage A: SVM Ensemble ──")
    ckpt = CKPT_DIR / "stage_a_ensemble.pkl"
    if not ckpt.exists():
        print(f"  No checkpoint at {ckpt}")
        return

    with open(ckpt, "rb") as f:
        data = pickle.load(f)

    print(f"  Trained on    : {data['n_samples']} samples")
    print(f"  LOO Accuracy  : {data['loo_accuracy']*100:.1f}%")
    print(f"  LOO AUC-ROC   : {data['loo_auc']:.3f}")

    pipe = data["pipeline"]
    X_test = np.random.randn(2, 14).astype(np.float32)
    proba  = pipe.predict_proba(X_test)[:, 1]
    print(f"  Predict test  : {proba}")


def evaluate_stage_b():
    print("\n── Stage B: CrossModalTransformer ──")
    ckpt = CKPT_DIR / "cross_modal_v2.pt"
    if not ckpt.exists():
        print(f"  No checkpoint at {ckpt}")
        return

    try:
        import torch
        from model import CrossModalTransformer, DEVICE
    except ImportError:
        print("  PyTorch not available for evaluation.")
        return

    model = CrossModalTransformer().to(DEVICE)
    state = torch.load(str(ckpt), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    S_list, V_list, y_list = [], [], []
    for f in sorted(PROC_DIR.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if "speech_emb" not in d or "visual_emb" not in d:
            continue
        s = d["speech_emb"].astype(np.float32)
        v = d["visual_emb"].astype(np.float32)
        s = _pad(s, 100, 768)
        v = _pad_to_1404(v, 20)
        S_list.append(s)
        V_list.append(v)
        y_list.append(int(d["label"]))

    if not S_list:
        print("  No files with embeddings found for evaluation.")
        return

    S = torch.tensor(np.array(S_list)).to(DEVICE)
    V = torch.tensor(np.array(V_list)).to(DEVICE)
    y = np.array(y_list)

    with torch.no_grad():
        logits, risk = model(S, V)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=1).cpu().numpy()
        p_ad   = probs[:, 1].cpu().numpy()
        risk_s = risk.squeeze(1).cpu().numpy()

    try:
        from sklearn.metrics import (accuracy_score, roc_auc_score,
                                     classification_report, confusion_matrix)
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, p_ad) if len(np.unique(y)) > 1 else 0.0
        cm  = confusion_matrix(y, preds)

        print(f"  Samples       : {len(y)}  (AD={int(y.sum())}, HC={len(y)-int(y.sum())})")
        print(f"  Accuracy      : {acc*100:.1f}%")
        print(f"  AUC-ROC       : {auc:.3f}")
        print(f"  Confusion matrix:")
        print(f"               Pred HC  Pred AD")
        print(f"    True HC :    {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"    True AD :    {cm[1,0]:4d}    {cm[1,1]:4d}")
        print(f"\n  Classification report:")
        print(classification_report(y, preds, target_names=["HC", "AD"],
                                     zero_division=0))

        confidences = np.abs(p_ad - 0.5) * 200
        print(f"  Confidence scores — mean={confidences.mean():.1f}%  "
              f"min={confidences.min():.1f}%  max={confidences.max():.1f}%")

        if auc >= 0.80:
            print("\n  ✓ AUC ≥ 0.80 — model is performing well!")
        elif auc >= 0.65:
            print("\n  ℹ AUC 0.65-0.80 — acceptable with small dataset.")
        else:
            print("\n  ⚠ AUC < 0.65 — collect more data or train longer.")

    except ImportError:
        print(f"  Accuracy: {(preds == y).mean()*100:.1f}%")


def _pad(arr, max_t, dim):
    t = arr.shape[0]
    if t >= max_t:
        return arr[:max_t]
    return np.pad(arr, ((0, max_t - t), (0, 0)))


def _pad_to_1404(arr, max_t):
    d = arr.shape[1] if len(arr.shape) > 1 else 1
    if d < 1404:
        arr = np.pad(arr, ((0, 0), (0, 1404 - d)))
    elif d > 1404:
        arr = arr[:, :1404]
    t = arr.shape[0]
    if t >= max_t:
        return arr[:max_t]
    return np.pad(arr, ((0, max_t - t), (0, 0)))


def print_dataset_stats():
    print("\n── Dataset Statistics ──")
    files  = list(PROC_DIR.glob("*.npz"))
    n_ad   = sum(1 for f in files if f.name.startswith("AD_"))
    n_hc   = sum(1 for f in files if f.name.startswith("HC_"))
    with_s = sum(1 for f in files
                 if "speech_emb" in np.load(f, allow_pickle=True))
    with_v = sum(1 for f in files
                 if "visual_emb" in np.load(f, allow_pickle=True))

    print(f"  Total files   : {len(files)}")
    print(f"  AD            : {n_ad}")
    print(f"  HC            : {n_hc}")
    print(f"  With speech   : {with_s}")
    print(f"  With visual   : {with_v}")


def main():
    parser = argparse.ArgumentParser(description="Neuro Fusion v2 — Evaluator")
    parser.add_argument("--integrate", action="store_true",
                        help="Check checkpoint loading path")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  NEURO FUSION v2 — EVALUATION REPORT")
    print("="*55)

    if PROC_DIR.exists():
        print_dataset_stats()
    else:
        print("No processed data found. Run preprocessor first.")

    evaluate_stage_a()
    evaluate_stage_b()

    if args.integrate:
        print("\n── Integration Check ──")
        ckpt_b = CKPT_DIR / "cross_modal_v2.pt"
        ckpt_a = CKPT_DIR / "stage_a_ensemble.pkl"
        if ckpt_b.exists():
            print(f"  ✓ cross_modal_v2.pt found — app.py loads it automatically")
        else:
            print(f"  ✗ cross_modal_v2.pt missing — run Stage B training first")
        if ckpt_a.exists():
            print(f"  ✓ stage_a_ensemble.pkl found")
        print("\n  Restart server to apply:")
        print("    uvicorn app:app --reload --host 0.0.0.0 --port 8000")

    print("\n" + "="*55)


if __name__ == "__main__":
    main()
