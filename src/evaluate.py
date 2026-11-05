import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from data import make_ds, read_split

BETA = 1.0   # 1 = balanced F1; >1 favours recall (catch fakes); <1 favours precision
MODELS = {
    "baseline": ("Baseline CNN", "models/baseline.keras"),
    "mnv2": ("MobileNetV2 (fine-tuned)", "models/exp_D_lr2_1e-4_s2.keras"),
}
os.makedirs("figures", exist_ok=True)


def predict(path, split):
    m = tf.keras.models.load_model(path, compile=False)
    return m.predict(make_ds(split), verbose=0).ravel()      # no shuffle: order = split file


def pick_threshold(y, p, beta):
    grid = np.linspace(0.05, 0.95, 91)
    scores = np.array([fbeta_score(y, p >= t, beta=beta, zero_division=0) for t in grid])
    return float(grid[int(np.argmax(scores))]), grid, scores


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    return {"threshold": round(t, 2),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p)}, pred


yv = np.array(read_split("val")[1])
test_paths, yt = read_split("test")
yt = np.array(yt)

rows, cms, rocs, sweeps = [], {}, {}, {}
for tag, (disp, path) in MODELS.items():
    pv = predict(path, "val")
    thr, grid, scores = pick_threshold(yv, pv, BETA)     # threshold chosen on VALIDATION only
    sweeps[disp] = (grid, scores, thr)
    pt = predict(path, "test")                           # the single pass over the test set
    m, pred = metrics_at(yt, pt, thr)
    m05, _ = metrics_at(yt, pt, 0.5)
    rows.append({"model": disp, "rule": f"val-chosen t", **m})
    rows.append({"model": disp, "rule": "default t=0.5", **m05})
    cms[disp] = confusion_matrix(yt, pred)
    rocs[disp] = (*roc_curve(yt, pt)[:2], m["roc_auc"])
    pd.DataFrame({"path": test_paths, "label": yt, "prob_ai": pt, "pred": pred}
                 ).to_csv(f"figures/test_predictions_{tag}.csv", index=False)

df = pd.DataFrame(rows)
cols = list(df.columns)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in df.iterrows():
    md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
open("figures/test_metrics.md", "w").write("\n".join(md))
print("\n".join(md))

# confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, (disp, cm) in zip(ax, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot(ax=a, colorbar=False)
    a.set_title(disp)
plt.tight_layout(); plt.savefig("figures/confusion_matrices.png", dpi=120); plt.close()

# ROC curves
plt.figure(figsize=(5.5, 5))
for disp, (fpr, tpr, auc) in rocs.items():
    plt.plot(fpr, tpr, label=f"{disp} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves, test set"); plt.legend(); plt.tight_layout()
plt.savefig("figures/roc_curves.png", dpi=120); plt.close()

# threshold sweep on validation (justification figure)
plt.figure(figsize=(6, 4))
for disp, (grid, scores, thr) in sweeps.items():
    plt.plot(grid, scores, label=f"{disp} (best t={thr:.2f})")
    plt.axvline(thr, ls=":", alpha=0.6)
plt.xlabel("Decision threshold"); plt.ylabel(f"F-beta on validation (beta={BETA})")
plt.title("Threshold chosen on validation data"); plt.legend(); plt.tight_layout()
plt.savefig("figures/threshold_sweep.png", dpi=120); plt.close()
print("Saved figures")