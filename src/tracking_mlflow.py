"""Shared MLflow helpers for Task 3: tracking setup, run tags, test evaluation, artifacts."""
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from data import make_ds, read_split

SEED = 42


def setup(experiment="visionguard"):
    """Point MLflow at the server in .env (URI, user and token come from environment variables)."""
    load_dotenv()
    if not os.getenv("MLFLOW_TRACKING_URI"):
        sys.exit("MLFLOW_TRACKING_URI is not set. Check your .env file.")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    print("Tracking URI:", mlflow.get_tracking_uri())


def _out(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def run_tags(extra=None):
    """Reproducibility tags: code commit, data version (DVC), seed, library version."""
    dirty = bool(_out(["git", "status", "--porcelain", "--", "src", "splits"]))
    if dirty:
        print("WARNING: uncommitted changes in src/ or splits/. Commit first so the "
              "git_commit tag matches the code that actually ran.")
    tags = {
        "git_commit": _out(["git", "rev-parse", "--short", "HEAD"]),
        "git_uncommitted_changes": str(dirty),
        "dvc_data_md5": re.search(r"md5:\s*(\S+)", open("data/raw.dvc").read()).group(1),
        "dvc_data_git_commit": _out(["git", "log", "-1", "--format=%h", "--", "data/raw.dvc"]),
        "seed": str(SEED),
        "tensorflow": tf.__version__,
    }
    tags.update(extra or {})
    return tags


class MlflowLogger(tf.keras.callbacks.Callback):
    """Manual logging: every epoch, send the metrics Keras computed (plus the learning rate)."""

    def __init__(self, prefix=""):
        super().__init__()
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        values = {f"{self.prefix}{k}": float(v) for k, v in (logs or {}).items()}
        values[f"{self.prefix}learning_rate"] = float(self.model.optimizer.learning_rate)
        mlflow.log_metrics(values, step=epoch)


def _best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def log_test_results(model, img_size):
    """Test metrics, confusion matrix and ROC curve. Threshold is chosen on validation only."""
    yv = np.array(read_split("val")[1])
    yt = np.array(read_split("test")[1])
    pv = model.predict(make_ds("val", img_size=img_size), verbose=0).ravel()
    pt = model.predict(make_ds("test", img_size=img_size), verbose=0).ravel()
    thr = _best_f1_threshold(yv, pv)
    pred = (pt >= thr).astype(int)
    mlflow.log_metrics({
        "test_threshold": thr,
        "test_accuracy": accuracy_score(yt, pred),
        "test_precision": precision_score(yt, pred, zero_division=0),
        "test_recall": recall_score(yt, pred),
        "test_f1": f1_score(yt, pred),
        "test_roc_auc": roc_auc_score(yt, pt),
    })
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(yt, pred), display_labels=["Real", "AI"]).plot(
        ax=ax, colorbar=False)
    ax.set_title(f"Test confusion matrix (t={thr:.2f})")
    fig.tight_layout()
    mlflow.log_figure(fig, "evaluation/confusion_matrix.png")
    plt.close(fig)
    fpr, tpr, _ = roc_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(yt, pt):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curve"); ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "evaluation/roc_curve.png")
    plt.close(fig)


def log_artifacts(model, tag_prefix):
    """Split lists, package list, training curves, trainable-layer lists, and the model."""
    mlflow.log_artifacts("splits", artifact_path="splits")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    mlflow.log_text(freeze, "environment/pip_freeze.txt")
    for stage in ("s1", "s2"):
        for ext in ("curves.png", "trainable.txt"):
            path = f"figures/{tag_prefix}_{stage}_{ext}"
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="training")
    mlflow.keras.log_model(model, name="model")