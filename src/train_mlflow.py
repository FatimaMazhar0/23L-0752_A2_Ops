"""Task 3.3 / 3.4: train a VisionGuard configuration with MLflow tracking.

  python src/train_mlflow.py manual  D_lr2_1e-4    # Keras callback logs everything by hand
  python src/train_mlflow.py autolog D_lr2_1e-4    # MLflow TensorFlow autologging
  python src/train_mlflow.py compare A_ref C_aug_flip ...   # Task 3.5: validation metrics only
"""
import sys
import time
from contextlib import nullcontext

import mlflow
import tensorflow as tf

import tracking_mlflow as tracking
from data import make_ds
from experiments import DEFAULTS, EXPERIMENTS
from models import build_transfer, compile_model, unfreeze_last_block
from train_utils import fit_stage, list_trainable, set_seed

BATCH_SIZE = 32

CHANGES = {                                   # what each run changes, and relative to which run
    "A_ref": "reference, no change",
    "B_size224": "image size 160 to 224 (vs A_ref)",
    "C_aug_flip": "flip-only augmentation (vs A_ref)",
    "D_lr2_1e-4": "stage-2 learning rate 1e-5 to 1e-4 (vs A_ref)",
    "E_unfreeze_14": "unfreeze from block_14 instead of block_16 (vs D_lr2_1e-4)",
    "F_effnetb0": "backbone MobileNetV2 to EfficientNetB0 (vs A_ref)",
}


def train(cfg, tag, stage_run=None, callbacks=None):
    """Two-stage training (frozen backbone, then last block unfrozen). Returns model + histories."""
    stage_run = stage_run or (lambda name: nullcontext())
    callbacks = callbacks or {}
    tf.keras.backend.clear_session()
    set_seed()
    train_ds = make_ds("train", train=True, img_size=cfg["img_size"], aug=cfg["aug"])
    val_ds = make_ds("val", img_size=cfg["img_size"])

    model = compile_model(build_transfer(cfg["img_size"], cfg["dropout"], cfg["backbone"]),
                          cfg["lr1"])
    list_trainable(model, f"{tag}_s1")
    with stage_run("stage1_frozen_backbone"):
        h1, b1 = fit_stage(model, f"{tag}_s1", train_ds, val_ds, 25, callbacks.get("s1", ()))

    unfreeze_last_block(model, cfg["start"] or None)
    model = compile_model(model, cfg["lr2"])
    list_trainable(model, f"{tag}_s2")
    with stage_run("stage2_finetune"):
        h2, b2 = fit_stage(model, f"{tag}_s2", train_ds, val_ds, 20, callbacks.get("s2", ()))
    return model, (h1, b1), (h2, b2)


def config_params(cfg):
    params = {**cfg, "start": cfg["start"] or "default", "batch_size": BATCH_SIZE,
              "stage1_max_epochs": 25, "stage2_max_epochs": 20,
              "early_stopping_patience": 5, "lr_reduce_patience": 2}
    return {k: str(v) for k, v in params.items()}


def final_val_metrics(h2, b2):
    i = b2 - 1
    return {"best_epoch_stage2": b2, "val_loss": h2["val_loss"][i], "val_accuracy": h2["val_accuracy"][i],
            "val_auc": h2["val_auc"][i], "val_precision": h2["val_precision"][i],
            "val_recall": h2["val_recall"][i]}


def run_manual(name, task="3.3", with_test=True):
    """Manual logging with a Keras callback. task 3.5 runs skip the test set entirely."""
    cfg = {**DEFAULTS, **EXPERIMENTS[name]}
    kind = "manual" if task == "3.3" else "compare"
    tag = f"mlflow_{name}_{kind}"
    t0 = time.time()
    with mlflow.start_run(run_name=f"{name}_{kind}"):
        mlflow.set_tags(tracking.run_tags({"approach": "manual-callback", "task": task,
                                           "config": name, "change": CHANGES[name]}))
        mlflow.log_params(config_params(cfg))
        model, (h1, b1), (h2, b2) = train(
            cfg, tag, callbacks={"s1": [tracking.MlflowLogger("s1_")],
                                 "s2": [tracking.MlflowLogger("s2_")]})
        mlflow.log_metrics({"best_epoch_stage1": b1, **final_val_metrics(h2, b2),
                            "train_minutes": (time.time() - t0) / 60})
        if with_test:
            tracking.log_test_results(model, cfg["img_size"])
        tracking.log_artifacts(model, tag)


def run_autolog(name):
    cfg = {**DEFAULTS, **EXPERIMENTS[name]}
    tag = f"mlflow_{name}_autolog"
    # checkpoint=False: otherwise autologging uploads a model checkpoint from every stage
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False, checkpoint=False)
    tags = tracking.run_tags({"approach": "autolog", "task": "3.3", "config": name})

    def stage_run(stage_name):
        run = mlflow.start_run(run_name=stage_name, nested=True)
        mlflow.set_tags({**tags, "stage": stage_name})    # autolog never sets tags itself
        return run

    try:
        with mlflow.start_run(run_name=f"{name}_autolog"):
            mlflow.set_tags(tags)
            # autolog does not know about the image size, augmentation, backbone or dropout,
            # so the configuration is added by hand on the parent run
            mlflow.log_params(config_params(cfg))
            # autolog expects one fit() per run, so each training stage gets a child run
            model, (h1, b1), (h2, b2) = train(cfg, tag, stage_run=stage_run)
            mlflow.log_metrics({"best_epoch_stage1": b1, **final_val_metrics(h2, b2)})
            tracking.log_test_results(model, cfg["img_size"])
            tracking.log_artifacts(model, tag)
    finally:
        mlflow.autolog(disable=True)


if __name__ == "__main__":
    modes = ("manual", "autolog", "compare")
    if len(sys.argv) < 3 or sys.argv[1] not in modes or any(n not in EXPERIMENTS for n in sys.argv[2:]):
        sys.exit(f"usage: python src/train_mlflow.py [{'|'.join(modes)}] [{'|'.join(EXPERIMENTS)}]")
    tracking.setup()
    if sys.argv[1] == "compare":
        for n in sys.argv[2:]:
            run_manual(n, task="3.5", with_test=False)
    elif len(sys.argv) == 3:
        {"manual": run_manual, "autolog": run_autolog}[sys.argv[1]](sys.argv[2])
    else:
        sys.exit("manual and autolog take exactly one configuration name")