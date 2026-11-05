import csv, os, sys, time
import pandas as pd
import tensorflow as tf
from data import make_ds
from models import build_transfer, unfreeze_last_block, compile_model
from train_utils import set_seed, list_trainable, fit_stage

DEFAULTS = dict(backbone="mobilenetv2", img_size=160, aug="default",
                dropout=0.3, lr1=1e-3, lr2=1e-5, start="")

EXPERIMENTS = {
    "A_ref":         {},                                   # reference
    "B_size224":     dict(img_size=224),                   # one change vs A
    "C_aug_flip":    dict(aug="flip"),                     # one change vs A
    "D_lr2_1e-4":    dict(lr2=1e-4),                       # one change vs A
    "E_unfreeze_14": dict(lr2=1e-4, start="block_14"),     # one change vs D
    "F_effnetb0":    dict(backbone="efficientnetb0"),      # one change vs A
}
CSV = "figures/experiments.csv"


def run(name):
    cfg = {**DEFAULTS, **EXPERIMENTS[name]}
    tf.keras.backend.clear_session()
    set_seed()
    t0 = time.time()
    train_ds = make_ds("train", train=True, img_size=cfg["img_size"], aug=cfg["aug"])
    val_ds = make_ds("val", img_size=cfg["img_size"])

    model = compile_model(build_transfer(cfg["img_size"], cfg["dropout"], cfg["backbone"]),
                          cfg["lr1"])
    h1, b1 = fit_stage(model, f"exp_{name}_s1", train_ds, val_ds, 25)

    unfreeze_last_block(model, cfg["start"] or None)
    model = compile_model(model, cfg["lr2"])
    list_trainable(model, f"exp_{name}_s2")
    h2, b2 = fit_stage(model, f"exp_{name}_s2", train_ds, val_ds, 20)

    i = b2 - 1
    row = {"name": name, **cfg,
           "s1_best_epoch": b1, "s1_val_loss": min(h1["val_loss"]),
           "s2_best_epoch": b2, "val_loss": h2["val_loss"][i],
           "val_acc": h2["val_accuracy"][i], "val_auc": h2["val_auc"][i],
           "val_precision": h2["val_precision"][i], "val_recall": h2["val_recall"][i],
           "minutes": round((time.time() - t0) / 60, 1)}
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(EXPERIMENTS)):
        run(n)
    print(pd.read_csv(CSV)[["name", "val_loss", "val_acc", "val_auc", "minutes"]].round(4))