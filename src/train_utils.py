import json, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 42


def set_seed(seed=SEED):
    tf.keras.utils.set_random_seed(seed)   # python, numpy and TF seeds in one call


def get_callbacks(tag, patience=5):
    os.makedirs("models", exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(f"models/{tag}.keras", monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-7, verbose=1),
    ]


def list_trainable(model, tag):
    lines = []
    for l in model.layers:
        if isinstance(l, tf.keras.Model):          # the MobileNetV2 backbone
            if not l.trainable:
                lines.append(f"{l.name}: FROZEN ({len(l.layers)} layers)")
            else:
                for s in l.layers:
                    if s.trainable and s.weights:
                        lines.append(f"{l.name}/{s.name}")
        elif l.trainable and l.weights:
            lines.append(l.name)
    n_train = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    n_total = model.count_params()
    lines.append(f"Trainable params: {n_train:,} of {n_total:,}")
    os.makedirs("figures", exist_ok=True)
    open(f"figures/{tag}_trainable.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


def plot_curves(h, best, tag):
    ep = range(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k, t in zip(ax, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        a.plot(ep, h[k], label="train")
        a.plot(ep, h["val_" + k], label="validation")
        a.axvline(best, color="red", ls="--", label=f"best epoch = {best}")
        a.set_xlabel("Epoch"); a.set_ylabel(t); a.set_title(f"{tag}: {t}"); a.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{tag}_curves.png", dpi=120)
    plt.close(fig)


def fit_stage(model, tag, train_ds, val_ds, epochs, extra_callbacks=()):
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=get_callbacks(tag) + list(extra_callbacks), verbose=2)    
    h = hist.history
    best = int(np.argmin(h["val_loss"])) + 1
    json.dump({"history": h, "best_epoch": best},
              open(f"figures/{tag}_history.json", "w"), default=float)
    plot_curves(h, best, tag)
    print(f"[{tag}] best epoch = {best}, val_loss = {min(h['val_loss']):.4f}, "
          f"val_acc = {h['val_accuracy'][best-1]:.4f}")
    return h, best