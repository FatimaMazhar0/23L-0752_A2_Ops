import os
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from data import make_ds
from models import build_transfer, unfreeze_last_block, compile_model
from train_utils import set_seed

N_TRIALS = 8
os.makedirs("figures", exist_ok=True)


class VisionGuardHyperModel(kt.HyperModel):
    def build(self, hp):
        tf.keras.backend.clear_session()
        set_seed()
        dropout = hp.Float("dropout", 0.2, 0.5, step=0.1)
        hp.Float("lr2", 1e-5, 1e-3, sampling="log")
        hp.Choice("unfrozen_blocks", [1, 2, 3])
        return compile_model(build_transfer(160, dropout), 1e-3)   # stage 1: frozen backbone

    def fit(self, hp, model, train_ds, validation_data=None, callbacks=None, **kwargs):
        def stop(p):
            return tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=p,
                                                    restore_best_weights=True)
        # stage 1: train the new head only
        model.fit(train_ds, validation_data=validation_data, epochs=15,
                  callbacks=[stop(4)], verbose=0)
        # stage 2: unfreeze the last N blocks, smaller learning rate
        start = f"block_{17 - hp.get('unfrozen_blocks')}"
        unfreeze_last_block(model, start)
        compile_model(model, hp.get("lr2"))
        return model.fit(train_ds, validation_data=validation_data, epochs=12,
                         callbacks=[stop(3)] + list(callbacks or []), verbose=0)


train_ds = make_ds("train", train=True)
val_ds = make_ds("val")

tuner = kt.RandomSearch(VisionGuardHyperModel(), objective="val_loss", max_trials=N_TRIALS,
                        seed=42, directory="tuner", project_name="visionguard", overwrite=True)
tuner.search(train_ds, validation_data=val_ds)

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_loss": t.score}
        for t in tuner.oracle.get_best_trials(N_TRIALS)]
df = pd.DataFrame(rows)
df["dropout"] = df["dropout"].round(2)
df["lr2"] = df["lr2"].map(lambda v: float(f"{v:.2e}"))
df["val_loss"] = df["val_loss"].round(4)
df.to_csv("figures/tuner_visionguard_results.csv", index=False)
print(df.to_string(index=False))

print("\nMean val_loss by setting (rough guide only, 8 trials):")
print(df.groupby("unfrozen_blocks")["val_loss"].mean().round(4))
print(df.groupby("dropout")["val_loss"].mean().round(4))
print("Spearman correlation, lr2 vs val_loss:",
      round(df["lr2"].corr(df["val_loss"], method="spearman"), 3))
print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)
print("Reference: config D validation loss = 0.4535")