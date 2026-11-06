"""Task 3.1 / 3.2 toy run: tiny Keras model on Fashion-MNIST, tracked with MLflow.

  python src/mlflow_toy.py local   -> local MLflow (sqlite file mlflow_toy.db)
  python src/mlflow_toy.py         -> tracking server from .env (DagsHub)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

SEED = 42
UNITS, EPOCHS, BATCH = 64, 5, 64
LEARNING_RATES = [1e-2, 1e-4]            # two runs that differ only in learning rate

load_dotenv()                             # reads .env if it exists, otherwise does nothing
if len(sys.argv) > 1 and sys.argv[1] == "local":
    mlflow.set_tracking_uri("sqlite:///mlflow_toy.db")
else:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        sys.exit("MLFLOW_TRACKING_URI is not set. Create .env, or run: python src/mlflow_toy.py local")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])   # URI, user and token come from env vars
print("Tracking URI:", mlflow.get_tracking_uri())
mlflow.set_experiment("toy-fashion-mnist")

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()           # test split unused
x = x.astype("float32") / 255.0
x_train, y_train = x[:10000], y[:10000]
x_val, y_val = x[50000:52000], y[50000:52000]


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual per-epoch logging: one metric value per epoch, with the epoch as the step."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


for lr in LEARNING_RATES:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(UNITS, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=f"toy_lr_{lr:g}"):
        mlflow.set_tags({"purpose": "tool-lab", "dataset": "fashion-mnist"})
        mlflow.log_params({"learning_rate": lr, "units": UNITS, "epochs": EPOCHS,
                           "batch_size": BATCH, "seed": SEED})
        h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS,
                      batch_size=BATCH, callbacks=[MlflowEpochLogger()], verbose=2)
        mlflow.log_metric("final_val_accuracy", h.history["val_accuracy"][-1])

        fig, ax = plt.subplots(figsize=(5, 3.5))                  # one plot as an artifact
        ax.plot(h.history["loss"], label="train loss")
        ax.plot(h.history["val_loss"], label="validation loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(f"lr = {lr:g}"); ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        mlflow.keras.log_model(model, name="model")               # the trained model