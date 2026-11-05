import os
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

os.makedirs("figures", exist_ok=True)
tf.keras.utils.set_random_seed(42)

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()   # official test split is ignored
x = x.astype("float32") / 255.0
x_train, y_train = x[:50000], y[:50000]
x_val, y_val = x[50000:], y[50000:]


def build(hp):
    units = hp.Int("units", 32, 256, step=32)
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=10, seed=42,
                        directory="tuner", project_name="fashion_toy", overwrite=True)
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val), verbose=0)

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
        for t in tuner.oracle.get_best_trials(10)]
df = pd.DataFrame(rows).round(4)
df.to_csv("figures/tuner_toy_results.csv", index=False)
print(df.to_string(index=False))
print("Best settings:", tuner.get_best_hyperparameters(1)[0].values)