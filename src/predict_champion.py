import os, sys
import numpy as np
import mlflow
import tensorflow as tf
import tracking_mlflow as tracking

MODEL_NAME = "23L-0752_A2_Ops"
IMG_SIZE = 160
THRESHOLD = float(os.getenv("THRESHOLD", "0.46"))   # val-chosen in 2.5 for config D

tracking.setup()
uri = f"models:/{MODEL_NAME}@champion"
try:
    model = mlflow.keras.load_model(uri)
except Exception:
    model = mlflow.pyfunc.load_model(uri)
print("Loaded", uri)


def preprocess(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))        # same as training; no rescale here
    return tf.expand_dims(img, 0).numpy().astype("float32")


if len(sys.argv) > 1:
    path, truth = sys.argv[1], "unknown"
else:                                                       # default: first validation image
    line = open("splits/val.txt").readline().strip()
    path, lab = line.rsplit(",", 1)
    truth = "AI" if lab == "1" else "Real"

p = float(np.ravel(model.predict(preprocess(path)))[0])
print(f"image: {path}")
print(f"p(AI) = {p:.3f} | threshold = {THRESHOLD} | prediction = {'AI' if p >= THRESHOLD else 'Real'} | true label = {truth}")