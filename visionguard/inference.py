import os
import threading
import time

import numpy as np
import tensorflow as tf

from . import config

_model = None
_lock = threading.Lock()


class ImageDecodeError(ValueError):
    pass


def load_model():
    """Load the Keras model once (called at app start-up) and warm it up."""
    global _model
    if _model is None:
        if not os.path.isfile(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {config.MODEL_PATH} "
                "(set MODEL_PATH or run src/export_champion.py)")
        _model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
        _model(np.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 3), "float32"), training=False)
    return _model


def model_loaded():
    return _model is not None


def preprocess(data: bytes):
    """Exactly the training steps from data.py: decode to 3 channels, resize, float32 0-255.
    No rescaling here: the model has its own preprocessing layer inside."""
    try:
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
    except (tf.errors.InvalidArgumentError, ValueError) as e:
        raise ImageDecodeError("image could not be decoded") from e
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    return tf.expand_dims(img, 0).numpy().astype("float32")


def predict(data: bytes):
    x = preprocess(data)
    with _lock:                                   # one prediction at a time
        p = float(_model(x, training=False).numpy()[0][0])
    is_ai = p >= config.THRESHOLD
    return {"label": "AI" if is_ai else "Real",
            "p_ai": p,
            "confidence": p if is_ai else 1.0 - p,
            "threshold": config.THRESHOLD}


if __name__ == "__main__":                        # console evidence for 4.1
    import sys
    paths, truth = sys.argv[1:], {}
    if not paths:                                 # default: first AI and first Real image of the validation split
        seen = {}
        for line in open("splits/val.txt"):
            p, l = line.strip().rsplit(",", 1)
            seen.setdefault(l, p)
            if len(seen) == 2:
                break
        paths = [seen["1"], seen["0"]]
        truth = {seen["1"]: "AI", seen["0"]: "Real"}
    t0 = time.perf_counter()
    load_model()
    print(f"model loaded once in {time.perf_counter() - t0:.2f} s")
    for p in paths:
        with open(p, "rb") as f:
            raw = f.read()
        t0 = time.perf_counter()
        r = predict(raw)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{p}\n  -> {r['label']} (confidence {r['confidence']:.3f}, p_ai {r['p_ai']:.3f}) "
              f"| true label: {truth.get(p, '?')} | {ms:.0f} ms")