import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, losses


def build_baseline(img_size=160, dropout=0.4, filters=(32, 64, 128, 256)):
    inp = layers.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(inp)
    for i, f in enumerate(filters, 1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return models.Model(inp, out, name="baseline_cnn")


def build_transfer(img_size=160, dropout=0.3, backbone="mobilenetv2"):
    shape = (img_size, img_size, 3)
    if backbone == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=shape, include_top=False, weights="imagenet")
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=shape, include_top=False, weights="imagenet")
    base.trainable = False                       # stage 1: backbone frozen

    inp = layers.Input(shape, name="image")
    if backbone == "efficientnetb0":
        x = inp                                  # EfficientNet has its own rescaling inside (expects 0-255)
    else:
        x = layers.Rescaling(1 / 127.5, offset=-1, name="mnv2_preprocess")(inp)
    x = base(x, training=False)                  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return models.Model(inp, out, name=f"{backbone}_transfer")


def unfreeze_last_block(model, start=None):
    """Unfreeze the backbone from the layer whose name starts with `start` onwards.
    BatchNorm layers stay frozen."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    if start is None:
        start = "block7a" if "efficientnet" in base.name else "block_16"
    base.trainable = True
    in_last = False
    for l in base.layers:
        if l.name.startswith(start):
            in_last = True
        l.trainable = in_last and not isinstance(l, layers.BatchNormalization)
    return model

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.BinaryCrossentropy(),
        metrics=["accuracy", metrics.AUC(name="auc"),
                 metrics.Precision(name="precision"), metrics.Recall(name="recall")],
    )
    return model