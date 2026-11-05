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


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.BinaryCrossentropy(),
        metrics=["accuracy", metrics.AUC(name="auc"),
                 metrics.Precision(name="precision"), metrics.Recall(name="recall")],
    )
    return model