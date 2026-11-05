import tensorflow as tf

IMG_SIZE = 160
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE


def read_split(name):
    paths, labels = [], []
    with open(f"splits/{name}.txt") as f:
        for line in f:
            p, l = line.strip().rsplit(",", 1)
            paths.append(p)
            labels.append(int(l))
    return paths, labels


def load_image(path, label, size=IMG_SIZE):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))      # squashes aspect ratio
    img.set_shape((size, size, 3))
    return img, tf.cast(label, tf.float32)


def build_augment(level="default"):
    L = tf.keras.layers
    if level == "flip":
        parts = [L.RandomFlip("horizontal")]
    else:
        parts = [L.RandomFlip("horizontal"), L.RandomRotation(0.05),
                 L.RandomZoom(0.1), L.RandomContrast(0.1)]
    return tf.keras.Sequential(parts, name="augmentation")


augment = build_augment("default")


def make_ds(name, train=False, seed=42, img_size=IMG_SIZE, aug="default"):
    paths, labels = read_split(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: load_image(p, l, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH)
    if train and aug != "none":
        aug_layer = build_augment(aug)
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)