import tensorflow as tf

IMG_SIZE = 160          # chosen size, see the notes below
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


def load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))   # squashes aspect ratio
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img, tf.cast(label, tf.float32)


augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")


def make_ds(name, train=False, seed=42):
    paths, labels = read_split(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH)
    if train:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)