import matplotlib.pyplot as plt
import numpy as np
from data import make_ds

ds = make_ds("train", train=True)
x, y = next(iter(ds))
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for ax, img, lab in zip(axes.ravel(), x.numpy(), y.numpy()):
    ax.imshow(np.clip(img, 0, 255).astype("uint8"))
    ax.set_title("AI" if lab == 1 else "Real")
    ax.axis("off")
fig.suptitle("One augmented training batch (160x160)")
plt.tight_layout()
plt.savefig("figures/augmented_batch.png", dpi=120)
print("Saved")
