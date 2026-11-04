from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

Path("figures").mkdir(exist_ok=True)
df = pd.read_csv("audit.csv")
df = df[~df.corrupt & ~df.is_duplicate]
classes = sorted(df["class"].unique())

# Sample grids (4x4 per class)
for c in classes:
    sample = df[df["class"] == c].sample(16, random_state=42)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, p in zip(axes.ravel(), sample["path"]):
        with Image.open(p) as im:
            ax.imshow(im.convert("RGB"))
        ax.axis("off")
    fig.suptitle(f"Random samples: {c}")
    plt.tight_layout()
    fig.savefig(f"figures/samples_{c}.png", dpi=120)
    plt.close(fig)

# Histograms per class
for col, log in [("width", False), ("height", False), ("file_size", True)]:
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in classes:
        ax.hist(df[df["class"] == c][col], bins=40, alpha=0.6, label=c)
    ax.set_xlabel(col + (" (bytes)" if col == "file_size" else " (px)"))
    ax.set_ylabel("Number of images")
    if log:
        ax.set_xscale("log")
    ax.legend()
    ax.set_title(f"{col} by class")
    fig.tight_layout()
    fig.savefig(f"figures/hist_{col}.png", dpi=120)
    plt.close(fig)
print("Saved figures to figures/")