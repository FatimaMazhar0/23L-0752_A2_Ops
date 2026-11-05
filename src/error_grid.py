import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

df = pd.read_csv("figures/test_predictions_mnv2.csv")
wrong = df[df.label != df.pred].copy()
wrong["conf"] = (wrong.prob_ai - wrong.label).abs()      # distance from the true label
fn = wrong[wrong.label == 1].nlargest(5, "conf")         # AI called Real
fp = wrong[wrong.label == 0].nlargest(5, "conf")         # Real called AI
sel = pd.concat([fn, fp])
if len(sel) < 10:
    sel = pd.concat([sel, wrong.drop(sel.index).nlargest(10 - len(sel), "conf")])

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for ax, (_, r) in zip(axes.ravel(), sel.iterrows()):
    ax.imshow(Image.open(r.path).convert("RGB"))
    kind = "AI -> Real" if r.label == 1 else "Real -> AI"
    ax.set_title(f"{kind}\np(AI)={r.prob_ai:.2f}\n{os.path.basename(r.path)[:22]}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("figures/error_grid.png", dpi=120)
sel.to_csv("figures/error_cases.csv", index=False)
print(sel[["path", "label", "prob_ai"]])