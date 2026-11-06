"""Task 3.5: collect the comparison runs from MLflow into one table and one chart."""
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tracking
from train_mlflow import CHANGES

tracking.setup()
runs = mlflow.search_runs(
    experiment_names=["visionguard"], filter_string="tags.task = '3.5'",
    order_by=["attributes.start_time DESC"])
if runs.empty:
    raise SystemExit("No runs with tag task=3.5 found.")
runs = runs[runs["status"] == "FINISHED"].drop_duplicates("tags.config")   # newest run per config

df = runs[["tags.config", "metrics.val_loss", "metrics.val_auc", "metrics.val_accuracy",
           "metrics.val_precision", "metrics.val_recall", "metrics.best_epoch_stage2",
           "metrics.train_minutes", "tags.change", "run_id"]].copy()
df.columns = ["config", "val_loss", "val_auc", "val_accuracy", "val_precision", "val_recall",
              "best_epoch_s2", "minutes", "change", "run_id"]
df = df.sort_values("val_loss").reset_index(drop=True)
df.to_csv("figures/mlflow_comparison.csv", index=False)
print(df.drop(columns=["run_id"]).round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, col, title in zip(ax, ["val_loss", "val_auc"], ["Validation loss (lower is better)",
                                                       "Validation AUC (higher is better)"]):
    a.barh(df["config"][::-1], df[col][::-1])
    a.set_title(title)
    lo, hi = df[col].min(), df[col].max()
    pad = (hi - lo) * 0.5 + 0.005
    a.set_xlim(lo - pad, hi + pad)
plt.tight_layout()
plt.savefig("figures/mlflow_comparison.png", dpi=120)
print("Saved figures/mlflow_comparison.csv and figures/mlflow_comparison.png")