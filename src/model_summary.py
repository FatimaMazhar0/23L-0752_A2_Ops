from models import build_baseline

m = build_baseline()
m.summary()
rows = ["| Layer | Output shape | Params |", "|---|---|---|"]
for l in m.layers:
    rows.append(f"| {l.name} ({l.__class__.__name__}) | {l.output.shape} | {l.count_params():,} |")
rows.append(f"| **Total** | | **{m.count_params():,}** |")
open("figures/baseline_layers.md", "w").write("\n".join(rows))
print("\n".join(rows))