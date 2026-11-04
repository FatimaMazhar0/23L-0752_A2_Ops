from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image

ROOT = Path("data/raw")
rows = []

for p in sorted(ROOT.rglob("*")):
    if not p.is_file():
        continue
    rec = {"path": p.as_posix(), "class": p.relative_to(ROOT).parts[0],
           "file_size": p.stat().st_size, "corrupt": False, "error": ""}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    rec["sha256"] = h.hexdigest()
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            rec.update(width=im.width, height=im.height,
                       mode=im.mode, format=im.format)
    except Exception as e:
        rec.update(corrupt=True, error=str(e))
    rows.append(rec)

df = pd.DataFrame(rows)
df["is_duplicate"] = df.duplicated("sha256", keep="first")
df["in_dup_group"] = df.duplicated("sha256", keep=False)
df.to_csv("audit.csv", index=False)

ok = df[~df.corrupt]
print("Counts per class:\n", df["class"].value_counts(), "\n")
print("Corrupt files:", int(df.corrupt.sum()))
print("Duplicate extras:", int(df.is_duplicate.sum()))
print("Cross-class duplicate hashes:",
      int((ok.groupby("sha256")["class"].nunique() > 1).sum()))
print(ok.groupby("class")[["width", "height", "file_size"]]
        .agg(["min", "median", "max"]).T)
for col in ["mode", "format"]:
    print(f"\n{col} per class:")
    print(pd.crosstab(ok["class"], ok[col]))