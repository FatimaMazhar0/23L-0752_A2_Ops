from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("audit.csv")
df = df[~df.corrupt & ~df.is_duplicate].copy()
df["label"] = (df["class"] == "AiArtData").astype(int)

train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

Path("splits").mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    with open(f"splits/{name}.txt", "w") as f:
        for p, l in zip(part["path"], part["label"]):
            f.write(f"{p},{l}\n")
    print(name, len(part), part["label"].value_counts().to_dict())

assert not (set(train.path) & set(val.path) | set(train.path) & set(test.path) | set(val.path) & set(test.path))
print("No overlap between splits.")