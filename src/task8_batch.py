import os
import sys

sys.path.insert(0, os.getcwd())          # run from the project root

from werkzeug.utils import secure_filename  # noqa: E402

from visionguard import config, create_app, db  # noqa: E402

N = 5
LIMIT = config.MAX_UPLOAD_MB * 1024 * 1024
if config.MODEL_BACKEND != "remote":
    raise SystemExit("Set MODEL_BACKEND=remote, MODEL_API_URL and MODEL_API_KEY first.")

picked, count = [], {"0": 0, "1": 0}
for line in open("splits/val.txt"):
    path, lab = line.strip().rsplit(",", 1)
    if os.path.getsize(path) > LIMIT:        # skip files over the upload limit
        continue
    if count[lab] < N:
        count[lab] += 1
        picked.append((path, lab))
    if min(count.values()) == N:
        break

client = create_app().test_client()
before, _ = db.recent(1)
last_id = before[0].id if before else 0

truth, failed = {}, 0
for path, lab in picked:
    base = os.path.basename(path)
    truth[secure_filename(base)] = "AI" if lab == "1" else "Real"
    with open(path, "rb") as f:
        r = client.post("/predict", data={"image": (f, base)})
    failed += r.status_code != 200
    print(f"POST /predict {base[:40]:40s} -> HTTP {r.status_code}")

rows, _ = db.recent(len(picked) + 20)
new = [p for p in reversed(rows) if p.id > last_id]
print(f"\n{'#':>3} {'file':34} {'true':5} {'decision':8} {'p(AI)':>6} {'conf':>6}  model")
right = 0
for p in new:
    t = truth.get(p.filename, "?")
    right += t == p.label
    print(f"{p.id:>3} {p.filename[:34]:34} {t:5} {p.label:8} {p.p_ai:6.3f} "
          f"{p.confidence:6.3f}  {p.model_name} v{p.model_version}  {'ok' if t == p.label else 'WRONG'}")
print(f"\n{len(picked)} images sent, {len(new)} saved to the database, "
      f"{failed} rejected, {right} of {len(new)} correct")