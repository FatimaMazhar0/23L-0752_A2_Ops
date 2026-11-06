import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from visionguard import config

engine = create_engine(config.DATABASE_URL)

QUERIES = [
    ("Latest 5 decisions (what, when, which model, how confident)",
     "SELECT id, created_at, filename, label, confidence, model_name, model_version "
     "FROM predictions ORDER BY id DESC LIMIT 5"),
    ("Decisions per label",
     "SELECT label, COUNT(*) AS n, AVG(confidence) AS avg_confidence "
     "FROM predictions GROUP BY label ORDER BY label"),
    ("Checks per model version",
     "SELECT model_name, model_version, COUNT(*) AS n, AVG(p_ai) AS avg_p_ai "
     "FROM predictions GROUP BY model_name, model_version"),
    ("Borderline cases (p_ai within 0.05 of the threshold)",
     "SELECT id, filename, label, p_ai, threshold FROM predictions "
     "WHERE ABS(p_ai - threshold) < 0.05 ORDER BY ABS(p_ai - threshold)"),
    ("Same image checked more than once (by SHA-256)",
     "SELECT SUBSTR(sha256, 1, 12) AS sha_prefix, COUNT(*) AS times "
     "FROM predictions GROUP BY sha256 HAVING COUNT(*) > 1"),
    ("Time span",
     "SELECT MIN(created_at) AS first_check, MAX(created_at) AS last_check, COUNT(*) AS total "
     "FROM predictions"),
]

for title, sql in QUERIES:
    with engine.connect() as c:
        res = c.execute(text(sql))
        cols, rows = list(res.keys()), res.fetchall()
    print(f"\n== {title} ==")
    print(" | ".join(cols))
    if not rows:
        print("(no rows)")
    for r in rows:
        print(" | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r))