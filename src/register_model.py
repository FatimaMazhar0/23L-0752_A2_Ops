import sys
import mlflow
from mlflow import MlflowClient
import tracking_mlflow as tracking

MODEL_NAME = "23L-0752_A2_Ops"

tracking.setup()
client = MlflowClient()

# usage: python src\register_model.py <challenger_run_id> <champion_run_id>
challenger_run, champion_run = sys.argv[1], sys.argv[2]
plan = [
    (challenger_run, "challenger", "C_aug_flip: flip-only augmentation"),
    (champion_run, "champion", "D_lr2_1e-4: stage-2 lr 1e-4, selected in 3.5"),
]

exp_ids = [e.experiment_id for e in client.search_experiments()]
all_models = list(client.search_logged_models(experiment_ids=exp_ids, max_results=500))


def models_for(run_id):
    ms = [m for m in all_models if getattr(m, "source_run_id", None) == run_id]
    return sorted(ms, key=lambda m: getattr(m, "creation_timestamp", 0))


# 1) resolve both runs first, so nothing is registered unless both are found
resolved, missing = [], False
for run_id, alias, note in plan:
    ms = models_for(run_id)
    print(f"run {run_id[:8]} ({alias}): logged models = {[(m.name, m.model_id) for m in ms]}")
    if not ms:
        missing = True
    else:
        resolved.append((run_id, alias, note, ms[-1]))     # newest = final stage-2 model
if missing:
    print("No logged model found for at least one run. Nothing was registered.")
    sys.exit(1)

# 2) register and set aliases
for run_id, alias, note, m in resolved:
    mv = mlflow.register_model(f"models:/{m.model_id}", MODEL_NAME)
    client.update_model_version(MODEL_NAME, mv.version, description=note)
    client.set_model_version_tag(MODEL_NAME, mv.version, "source_run", run_id)
    client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    print(f"registered version {mv.version} -> alias '{alias}' ({note})")

for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print("version", v.version, "| run", (v.run_id or "")[:8], "|", v.description)
champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
print("champion is version", champ.version)