import sys
import mlflow
from mlflow import MlflowClient
import tracking_mlflow as tracking

MODEL_NAME = "23L-0752_A2_Ops"
MODEL_PATH = "model"          # artifact name the model was logged under

tracking.setup()
client = MlflowClient()

# usage: python src\register_model.py <challenger_run_id> <champion_run_id>
challenger_run, champion_run = sys.argv[1], sys.argv[2]

plan = [
    (challenger_run, "challenger", "C_aug_flip: flip-only augmentation"),
    (champion_run, "champion", "D_lr2_1e-4: stage-2 lr 1e-4, selected in 3.5"),
]

for run_id, alias, note in plan:
    mv = mlflow.register_model(f"runs:/{run_id}/{MODEL_PATH}", MODEL_NAME)
    client.update_model_version(MODEL_NAME, mv.version, description=note)
    client.set_model_version_tag(MODEL_NAME, mv.version, "source_run", run_id)
    client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    print(f"registered version {mv.version} -> alias '{alias}' ({note})")

for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print("version", v.version, "| run", v.run_id[:8], "|", v.description)
champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
print("champion is version", champ.version)