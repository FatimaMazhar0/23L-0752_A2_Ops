import mlflow
import tracking_mlflow as tracking

tracking.setup()
with mlflow.start_run(run_id="4d40317492b743f0be512fefc0c65760"):   # D_lr2_1e-4_manual
    mlflow.log_artifact("models/mlflow_D_lr2_1e-4_manual_s2.keras", artifact_path="model_file")