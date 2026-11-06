from mlflow import MlflowClient
import tracking_mlflow as tracking

tracking.setup()
client = MlflowClient()
exp_ids = [e.experiment_id for e in client.search_experiments()]
for r in client.search_runs(exp_ids, max_results=100, order_by=["attributes.start_time DESC"]):
    print(r.info.run_id, "|", r.info.run_name, "|", r.info.status,
          "| val_loss =", r.data.metrics.get("val_loss"))