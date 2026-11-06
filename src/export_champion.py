import os, shutil
import mlflow
import tensorflow as tf
import tracking_mlflow as tracking

NAME = "23L-0752_A2_Ops"
OUT = "models/champion.keras"
uri = f"models:/{NAME}@champion"

tracking.setup()
os.makedirs("models", exist_ok=True)
try:
    model = mlflow.keras.load_model(uri)
    model.save(OUT)
except Exception as e:                      # fallback: copy the .keras file out of the artifacts
    print("keras load failed:", e)
    d = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path="models/_download")
    found = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".keras")]
    print("found:", found)
    if not found:
        raise SystemExit("no .keras file found in the model artifacts")
    shutil.copy(found[0], OUT)

m = tf.keras.models.load_model(OUT, compile=False)
print("saved", OUT, "| size MB:", round(os.path.getsize(OUT) / 1e6, 1),
      "| input:", m.input_shape, "| output:", m.output_shape)