import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlflow
import joblib
from mlflow.tracking import MlflowClient
import warnings

warnings.filterwarnings("ignore")


def load_model(model_name="flare_detector_v1", stage=None):
    client = MlflowClient()

    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

    latest_run_id = latest_version.run_id

    print(f"Loading artifacts from Run ID: {latest_run_id}")

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "file://"
            + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns")),
        )
    )

    if stage:

        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise RuntimeError(
                f"No model versions found for {model_name} in stage {stage}"
            )
        run_id = versions[0].run_id
    else:

        versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
            raise RuntimeError(f"No runs found for registered model {model_name}")
        run_id = versions[0].run_id

    print(f"[model_loader] Loading artifacts from Run ID: {run_id}")

    local_artifacts_dir = mlflow.artifacts.download_artifacts(  # type: ignore
        run_id=latest_run_id, artifact_path=None
    )

    MODEL_PATH = os.path.join(local_artifacts_dir, "model_files", "lgbm_model.pkl")
    TFIDF_PATH = os.path.join(local_artifacts_dir, "preprocessing", "tfidf.joblib")
    SVD_PATH = os.path.join(local_artifacts_dir, "preprocessing", "svd.joblib")
    SCALER_PATH = os.path.join(local_artifacts_dir, "preprocessing", "scaler.joblib")

    clf = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    scaler = joblib.load(SCALER_PATH)

    return clf, tfidf, svd, scaler, local_artifacts_dir


if __name__ == "__main__":

    clf, tfidf, svd, scaler, local_artifacts_dir = load_model()
    print("Loaded model and preprocessing objects successfully.")
