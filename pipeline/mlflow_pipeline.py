import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mlflow
import joblib
import pandas as pd
import logging
from preprocessing import FeatureExtraction
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
import lightgbm as lgb

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlflow_pipeline.log"),
        logging.StreamHandler()
    ]
)

mlflow.set_experiment("flare_detection_pipeline")

def run_pipeline():
    with mlflow.start_run(run_name="feature_extraction_and_training") as run:
        logger.info("Starting Feature Extraction...")

        feature_extraction = FeatureExtraction()
        df = feature_extraction.extract_features()
        if df is None:
            logger.error("Feature extraction returned None.")
            return

        mlflow.log_param("n_rows", len(df))
        mlflow.log_metric("flare_signal_rate", df['flare_signal'].mean())
        mlflow.log_metric("steroid_use_rate", df['any_steroid_use'].mean())

        feature_path = "/tmp/features_v1.parquet"
        df.to_parquet(feature_path, index=False)
        mlflow.log_artifact(feature_path, "features")
        logger.info("Features logged to MLflow.")

        logger.info("Splitting data and saving preprocessing objects...")
        X_train, X_test, y_train, y_test = feature_extraction.split_data(df)

        preproc_dir = "/tmp/preproc"
        for f in ["tfidf.joblib", "svd.joblib", "scaler.joblib"]:
            mlflow.log_artifact(os.path.join(preproc_dir, f), "preprocessing")

        logger.info("Training LightGBM model...")
        clf = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            class_weight='balanced',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] #type:ignore

        report = classification_report(y_test, y_pred, output_dict=True) #type:ignore
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({
            "roc_auc": auc,
            "precision": report["weighted avg"]["precision"], #type:ignore
            "recall": report["weighted avg"]["recall"], #type:ignore
            "f1": report["weighted avg"]["f1-score"] #type:ignore
        })#type:ignore

        logger.info(f"Model training complete. ROC-AUC = {auc:.4f}")

        model_path = "/tmp/lgbm_model.pkl"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path, "model_files")
        mlflow.lightgbm.log_model(clf, artifact_path="model")#type:ignore


        logger.info(f"Model training complete. ROC-AUC = {auc:.4f}")


        mlflow.register_model(f"runs:/{run.info.run_id}/model", "flare_detector_v1")

        logger.info("Model and artifacts logged successfully.")




if __name__ == "__main__":
    run_pipeline()