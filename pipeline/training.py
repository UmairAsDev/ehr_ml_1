import pandas as pd
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import FeatureExtraction
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support


log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_directory, "training.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)









safe_numeric_cols = [
    "patient_age", "has_psoriasis", "on_steroid_med", "on_biologic",
    "itch_present", "dry_skin", "plaques_present", "silvery_scale",
    "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
    "family_melanoma"
]






def train_model_from_features() -> None:
    """Train a model from features"""
    feature_extraction = FeatureExtraction()
    df = feature_extraction.extract_features()
    X_train, X_test, y_train, y_test = feature_extraction.split_data(df) #type:ignore
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
    y_proba = clf.predict_proba(X_test)[:,1] #type:ignore
    print(classification_report(y_test, y_pred)) #type:ignore
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))



    




if __name__ == "__main__":
    train_model_from_features()