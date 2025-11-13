import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shap
import numpy as np
from app.load_model import load_model
from app.inference import preprocess_single
import warnings
warnings.filterwarnings("ignore")

class ModelService:
    def __init__(self, model_name="flare_detector_v1", stage=None):
        self.clf, self.tfidf, self.svd, self.scaler, self.artifacts_dir = load_model(model_name=model_name, stage=stage)
        self.explainer = shap.TreeExplainer(self.clf)
        self.feature_names = (
            ["patient_age", "has_psoriasis", "on_steroid_med", "on_biologic",
             "itch_present", "dry_skin", "plaques_present", "silvery_scale",
             "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
             "family_melanoma"] +
            [f"svd_{i}" for i in range(self.svd.n_components)]
        )

    def predict_note(self, raw_note: dict):
        X, debug = preprocess_single(raw_note, self.tfidf, self.svd, self.scaler)
        proba = float(self.clf.predict_proba(X)[:, 1][0])
        label = int(proba >= 0.5)


        if proba < 0.33:
            risk_level = "Low"
        elif proba < 0.67:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  

        abs_vals = np.abs(shap_values[0])
        top_idx = np.argsort(abs_vals)[-5:][::-1]
        important_feats = [
            {
                "feature": self.feature_names[i],
                "impact": round(float(shap_values[0][i]), 4),
                "direction": "↑ flare risk" if shap_values[0][i] > 0 else "↓ flare risk"
            }
            for i in top_idx
        ]

        return {
            "patientId": raw_note.get("patientId"),
            "noteDate": raw_note.get("noteDate"),
            "flare_probability": round(proba, 3),
            "flare_label": label,
            "flare_risk_level": risk_level,
            "key_influences": important_feats,
            "explanation_summary": (
                f"Model predicted {risk_level} risk of flare "
                f"(probability {round(proba*100, 1)}%). "
                f"Top influencing factors include: "
                + ", ".join([f["feature"] for f in important_feats])
            ),
            "debug": debug
        }

