import os
import sys
import numpy as np
import shap
from app.load_model import load_model
from app.inference import preprocess_single
import warnings
warnings.filterwarnings("ignore")

class ModelService:
    def __init__(self, model_name="flare_detector_v1", stage=None):
        self.clf, self.tfidf, self.svd, self.scaler, self.artifacts_dir = load_model(
            model_name=model_name, stage=stage
        )
        

        self.explainer = shap.TreeExplainer(self.clf, model_output="raw")
        

        self.numeric_features = [
            "patient_age", "has_psoriasis", "on_steroid_med", "on_biologic",
            "itch_present", "dry_skin", "plaques_present", "silvery_scale",
            "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
            "family_melanoma"
        ]
        self.svd_features = [f"svd_{i}" for i in range(self.svd.n_components)]
        self.feature_names = self.numeric_features + self.svd_features

    def predict_note(self, raw_note: dict, hide_svd: bool = True):
        """
        Predicts one note with optimized SHAP.
        hide_svd=True will group all text features as one 'text_signal'.
        """
        X, debug = preprocess_single(raw_note, self.tfidf, self.svd, self.scaler)
        
        # Predict probability & label
        proba = float(self.clf.predict_proba(X)[:, 1][0])
        label = int(proba >= 0.5)
        
        # Risk level
        if proba < 0.33:
            risk_level = "Low"
        elif proba < 0.67:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        # SHAP values (fast, single-row)
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        
        shap_values = shap_values.reshape(1, -1)
        abs_vals = np.abs(shap_values[0])
        top_idx = np.argsort(abs_vals)[-5:][::-1]

        # Prepare readable explanations
        important_feats = []
        for i in top_idx:
            name = self.feature_names[i]
            impact = round(float(shap_values[0][i]), 4)
            if hide_svd and name.startswith("svd_"):
                name = "text_signal"
            important_feats.append({
                "feature": name,
                "impact": impact,
                "direction": "↑ flare risk up" if shap_values[0][i] > 0 else "↓ flare risk down"
            })

        # Remove duplicate text_signal entries if hide_svd
        if hide_svd:
            seen = set()
            unique_feats = []
            for f in important_feats:
                if f["feature"] not in seen:
                    seen.add(f["feature"])
                    unique_feats.append(f)
            important_feats = unique_feats

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
                f"Top influencing factors: "
                + ", ".join([f["feature"] for f in important_feats])
            ),
            "debug": debug
        }

    def predict_patient_notes(self, notes: list[dict], patient_id: str):
        """
        Aggregate prediction for all notes of a patient.
        """
        per_note_results = []
        flare_labels = []
        flare_probs = []

        for note in notes:
            result = self.predict_note(note)
            per_note_results.append(result)
            flare_labels.append(result["flare_label"])
            flare_probs.append(result["flare_probability"])

        # Patient-level aggregation
        if not flare_labels:
            final_label, final_prob, patient_risk = None, None, "Unknown"
        else:
            final_label = int(sum(flare_labels) > len(flare_labels)/2)
            final_prob = float(np.mean(flare_probs))
            if final_prob < 0.33:
                patient_risk = "Low"
            elif final_prob < 0.67:
                patient_risk = "Moderate"
            else:
                patient_risk = "High"

        return {
            "patientId": patient_id,
            "total_notes": len(notes),
            "final_flare_label": final_label,
            "final_flare_probability": round(final_prob,3) if final_prob is not None else None,
            "final_risk_level": patient_risk,
            "notes": per_note_results
        }
