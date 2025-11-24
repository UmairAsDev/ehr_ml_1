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
        

        proba = float(self.clf.predict_proba(X)[:, 1][0])
        label = int(proba >= 0.5)
        

        if proba < 0.33:
            risk_level = "Low"
        elif proba < 0.67:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        

        shap_values = self.explainer.shap_values(X, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  
        
        shap_values = shap_values.reshape(1, -1)
        abs_vals = np.abs(shap_values[0])
        top_idx = np.argsort(abs_vals)[-5:][::-1]


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
        notes: list of raw note dicts (same structure as preprocess_single input)
        returns: patient-level aggregated prediction + sorted list of risky notes
        """
        per_note_results = []
        flare_labels = []
        flare_scores = []

        INTERPRETABLE_FEATURES = [
            "has_psoriasis", "on_steroid_med", "on_biologic",
            "itch_present", "dry_skin", "plaques_present", "silvery_scale",
            "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
            "family_melanoma",
            "complaint_flare_kw", "complaint_no_relief", "flare_in_assessment",
            "trigger_mentioned", "steroid_started", "has_medications"
        ]

        for raw in notes:
            result = self.predict_note(raw)
    
            filtered_influences = [
                f for f in result["key_influences"] if f["feature"] in INTERPRETABLE_FEATURES
            ]
            result_clean = {
                "noteId": raw.get("noteId"),
                "noteDate": result.get("noteDate"),
                "flare_probability": result.get("flare_probability"),
                "flare_risk_level": result.get("flare_risk_level"),
                "key_influences": filtered_influences,
                "text_signals": {f["feature"]: f["impact"] for f in filtered_influences if "complaint" in f["feature"] or "flare" in f["feature"]}
            }
            per_note_results.append(result_clean)
            flare_labels.append(result["flare_label"])
            flare_scores.append(result["flare_probability"])


        if len(flare_labels) == 0:
            final_label = None
            final_probability = None
        else:
            final_label = 1 if sum(flare_labels) > len(flare_labels) / 2 else 0
            final_probability = float(np.mean(flare_scores))

        if final_probability is not None:
            if final_probability < 0.33:
                patient_risk = "Low"
            elif final_probability < 0.50:
                patient_risk = "Moderate"
            elif final_probability < 0.67:
                patient_risk = "Elevated"
            else:
                patient_risk = "High"
        else:
            patient_risk = "Unknown"

        risky_notes_sorted = sorted(per_note_results, key=lambda x: x["flare_probability"], reverse=True)

        return {
            "patientId": patient_id,
            "total_notes": len(notes),
            "final_flare_label": final_label,
            "final_risk_level": patient_risk,
            "risky_notes": risky_notes_sorted
        }
