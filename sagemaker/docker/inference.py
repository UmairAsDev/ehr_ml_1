"""
SageMaker-compatible inference handler for psoriasis flare prediction.

This module provides the model serving interface required by AWS SageMaker.
It handles model loading, input/output transformations, and predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import traceback
from io import StringIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model artifacts directory (SageMaker loads model to /opt/ml/model)
MODEL_PATH = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')


class ModelHandler:
    """Handler for model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.svd = None
        self.scaler = None
        self.feature_names = None
        self.loaded = False
        
    def load_model(self):
        """Load model and preprocessing artifacts."""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            
            # Load model artifacts
            self.model = joblib.load(os.path.join(MODEL_PATH, 'lgbm_model.pkl'))
            self.tfidf = joblib.load(os.path.join(MODEL_PATH, 'tfidf.joblib'))
            self.svd = joblib.load(os.path.join(MODEL_PATH, 'svd.joblib'))
            self.scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))
            
            # Define feature names
            self.numeric_features = [
                "patient_age", "has_psoriasis", "on_steroid_med", "on_biologic",
                "itch_present", "dry_skin", "plaques_present", "silvery_scale",
                "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
                "family_melanoma"
            ]
            self.svd_features = [f"svd_{i}" for i in range(self.svd.n_components)]
            self.feature_names = self.numeric_features + self.svd_features
            
            self.loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# Global model handler instance
model_handler = ModelHandler()


def model_fn(model_dir):
    """
    Load the model for inference (SageMaker required function).
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Loaded model handler
    """
    global MODEL_PATH
    MODEL_PATH = model_dir
    
    if not model_handler.loaded:
        model_handler.load_model()
    
    return model_handler


def input_fn(request_body, content_type='application/json'):
    """
    Deserialize and prepare the input data (SageMaker required function).
    
    Args:
        request_body: The request payload
        content_type: The content type of the request
        
    Returns:
        Deserialized input data
    """
    logger.info(f"Received content type: {content_type}")
    
    if content_type == 'application/json':
        data = json.loads(request_body)
        return data
    elif content_type == 'text/csv':
        # Handle CSV input
        df = pd.read_csv(StringIO(request_body))
        return df.to_dict(orient='records')
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def preprocess_single(raw_note, tfidf, svd, scaler):
    """
    Preprocess a single patient note for prediction.
    
    Args:
        raw_note: Dictionary containing patient note data
        tfidf: Fitted TF-IDF vectorizer
        svd: Fitted SVD transformer
        scaler: Fitted standard scaler
        
    Returns:
        Preprocessed feature array
    """
    try:
        df = pd.DataFrame([raw_note]).copy()
        
        # Text columns
        text_cols = ['complaints', 'pastHistory', 'assesment', 'reviewofsystem',
                     'currentmedication', 'procedure', 'allergy', 'examination',
                     'patientSummary', 'diagnoses']
        
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("")
        
        # Feature engineering (same as training)
        df['has_psoriasis'] = df.get('diagnoses', pd.Series([""])).str.contains('L40', case=False, na=False).astype(int)
        
        flare_terms = ['flare', 'worse', 'itch', 'red', 'scaling', 'burning', 'rash', 'lesion']
        
        # Medication features
        df['on_steroid_med'] = df.get('currentmedication', pd.Series([""])).str.contains(
            "steroid|triamcinolone|clobetasol|hydrocortisone", case=False, na=False
        ).astype(int)
        
        df['on_biologic'] = df.get('currentmedication', pd.Series([""])).str.contains(
            "adalimumab|secukinumab|ixekizumab|etanercept", case=False, na=False
        ).astype(int)
        
        # Examination features
        df['plaques_present'] = df.get('examination', pd.Series([""])).str.contains("plaque", case=False, na=False).astype(int)
        df['silvery_scale'] = df.get('examination', pd.Series([""])).str.contains("silvery|scale", case=False, na=False).astype(int)
        df['elbows_involved'] = df.get('examination', pd.Series([""])).str.contains("elbow", case=False, na=False).astype(int)
        df['hyperpigmentation'] = df.get('examination', pd.Series([""])).str.contains("hyperpigment", case=False, na=False).astype(int)
        
        # Review of system features
        df['itch_present'] = df.get('reviewofsystem', pd.Series([""])).str.contains("itch", case=False, na=False).astype(int)
        df['dry_skin'] = df.get('reviewofsystem', pd.Series([""])).str.contains("dry skin", case=False, na=False).astype(int)
        
        # Past history features
        df['smoker'] = df.get('pastHistory', pd.Series([""])).str.contains("smoker", case=False, na=False).astype(int)
        df['alcohol_use'] = df.get('pastHistory', pd.Series([""])).str.contains("alcohol.*yes", case=False, na=False).astype(int)
        df['family_melanoma'] = df.get('pastHistory', pd.Series([""])).str.contains("melanoma.*yes", case=False, na=False).astype(int)
        
        # Patient age extraction
        age_match = df.get('patientSummary', pd.Series([""])).str.extract(r'(\d{1,2})\s*year', expand=False)
        df['patient_age'] = pd.to_numeric(age_match, errors='coerce').fillna(0).astype(float)
        
        # Combine text features
        text_combined = (
            df.get('assesment', "").fillna('') + ' ' +
            df.get('complaints', "").fillna('') + ' ' +
            df.get('examination', "").fillna('')
        ).astype(str)
        
        # TF-IDF and SVD transformation
        X_text_tfidf = tfidf.transform(text_combined)
        X_text_svd = svd.transform(X_text_tfidf)
        
        # Numeric features
        numeric_cols = [
            "patient_age", "has_psoriasis", "on_steroid_med", "on_biologic",
            "itch_present", "dry_skin", "plaques_present", "silvery_scale",
            "elbows_involved", "hyperpigmentation", "smoker", "alcohol_use",
            "family_melanoma"
        ]
        
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
        
        X_num = df[numeric_cols].fillna(0).astype(float).values
        X_num_scaled = scaler.transform(X_num)
        
        # Combine features
        X_final = np.hstack([X_num_scaled, X_text_svd])
        
        return X_final
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def predict_fn(input_data, model_handler):
    """
    Make predictions on the preprocessed data (SageMaker required function).
    
    Args:
        input_data: Preprocessed input data
        model_handler: Loaded model handler
        
    Returns:
        Predictions
    """
    try:
        logger.info("Making predictions...")
        
        # Handle both single note and batch predictions
        if isinstance(input_data, dict) and 'notes' in input_data:
            # Batch prediction
            notes = input_data['notes']
            results = []
            
            for note in notes:
                X = preprocess_single(note, model_handler.tfidf, 
                                    model_handler.svd, model_handler.scaler)
                proba = float(model_handler.model.predict_proba(X)[:, 1][0])
                label = int(proba >= 0.5)
                
                # Risk level classification
                if proba < 0.33:
                    risk_level = "Low"
                elif proba < 0.67:
                    risk_level = "Moderate"
                else:
                    risk_level = "High"
                
                results.append({
                    "patientId": note.get("patientId"),
                    "noteId": note.get("noteId"),
                    "flare_probability": round(proba, 3),
                    "flare_label": label,
                    "risk_level": risk_level
                })
            
            return {"predictions": results}
            
        elif isinstance(input_data, dict):
            # Single prediction
            X = preprocess_single(input_data, model_handler.tfidf,
                                model_handler.svd, model_handler.scaler)
            proba = float(model_handler.model.predict_proba(X)[:, 1][0])
            label = int(proba >= 0.5)
            
            # Risk level classification
            if proba < 0.33:
                risk_level = "Low"
            elif proba < 0.67:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            return {
                "patientId": input_data.get("patientId"),
                "flare_probability": round(proba, 3),
                "flare_label": label,
                "risk_level": risk_level
            }
        
        elif isinstance(input_data, list):
            # List of notes
            return predict_fn({"notes": input_data}, model_handler)
        
        else:
            raise ValueError(f"Unsupported input format: {type(input_data)}")
            
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def output_fn(prediction, accept='application/json'):
    """
    Serialize the prediction output (SageMaker required function).
    
    Args:
        prediction: Model predictions
        accept: Requested output content type
        
    Returns:
        Serialized predictions
    """
    if accept == 'application/json':
        return json.dumps(prediction), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Health check endpoint (for SageMaker)
def ping():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    try:
        if not model_handler.loaded:
            model_handler.load_model()
        
        return {
            "status": "healthy",
            "model_loaded": model_handler.loaded
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
