from fastapi import APIRouter
from pipeline.preprocessing import FeatureExtraction




app = APIRouter(
    prefix="/api",
    tags=["psoraisis_flare"],
)


@app.post("/predict")
def predict(request: dict):
    """
    Predict psoraisis flare.

    Parameters:
    - request: JSON payload with patient features.

    Returns:
    - JSON response with prediction and probability.
    """
    # Load model from MLflow
    model_uri = "models:/flare_detector_v1/1"
    model = mlflow.lightgbm.load_model(model_uri)#type:ignore

    # Preprocess input data
    preprocessor = FeatureExtraction()
    features = preprocessor.extract_features()

    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1] #type:ignore

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }