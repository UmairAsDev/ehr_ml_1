import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from db.db import get_db
from pipeline.get_patient import fetch_final_data
from app.model_service import ModelService
from app.schemas import (
    PredictRequest,
    PatientPredictionResponse,
    TrainResponse,
    HealthResponse,
    ErrorResponse,
    BatchPredictRequest,
)
from pipeline.mlflow_pipeline import run_pipeline
import warnings
import logging
import traceback

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize model service
try:
    svc = ModelService()
    logger.info("✓ Model service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model service: {str(e)}")
    svc = None

router = APIRouter()


# Exception handlers
@router.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500,
        },
    )


def predict_patient(patient_id: str):
    """
    Predict psoriasis flare risk for a patient.

    Args:
        patient_id: Patient identifier

    Returns:
        Prediction results with risk assessment
    """
    try:
        logger.info(f"Fetching data for patient: {patient_id}")

        db = next(get_db())
        notes_df = fetch_final_data(db, patient_id)

        logger.info(f"Fetched {len(notes_df)} notes for patient {patient_id}")

        if notes_df.empty:
            logger.warning(f"No notes found for patient {patient_id}")
            return {"patientId": patient_id, "error": "No notes found"}

        notes = notes_df.to_dict(orient="records")

        if svc is None:
            raise HTTPException(status_code=503, detail="Model service not initialized")

        result = svc.predict_patient_notes(notes, patient_id)
        logger.info(f"Prediction completed for patient {patient_id}")

        return result

    except Exception as e:
        logger.error(f"Error predicting for patient {patient_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        Health status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=svc is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0",
    )


@router.get("/ping", response_model=HealthResponse)
async def ping():
    """
    SageMaker health check endpoint (alias for /health).

    Returns:
        Health status
    """
    return await health_check()


# Prediction endpoints
@router.post("/predict", response_model=PatientPredictionResponse)
async def predict(request: PredictRequest):
    """
    Predict psoriasis flare risk for a patient.

    Args:
        request: Prediction request with patient ID

    Returns:
        Patient-level risk assessment
    """
    try:
        logger.info(f"Prediction request for patient: {request.patient_id}")
        res = predict_patient(request.patient_id)
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_batch")
async def predict_batch(request: BatchPredictRequest):
    """
    Batch prediction for multiple patient notes.

    Args:
        request: Batch prediction request with notes

    Returns:
        List of predictions
    """
    try:
        logger.info(f"Batch prediction request for {len(request.notes)} notes")

        if svc is None:
            raise HTTPException(status_code=503, detail="Model service not initialized")

        results = []
        for note in request.notes:
            try:
                result = svc.predict_note(note.dict())
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict note {note.noteId}: {str(e)}")
                results.append({"noteId": note.noteId, "error": str(e)})

        logger.info(f"Batch prediction completed: {len(results)} results")
        return {"predictions": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainResponse)
def train_model():
    """
    Trigger the ML training pipeline.

    Returns:
        Training status
    """
    try:
        logger.info("Starting model training pipeline")
        run_pipeline()
        logger.info("Model training completed successfully")

        return TrainResponse(status="success", output="Model trained successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())

        return TrainResponse(status="error", output=str(e))
