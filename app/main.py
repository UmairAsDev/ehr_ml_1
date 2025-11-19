import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import APIRouter
from app.config import PatientNote
from app.model_service import ModelService
from pipeline.mlflow_pipeline import run_pipeline
import subprocess
import warnings
warnings.filterwarnings("ignore")

svc = ModelService()    


router  = APIRouter()


@router.post("/predict")
async def predict(note: PatientNote):
    print(note.model_dump())
    res = svc.predict_note(note.model_dump())
    return {
        "patientId": res["patientId"],
        "noteDate": res["noteDate"],
        "flare_probability": res["flare_probability"],
        "flare_label": res["flare_label"],
        "flare_risk_level": res["flare_risk_level"],
        "explanation_summary": res["explanation_summary"],
        "key_influences": res["key_influences"],
    }



@router.post("/train")
def train_model():
    """
    Trigger the ML training pipeline.
    """
    try:
        run_pipeline()
        return {"status": "success", "output": "Model trained successfully"}
    except Exception as e:
        return {"status": "error", "output": str(e)}
