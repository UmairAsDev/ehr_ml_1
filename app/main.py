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


import asyncio
from db.db import get_db
from pipeline.get_patient import get_patient_data  
from app.model_service import ModelService

model_service = ModelService() 

async def predict_patient(patient_id: str):
    db = next(get_db())
    notes_df = await get_patient_data(db, [patient_id])
    
    if notes_df.empty:
        return {"patientId": patient_id, "error": "No notes found"}
    
    notes = notes_df.to_dict(orient="records")
    return model_service.predict_patient_notes(notes, patient_id)










@router.post("/predict")
async def predict(note: PatientNote):
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
