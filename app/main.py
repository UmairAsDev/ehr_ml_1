import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import APIRouter
from db.db import get_db
from pipeline.get_patient import fetch_final_data
from app.model_service import ModelService
from pipeline.mlflow_pipeline import run_pipeline
import warnings
warnings.filterwarnings("ignore")

svc = ModelService()    
router  = APIRouter()





def predict_patient(patient_id: str):
    db = next(get_db())
    notes_df = fetch_final_data(db, patient_id)

    print(f"Fetched {len(notes_df)} notes for patient {patient_id}")

    if notes_df.empty:
        return {"patientId": patient_id, "error": "No notes found"}

    notes = notes_df.to_dict(orient="records")
    return svc.predict_patient_notes(notes, patient_id)









@router.post("/predict")
async def predict(patient_id: str):
    res = predict_patient(patient_id)
    return res



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
