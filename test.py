import asyncio
from db.db import get_db
from pipeline.get_patient import fetch_final_data
from contextlib import closing
from app.model_service import ModelService

model_service = ModelService() 

def predict_patient(patient_id: str):
    db = next(get_db())
    notes_df = fetch_final_data(db, patient_id)

    print(f"Fetched {len(notes_df)} notes for patient {patient_id}")

    if notes_df.empty:
        return {"patientId": patient_id, "error": "No notes found"}

    notes = notes_df.to_dict(orient="records")
    return model_service.predict_patient_notes(notes, patient_id)





if __name__ == '__main__':
    from pprint import pprint
    import asyncio

    test_patient_id = "185562"
    result = predict_patient(test_patient_id)

    pprint(result)