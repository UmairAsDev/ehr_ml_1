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



if __name__ == '__main__':
    from pprint import pprint
    test_patient_id = "654057"  
    result = asyncio.run(predict_patient(test_patient_id))
    
    pprint(result)