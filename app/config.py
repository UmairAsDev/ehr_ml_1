from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class PatientNote(BaseModel):
    noteId: int
    provider: int
    physician: int
    referringPhysician: Optional[float] = None
    noteDate: datetime
    patientId: int
    complaints: Optional[str] = None
    pastHistory: Optional[str] = None
    assesment: Optional[str] = None
    reviewofsystem: Optional[str] = None
    currentmedication: Optional[str] = None
    procedure: Optional[str] = None
    biopsyNotes: Optional[str] = None
    mohsNotes: Optional[str] = None
    allergy: Optional[str] = None
    examination: Optional[str] = None
    patientSummary: Optional[str] = None
    diagnoses: Optional[str] = None
    PlaceOfService: Optional[str] = None
    Rendering_Provider: Optional[str] = None
    Physician: Optional[str] = None
    Referring_Provider: Optional[str] = None
    Billing_Provider: Optional[str] = None

    class Config:
        from_attributes = True
        populate_by_name = True


