"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime


class PatientNote(BaseModel):
    """Schema for patient note data."""

    patientId: str = Field(..., description="Patient identifier", min_length=1)
    noteId: Optional[str] = Field(None, description="Note identifier")
    noteDate: Optional[str] = Field(None, description="Date of the note (YYYY-MM-DD)")

    # Patient summary
    patientSummary: Optional[str] = Field("", description="Patient summary information")

    # Clinical text fields
    complaints: Optional[str] = Field("", description="Chief complaints")
    assesment: Optional[str] = Field("", description="Clinical assessment")
    examination: Optional[str] = Field("", description="Physical examination findings")
    reviewofsystem: Optional[str] = Field("", description="Review of systems")
    currentmedication: Optional[str] = Field("", description="Current medications")
    pastHistory: Optional[str] = Field("", description="Past medical history")
    diagnoses: Optional[str] = Field("", description="Diagnoses")
    procedure: Optional[str] = Field("", description="Procedures")
    allergy: Optional[str] = Field("", description="Allergies")

    class Config:
        schema_extra = {
            "example": {
                "patientId": "PAT001",
                "noteId": "NOTE12345",
                "noteDate": "2024-01-15",
                "patientSummary": "45 year old Male patient",
                "complaints": "Increased itching and redness on elbows",
                "assesment": "Psoriasis flare-up",
                "examination": "Erythematous plaques with silvery scale on bilateral elbows",
                "reviewofsystem": "Reports persistent itching and dry skin",
                "currentmedication": "Triamcinolone cream 0.1%",
                "pastHistory": "Non-smoker. No alcohol use.",
                "diagnoses": "L40.0 Psoriasis vulgaris",
            }
        }


class PredictRequest(BaseModel):
    """Schema for single prediction request."""

    patient_id: str = Field(..., description="Patient identifier", alias="patientId")

    class Config:
        schema_extra = {"example": {"patient_id": "PAT001"}}


class BatchPredictRequest(BaseModel):
    """Schema for batch prediction request."""

    notes: List[PatientNote] = Field(..., description="List of patient notes")

    @field_validator("notes")
    def validate_notes(cls, v):
        if len(v) < 1:
            raise ValueError("At least one note is required")
        if len(v) > 100:
            raise ValueError("Maximum 100 notes per batch request")
        return v

    class Config:
        schema_extra = {
            "example": {
                "notes": [
                    {
                        "patientId": "PAT001",
                        "noteId": "NOTE001",
                        "complaints": "Mild itching",
                        "assesment": "Stable psoriasis",
                    }
                ]
            }
        }


class KeyInfluence(BaseModel):
    """Schema for feature influence."""

    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    patientId: str
    noteId: Optional[str] = None
    noteDate: Optional[str] = None
    flare_probability: float = Field(..., ge=0.0, le=1.0)
    flare_label: int = Field(..., ge=0, le=1)
    flare_risk_level: str = Field(..., description="Low, Moderate, or High")
    key_influences: Optional[List[KeyInfluence]] = None
    explanation_summary: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "patientId": "PAT001",
                "noteDate": "2024-01-15",
                "flare_probability": 0.85,
                "flare_label": 1,
                "flare_risk_level": "High",
                "key_influences": [
                    {
                        "feature": "itch_present",
                        "impact": 0.234,
                        "direction": "↑ flare risk up",
                    }
                ],
            }
        }


class PatientPredictionResponse(BaseModel):
    """Schema for patient-level prediction response."""

    patientId: str
    total_notes: int
    final_flare_label: Optional[int] = Field(None, ge=0, le=1)
    final_risk_level: str
    risky_notes: List[Dict]

    class Config:
        schema_extra = {
            "example": {
                "patientId": "PAT001",
                "total_notes": 5,
                "final_flare_label": 1,
                "final_risk_level": "High",
                "risky_notes": [],
            }
        }


class TrainResponse(BaseModel):
    """Schema for training response."""

    status: str
    output: str
    metrics: Optional[Dict[str, float]] = None

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "output": "Model trained successfully",
                "metrics": {"roc_auc": 0.82, "accuracy": 0.74},
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    model_loaded: bool
    timestamp: str
    version: str = "1.0"

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0",
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error response."""

    error: str
    detail: Optional[str] = None
    status_code: int

    class Config:
        schema_extra = {
            "example": {
                "error": "Prediction failed",
                "detail": "Invalid input data",
                "status_code": 400,
            }
        }
