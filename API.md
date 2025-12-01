# API Documentation

**Psoriasis Flare Prediction System API**

Version: 1.0

---

## Base URL

- **Development**: `http://localhost:8000`
- **SageMaker**: `https://<sagemaker-endpoint-url>/invocations`

---

## Health Checks

### GET /health

Health check endpoint for monitoring.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0"
}
```

### GET /ping

Alias for `/health` (SageMaker compatible).

---

## Prediction Endpoints

### POST /predict

Predict psoriasis flare risk for a patient based on their clinical notes.

**Request Body**:
```json
{
  "patient_id": "PAT001"
}
```

**Response**:
```json
{
  "patientId": "PAT001",
  "total_notes": 5,
  "final_flare_label": 1,
  "final_risk_level": "High",
  "risky_notes": [
    {
      "noteId": "NOTE001",
      "noteDate": "2024-01-15",
      "flare_probability": 0.85,
      "flare_risk_level": "High",
      "key_influences": [
        {
          "feature": "itch_present",
          "impact": 0.234,
          "direction": "↑ flare risk up"
        }
      ]
    }
  ]
}
```

**Risk Levels**:
- **Low**: probability < 0.33
- **Moderate**: 0.33 ≤ probability < 0.50
- **Elevated**: 0.50 ≤ probability < 0.67
- **High**: probability ≥ 0.67

---

### POST /predict_batch

Batch prediction for multiple patient notes.

**Request Body**:
```json
{
  "notes": [
    {
      "patientId": "PAT001",
      "noteId": "NOTE001",
      "noteDate": "2024-01-15",
      "patientSummary": "45 year old Male patient",
      "complaints": "Increased itching and redness on elbows",
      "assesment": "Psoriasis flare-up",
      "examination": "Erythematous plaques with silvery scale",
      "reviewofsystem": "Reports persistent itching",
      "currentmedication": "Triamcinolone cream 0.1%",
      "pastHistory": "Non-smoker",
      "diagnoses": "L40.0 Psoriasis vulgaris"
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "patientId": "PAT001",
      "noteId": "NOTE001",
      "noteDate": "2024-01-15",
      "flare_probability": 0.75,
      "flare_label": 1,
      "risk_level": "High",
      "key_influences": [
        {
          "feature": "itch_present",
          "impact": 0.234,
          "direction": "↑ flare risk up"
        }
      ],
      "explanation_summary": "Model predicted High risk..."
    }
  ]
}
```

**Constraints**:
- Minimum: 1 note
- Maximum: 100 notes per batch

---

## Training Endpoint

### POST /train

Trigger the ML training pipeline.

**Request Body**: None

**Response**:
```json
{
  "status": "success",
  "output": "Model trained successfully"
}
```

---

## Input Schema

### PatientNote Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `patientId` | string | Yes | Patient identifier |
| `noteId` | string | No | Note identifier |
| `noteDate` | string | No | Date (YYYY-MM-DD format) |
| `patientSummary` | string | No | Patient demographics and summary |
| `complaints` | string | No | Chief complaints |
| `assesment` | string | No | Clinical assessment |
| `examination` | string | No | Physical examination findings |
| `reviewofsystem` | string | No | Review of systems |
| `currentmedication` | string | No | Current medications |
| `pastHistory` | string | No | Past medical history |
| `diagnoses` | string | No | Diagnoses (ICD codes) |
| `procedure` | string | No | Procedures performed |
| `allergy` | string | No | Allergies |

---

## Feature Importance

The model considers these key features:

### Numeric Features
- `patient_age` - Patient age in years
- `has_psoriasis` - Psoriasis diagnosis present (0/1)
- `on_steroid_med` - Currently on steroid medication (0/1)
- `on_biologic` - Currently on biologic therapy (0/1)
- `itch_present` - Itching reported (0/1)
- `dry_skin` - Dry skin reported (0/1)
- `plaques_present` - Plaques observed (0/1)
- `silvery_scale` - Silvery scale observed (0/1)
- `elbows_involved` - Elbows affected (0/1)
- `hyperpigmentation` - Hyperpigmentation present (0/1)
- `smoker` - Smoking history (0/1)
- `alcohol_use` - Alcohol use (0/1)
- `family_melanoma` - Family history of melanoma (0/1)

### Text Features
- Extracted from clinical notes using TF-IDF and dimensionality reduction
- Includes complaint keywords, assessment terms, and medication patterns

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid input",
  "detail": "Patient ID is required",
  "status_code": 400
}
```

### 404 Not Found
```json
{
  "patientId": "PAT001",
  "error": "No notes found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Prediction failed",
  "detail": "Model initialization error",
  "status_code": 500
}
```

### 503 Service Unavailable
```json
{
  "error": "Service unavailable",
  "detail": "Model service not initialized",
  "status_code": 503
}
```

---

## Usage Examples

### Python (requests)

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"patient_id": "PAT001"}
)
result = response.json()
print(f"Risk Level: {result['final_risk_level']}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict_batch",
    json={
        "notes": [
            {
                "patientId": "PAT001",
                "complaints": "Itching and redness",
                "assesment": "Psoriasis flare"
            }
        ]
    }
)
results = response.json()
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "PAT001"}'

# Batch prediction
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "notes": [{
      "patientId": "PAT001",
      "complaints": "Itching",
      "assesment": "Flare"
    }]
  }'
```

### SageMaker Runtime (boto3)

```python
import boto3
import json

client = boto3.client('sagemaker-runtime', region_name='us-east-1')

payload = {
    "patientId": "PAT001",
    "complaints": "Increased itching and redness",
    "assesment": "Psoriasis flare-up"
}

response = client.invoke_endpoint(
    EndpointName='psoriasis-flare-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read())
print(result)
```

---

## Rate Limits

- **Development**: No limits
- **Production**: Configure via API Gateway or SageMaker auto-scaling

---

## Model Information

- **Algorithm**: LightGBM Gradient Boosting
- **Features**: 113 total (13 numeric + 100 text SVD components)
- **Performance**:
  - Accuracy: 74%
  - ROC-AUC: 0.82
  - Precision: 0.77 (class 1)
  - Recall: 0.78 (class 1)

---

## Interactive Documentation

When running locally, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Support

For issues or questions:
1. Check CloudWatch logs for SageMaker deployments
2. Review the `DEPLOYMENT.md` guide
3. Verify input data matches the schema

---

## Changelog

### Version 1.0 (2024-11-27)
- Initial release
- Single and batch prediction endpoints
- Health check endpoints
- Training endpoint
- Comprehensive input validation
