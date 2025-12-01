# Psoriasis Flare Prediction System

Machine Learning system for predicting psoriasis flares from EHR data using LightGBM.

## 🎯 Project Overview

This project classifies and predicts psoriasis flares from the Legend EHR database. The system analyzes patient clinical notes, medications, and examination findings to predict which patients are at risk of experiencing a psoriasis flare.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 74% |
| ROC-AUC | 0.82 |
| Precision (Flare) | 0.77 |
| Recall (Flare) | 0.78 |

```
              precision    recall  f1-score   support

           0       0.69      0.69      0.69       917
           1       0.77      0.78      0.78      1250

    accuracy                           0.74      2167
   macro avg       0.73      0.73      0.73      2167
weighted avg       0.74      0.74      0.74      2167
```

## 🚀 Quick Start - AWS SageMaker Deployment

Deploy to production in 20 minutes! See [QUICKSTART.md](QUICKSTART.md)

```bash
# 1. Package model
cd sagemaker
python model_package.py --source mlflow

# 2. Build and push Docker
./build_and_push.sh

# 3. Deploy to SageMaker
python deploy_model.py --model-path model_package/model.tar.gz

# 4. Test
python test_endpoint.py --test all
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 20 minutes
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment guide
- **[API.md](API.md)** - API reference and examples

## 🏗️ Architecture

```
EHR Database (MySQL)
        ↓
Feature Engineering Pipeline
        ↓
TF-IDF + SVD (Text) + StandardScaler (Numeric)
        ↓
LightGBM Classifier
        ↓
Risk Prediction + SHAP Explanations
```

## 💻 Local Development

### Installation

```bash
# Using uv (recommended)
uv venv              # Create environment
uv sync              # Install from pyproject.toml
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Locally

```bash
# Start FastAPI server
python main.py

# API will be available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Training

```bash
# Run training pipeline
python pipeline/mlflow_pipeline.py

# View MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

## 🔑 Key Features

- **Automated Feature Engineering**: Extracts 113 features from clinical notes
- **SHAP Explanations**: Interpretable predictions with feature importance
- **Batch Processing**: Support for multiple patient predictions
- **Production Ready**: Health checks, logging, error handling
- **Cloud Deployment**: One-command deployment to AWS SageMaker

## 🧪 Testing

```bash
# Test local API
curl http://localhost:8000/health

# Test SageMaker endpoint
python sagemaker/test_endpoint.py --test all
```

## 📦 Project Structure

```
├── app/                    # FastAPI application
│   ├── inference.py       # Inference logic
│   ├── model_service.py   # Model service with SHAP
│   └── schemas.py         # Pydantic validation schemas
├── pipeline/              # ML pipeline
│   ├── preprocessing.py   # Feature engineering
│   ├── training.py        # Model training
│   └── mlflow_pipeline.py # MLflow integration
├── sagemaker/             # AWS SageMaker deployment
│   ├── inference.py       # SageMaker inference handler
│   ├── Dockerfile         # Container configuration
│   ├── model_package.py   # Model packaging utility
│   ├── deploy_model.py    # Deployment script
│   └── test_endpoint.py   # Testing utilities
├── db/                    # Database connection
├── utils/                 # Helper utilities
└── main.py               # Application entry point
```

## ⚙️ Configuration

Set environment variables:

```bash
# Database (for training/local inference)
export DB_LOCAL_HOST=localhost
export DB_LOCAL_NAME=legend_ehr
export DB_LOCAL_USERNAME=your_user
export DB_LOCAL_PASSWORD=your_password

# AWS SageMaker (for cloud deployment)
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::123456789012:role/SageMakerRole
export SAGEMAKER_S3_BUCKET=psoriasis-ml-sagemaker
```

## 🔒 Security

- Models use only safe clinical features (no leaked information)
- Text masking prevents post-flare term leakage
- Input validation with Pydantic schemas
- VPC deployment support for SageMaker

## 💰 Cost

AWS SageMaker deployment costs approximately $50-100/month depending on instance type. See [DEPLOYMENT.md](DEPLOYMENT.md#cost-estimation) for details.

## 📝 License

MIT License

## 👤 Author

**Umair Ashraf** - umairashrafbsse@gmail.com

---

**Ready to deploy?** Start with [QUICKSTART.md](QUICKSTART.md)!




