# Quick Start Guide - AWS SageMaker Deployment

Deploy your psoriasis flare prediction model to AWS SageMaker in 4 steps.

## Prerequisites

```bash
# Install AWS CLI and configure credentials
aws configure

# Set environment variables
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/SageMakerExecutionRole
export SAGEMAKER_S3_BUCKET=sagemaker-us-west-2-257394496419
```

## Step 1: Package Model (2 minutes)

```bash
cd sagemaker
python model_package.py --source mlflow --model-name flare_detector_v1
```

✅ Output: `model_package/model.tar.gz`

## Step 2: Build Docker Image (5 minutes)

```bash
chmod +x build_and_push.sh
./build_and_push.sh
```

✅ Output: Image pushed to ECR

## Step 3: Deploy to SageMaker (10 minutes)

```bash
python deploy_model.py --model-path model_package/model.tar.gz
```

✅ Output: Endpoint `InService`

## Step 4: Test (1 minute)

```bash
python test_endpoint.py --test all
```

✅ If all tests pass, you're ready to use the endpoint!

## Use the Endpoint

```python
import boto3
import json

client = boto3.client('sagemaker-runtime', region_name='us-west-2')

response = client.invoke_endpoint(
    EndpointName='psoriasis-flare-endpoint',
    ContentType='application/json',
    Body=json.dumps({
        "patientId": "PAT001",
        "complaints": "Itching and redness",
        "assesment": "Psoriasis flare"
    })
)

result = json.loads(response['Body'].read())
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['flare_probability']}")
```

## Cost

- **Monthly**: ~$50-100 (ml.t2.medium to ml.m5.large)
- **Delete endpoint when not in use**: `aws sagemaker delete-endpoint --endpoint-name psoriasis-flare-endpoint`

## Troubleshooting

- **Issue**: Commands fail → Check AWS credentials: `aws sts get-caller-identity`
- **Issue**: Docker build fails → Clean Docker: `docker system prune -a`
- **Issue**: Endpoint errors → Check logs: `aws logs tail /aws/sagemaker/Endpoints/psoriasis-flare-endpoint --follow`

## Full Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [API.md](API.md) - API documentation

---

**Total Time**: ~20 minutes to full production deployment! 🚀
