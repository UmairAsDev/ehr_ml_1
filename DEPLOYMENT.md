# AWS SageMaker Deployment Guide

**Psoriasis Flare Prediction System**

This guide provides step-by-step instructions for deploying the psoriasis flare prediction ML model to AWS SageMaker.

---

## 📋 Prerequisites

### Required Tools
- **AWS CLI** - [Install AWS CLI](https://aws.amazon.com/cli/)
- **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- **Python 3.10+**
- **AWS Account** with appropriate permissions

### AWS Configuration

1. **AWS Credentials**: Configure AWS credentials
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and default region
   ```

2. **Required IAM Permissions**:
   - `AmazonEC2ContainerRegistryFullAccess` (for ECR)
   - `AmazonSageMakerFullAccess` (for SageMaker)
   - `AmazonS3FullAccess` (for model storage)

3. **Create SageMaker Execution Role**:
   ```bash
   # The role ARN will be needed for deployment
   # It should have policies: AmazonSageMakerFullAccess, AmazonS3FullAccess
   ```

### Environment Variables

Set the following environment variables:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=your-account-id
export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::your-account-id:role/SageMakerExecutionRole
export SAGEMAKER_S3_BUCKET=psoriasis-ml-sagemaker
```

> **Tip**: Add these to your `~/.bashrc` or `~/.zshrc` for persistence

---

## 🚀 Deployment Steps

### Step 1: Package the Model

Extract model artifacts from MLflow and package them for SageMaker:

```bash
cd /home/umair/projects/ehr_ml_1/sagemaker

# Package model from MLflow
python model_package.py \
  --source mlflow \
  --model-name flare_detector_v1 \
  --output-dir ./model_package \
  --verify
```

**Output**: `./model_package/model.tar.gz`

**Verification**: The script will verify that all required files are present:
- `lgbm_model.pkl` - LightGBM model
- `tfidf.joblib` - TF-IDF vectorizer
- `svd.joblib` - SVD transformer
- `scaler.joblib` - Standard scaler

---

### Step 2: Build and Push Docker Image

Build the Docker container and push it to Amazon ECR:

```bash
# Make the script executable (if not already)
chmod +x build_and_push.sh

# Build and push to ECR
./build_and_push.sh
```

This script will:
1. ✓ Login to Amazon ECR
2. ✓ Create ECR repository (if doesn't exist)
3. ✓ Build Docker image
4. ✓ Tag image for ECR
5. ✓ Push image to ECR

**Output**: ECR image URI (save this for the next step)

---

### Step 3: Deploy to SageMaker

Deploy the model to a SageMaker endpoint:

```bash
# Deploy using the ECR image
python deploy_model.py \
  --model-path ./model_package/model.tar.gz \
  --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/psoriasis-flare-model:latest
```

This will:
1. Upload `model.tar.gz` to S3
2. Create SageMaker model
3. Create endpoint configuration
4. Create or update endpoint (5-10 minutes)

**Wait Time**: Endpoint creation takes approximately 5-10 minutes

---

### Step 4: Test the Endpoint

Once the endpoint is `InService`, test it:

```bash
# Check endpoint status
python deploy_model.py --status

# Run all tests
python test_endpoint.py --test all

# Test single prediction
python test_endpoint.py --test single

# Performance test
python test_endpoint.py --test performance --num-requests 20
```

---

## 📊 Monitoring and Maintenance

### CloudWatch Logs

View SageMaker endpoint logs:

```bash
aws logs tail /aws/sagemaker/Endpoints/psoriasis-flare-endpoint --follow
```

### Endpoint Status

Check endpoint status programmatically:

```python
import boto3

client = boto3.client('sagemaker', region_name='us-east-1')
response = client.describe_endpoint(EndpointName='psoriasis-flare-endpoint')
print(f"Status: {response['EndpointStatus']}")
```

### Update Model

To deploy a new version:

1. Package the new model:
   ```bash
   python model_package.py --model-name flare_detector_v2
   ```

2. Deploy (will automatically update existing endpoint):
   ```bash
   python deploy_model.py --model-path ./model_package/model.tar.gz
   ```

---

## 🔧 Configuration Options

### Instance Types

Modify instance type in `sagemaker/config.py`:

```python
# Cost-effective for low traffic
instance_type = "ml.t2.medium"  # ~$0.065/hour

# Recommended for production
instance_type = "ml.m5.large"   # ~$0.138/hour

# High performance
instance_type = "ml.c5.xlarge"  # ~$0.238/hour
```

### Auto-scaling

Enable auto-scaling for production:

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace sagemaker \
  --resource-id endpoint/psoriasis-flare-endpoint/variant/primary \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --min-capacity 1 \
  --max-capacity 3
```

---

## 🧪 Testing Locally

Test the Docker container locally before deploying:

```bash
# Build the image
cd sagemaker
docker build -t psoriasis-flare-model:latest -f Dockerfile .

# Run locally
docker run -p 8080:8080 \
  -v $(pwd)/../model_package/model:/opt/ml/model \
  psoriasis-flare-model:latest

# Test with curl
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "patientId": "TEST001",
    "complaints": "Increased itching and redness",
    "assesment": "Psoriasis flare-up"
  }'
```

---

## 💰 Cost Estimation

### Running Costs

| Component | Instance Type | Cost/Hour | Monthly (24/7) |
|-----------|--------------|-----------|----------------|
| Endpoint | ml.t2.medium | $0.065 | ~$47 |
| Endpoint | ml.m5.large | $0.138 | ~$100 |
| S3 Storage | - | $0.023/GB | ~$1 |
| ECR Storage | - | $0.10/GB | ~$1 |

**Estimated Monthly Cost**: $50-$102 (depending on instance type)

### Cost Optimization

1. **Use smaller instances** for low traffic
2. **Enable auto-scaling** to scale down during low usage
3. **Delete endpoint** when not needed:
   ```bash
   aws sagemaker delete-endpoint --endpoint-name psoriasis-flare-endpoint
   ```

---

## 🔒 Security Best Practices

1. **VPC Configuration**: Deploy endpoint in a VPC for network isolation
2. **Encryption**: Enable encryption at rest for model artifacts in S3
3. **IAM Roles**: Use least-privilege IAM roles
4. **API Authentication**: Add API Gateway with authentication in front of SageMaker

---

## ❌ Troubleshooting

### Common Issues

**Issue**: Docker build fails with "No space left on device"
- **Solution**: Clean up Docker: `docker system prune -a`

**Issue**: ECR push fails with authentication error
- **Solution**: Re-login to ECR: `aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com`

**Issue**: Endpoint stays in "Creating" state for too long
- **Solution**: Check CloudWatch logs for errors

**Issue**: Predictions return errors
- **Solution**: Verify model.tar.gz contains all required files
- **Solution**: Check CloudWatch logs for detailed error messages

### Getting Help

- Check CloudWatch logs: `/aws/sagemaker/Endpoints/psoriasis-flare-endpoint`
- View SageMaker console for endpoint status
- Review model package with `python model_package.py --verify`

---

## 📚 Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Docker Documentation](https://docs.docker.com/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

## 🔄 Quick Reference Commands

```bash
# Package model
python sagemaker/model_package.py --source mlflow

# Build and push Docker
cd sagemaker && ./build_and_push.sh

# Deploy to SageMaker
python sagemaker/deploy_model.py --model-path model_package/model.tar.gz

# Check status
python sagemaker/deploy_model.py --status

# Test endpoint
python sagemaker/test_endpoint.py --test all

# View logs
aws logs tail /aws/sagemaker/Endpoints/psoriasis-flare-endpoint --follow

# Delete endpoint (to save costs)
aws sagemaker delete-endpoint --endpoint-name psoriasis-flare-endpoint
```

---

## ✅ Success Checklist

- [ ] AWS credentials configured
- [ ] Environment variables set
- [ ] Model packaged successfully (model.tar.gz exists)
- [ ] Docker image built and pushed to ECR
- [ ] SageMaker endpoint created (status: InService)
- [ ] Test predictions return valid results
- [ ] CloudWatch logs show no errors
- [ ] Performance meets requirements (< 2s latency)

---

**Need help?** Check the troubleshooting section or review CloudWatch logs for detailed error messages.
