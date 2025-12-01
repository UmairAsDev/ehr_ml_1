# SageMaker Deployment

This directory contains all files needed for deploying the psoriasis flare prediction model to AWS SageMaker.

## Directory Structure

```
sagemaker/
├── docker/                  # Docker container configuration
│   ├── Dockerfile          # Container definition
│   ├── dockerd-entrypoint.py  # Container entrypoint script
│   ├── inference.py        # SageMaker inference handler
│   └── requirements.txt    # Python dependencies for container
├── scripts/                # Deployment and build scripts
│   ├── build_and_push.sh  # Build and push Docker image to ECR
│   ├── deploy_model.py    # Main deployment script
│   ├── quick_deploy.sh    # Quick deployment wrapper
│   ├── deploy_with_setup.sh  # Setup + deployment wrapper
│   └── check_aws_setup.py # AWS configuration checker
├── utils/                  # Utility modules
│   ├── config.py          # Deployment configuration
│   └── model_package.py   # Model packaging utility
├── tests/                  # Testing scripts
│   └── test_endpoint.py   # Endpoint testing script
└── model_package/          # Packaged model artifacts
    └── model.tar.gz       # Model artifact (includes inference.py)
```

## Quick Start

### 1. Build and Push Docker Image
```bash
cd scripts
export AWS_ACCOUNT_ID=your-account-id
export AWS_REGION=us-west-2
./build_and_push.sh
```

### 2. Deploy to SageMaker
```bash
cd scripts
./quick_deploy.sh
```

### 3. Test Endpoint
```bash
cd tests
python test_endpoint.py
```

## Environment Variables

Required:
- AWS_ACCOUNT_ID: Your AWS account ID
- SAGEMAKER_EXECUTION_ROLE: ARN of SageMaker execution role
- AWS_REGION: AWS region (default: us-west-2)
- SAGEMAKER_S3_BUCKET: S3 bucket for model artifacts
