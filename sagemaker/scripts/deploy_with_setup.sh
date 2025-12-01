#!/bin/bash

# Setup script for SageMaker deployment
# This script sets the correct environment variables and runs deployment

echo "============================================"
echo "SageMaker Deployment Setup"
echo "============================================"
echo ""

# Set AWS Region
export AWS_REGION=us-west-2
echo "✓ AWS Region: $AWS_REGION"

# Get AWS Account ID
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "257394496419")
echo "✓ AWS Account ID: $AWS_ACCOUNT_ID"

# Set S3 Bucket (using existing SageMaker bucket)
export SAGEMAKER_S3_BUCKET=sagemaker-us-west-2-257394496419
echo "✓ S3 Bucket: $SAGEMAKER_S3_BUCKET"

# Set SageMaker Execution Role
export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::${AWS_ACCOUNT_ID}:role/SageMakerExecutionRole
echo "✓ SageMaker Role: $SAGEMAKER_EXECUTION_ROLE"

echo ""
echo "============================================"
echo "Environment configured successfully!"
echo "============================================"
echo ""

# Check if model package exists
if [ ! -f "model_package/model.tar.gz" ]; then
    echo "❌ Error: model_package/model.tar.gz not found"
    echo "Please run: python model_package.py first"
    exit 1
fi

echo "Starting deployment..."
echo ""

# Run deployment
python deploy_model.py --model-path model_package/model.tar.gz

echo ""
echo "============================================"
echo "Deployment script completed"
echo "============================================"
