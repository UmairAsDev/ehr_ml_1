#!/bin/bash

# Quick fix script for SageMaker deployment
# This script runs the deployment with the correct environment variables

echo "🚀 Running SageMaker deployment with correct configuration..."
echo ""

# Export the correct environment variables
export AWS_REGION=us-west-2
export SAGEMAKER_S3_BUCKET=sagemaker-us-west-2-257394496419
export AWS_ACCOUNT_ID=257394496419
export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::257394496419:role/SageMakerExecutionRole

# Run deployment
python deploy_model.py --model-path ../model_package/model.tar.gz
