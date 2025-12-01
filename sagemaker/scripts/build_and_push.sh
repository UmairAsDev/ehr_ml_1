#!/bin/bash

# Build and push Docker image to AWS ECR
# This script builds the Docker image and pushes it to Amazon ECR

set -e

# Configuration (can be overridden by environment variables)
AWS_REGION=${AWS_REGION:-us-west-2}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-""}
ECR_REPOSITORY=${ECR_REPOSITORY:-psoriasis-flare-model}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AWS SageMaker Docker Build and Push${NC}"
echo -e "${GREEN}========================================${NC}"

# Validate AWS account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}Error: AWS_ACCOUNT_ID environment variable is not set${NC}"
    echo "Please set it using: export AWS_ACCOUNT_ID=your-account-id"
    exit 1
fi

# Set image URI
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  AWS Region: ${AWS_REGION}"
echo "  AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "  ECR Repository: ${ECR_REPOSITORY}"
echo "  Image Tag: ${IMAGE_TAG}"
echo "  Image URI: ${IMAGE_URI}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    echo "Please install it from: https://aws.amazon.com/cli/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Login to ECR
echo -e "\n${YELLOW}Step 1/5: Logging in to Amazon ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo -e "${GREEN}✓ Logged in to ECR${NC}"

# Create ECR repository if it doesn't exist
echo -e "\n${YELLOW}Step 2/5: Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} &> /dev/null; then
    echo "Repository does not exist. Creating..."
    aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION}
    echo -e "${GREEN}✓ Repository created${NC}"
else
    echo -e "${GREEN}✓ Repository exists${NC}"
fi

# Build Docker image
echo -e "\n${YELLOW}Step 3/5: Building Docker image...${NC}"
cd "$(dirname "$0")/../docker"

# Build for linux/amd64 platform and disable buildx features that create OCI manifests
# SageMaker requires Docker v2 manifest format, not OCI image index
DOCKER_BUILDKIT=0 docker build --platform linux/amd64 -t ${ECR_REPOSITORY}:${IMAGE_TAG} -f Dockerfile .
echo -e "${GREEN}✓ Image built successfully${NC}"

# Tag image for ECR
echo -e "\n${YELLOW}Step 4/5: Tagging image...${NC}"
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${IMAGE_URI}
echo -e "${GREEN}✓ Image tagged${NC}"

# Push to ECR
echo -e "\n${YELLOW}Step 5/5: Pushing image to ECR...${NC}"
# Disable provenance and SBOM to avoid OCI manifest format
DOCKER_BUILDKIT=0 docker push ${IMAGE_URI}
echo -e "${GREEN}✓ Image pushed successfully${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build and Push Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Image URI: ${IMAGE_URI}"
echo -e "\nYou can now use this image for SageMaker deployment:"
echo -e "  python deploy_model.py --image-uri ${IMAGE_URI}"
echo -e "${GREEN}========================================${NC}\n"
