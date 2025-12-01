"""
Configuration for AWS SageMaker deployment.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SageMakerConfig:
    """Configuration for SageMaker deployment."""

    # AWS Configuration
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_account_id: Optional[str] = os.getenv("AWS_ACCOUNT_ID")

    # S3 Configuration
    s3_bucket: str = os.getenv("SAGEMAKER_S3_BUCKET", "psoriasis-ml-sagemaker")
    s3_model_prefix: str = "models/psoriasis-flare"
    s3_data_prefix: str = "data"

    # Model Configuration
    model_name: str = "psoriasis-flare-predictor"
    model_version: str = "v1"

    # ECR Configuration
    ecr_repository: str = "psoriasis-flare-model"
    image_tag: str = "latest"

    # SageMaker Endpoint Configuration
    endpoint_name: str = "psoriasis-flare-endpoint"
    endpoint_config_name: str = "psoriasis-flare-config"
    instance_type: str = "ml.m5.large"  # Can also use ml.t2.medium for cost savings
    initial_instance_count: int = 1

    # Model execution role
    sagemaker_role: Optional[str] = os.getenv("SAGEMAKER_EXECUTION_ROLE")

    # Model package location
    model_package_path: str = "./model_package/model.tar.gz"

    @property
    def ecr_image_uri(self) -> str:
        """Get the ECR image URI."""
        if not self.aws_account_id:
            raise ValueError("AWS_ACCOUNT_ID environment variable must be set")
        return f"{self.aws_account_id}.dkr.ecr.{self.aws_region}.amazonaws.com/{self.ecr_repository}:{self.image_tag}"

    @property
    def s3_model_uri(self) -> str:
        """Get the S3 URI for model artifacts."""
        return f"s3://{self.s3_bucket}/{self.s3_model_prefix}/model.tar.gz"

    def validate(self):
        """Validate configuration."""
        errors = []

        if not self.aws_account_id:
            errors.append("AWS_ACCOUNT_ID environment variable is required")

        if not self.sagemaker_role:
            errors.append("SAGEMAKER_EXECUTION_ROLE environment variable is required")

        if errors:
            raise ValueError(
                f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return True


# Default configuration instance
config = SageMakerConfig()


def print_config():
    """Print current configuration."""
    print("\n" + "=" * 60)
    print("SageMaker Deployment Configuration")
    print("=" * 60)
    print(f"AWS Region: {config.aws_region}")
    print(f"AWS Account ID: {config.aws_account_id}")
    print(f"S3 Bucket: {config.s3_bucket}")
    print(f"Model S3 URI: {config.s3_model_uri}")
    print(f"ECR Repository: {config.ecr_repository}")
    try:
        print(f"ECR Image URI: {config.ecr_image_uri}")
    except ValueError:
        print("ECR Image URI: Not configured (AWS_ACCOUNT_ID missing)")
    print(f"Endpoint Name: {config.endpoint_name}")
    print(f"Instance Type: {config.instance_type}")
    print(f"SageMaker Role: {config.sagemaker_role or 'Not configured'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_config()

    try:
        config.validate()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration validation failed:\n{e}")
