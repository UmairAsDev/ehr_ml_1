"""
Deploy model to AWS SageMaker endpoint.

This script handles the complete deployment process:
1. Upload model.tar.gz to S3
2. Create SageMaker model
3. Create endpoint configuration
4. Create or update endpoint
"""

import boto3
import time
import logging
import argparse
from pathlib import Path
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import SageMakerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """Handle SageMaker model deployment."""

    def __init__(self, config: SageMakerConfig):
        self.config = config
        self.s3_client = boto3.client("s3", region_name=config.aws_region)
        self.sm_client = boto3.client("sagemaker", region_name=config.aws_region)

    def upload_model_to_s3(self, local_model_path: str) -> str:
        """
        Upload model.tar.gz to S3.

        Args:
            local_model_path: Local path to model.tar.gz

        Returns:
            S3 URI of uploaded model
        """
        try:
            logger.info(f"Uploading model to S3: {self.config.s3_model_uri}")

            # Check if bucket exists
            bucket_exists = False
            try:
                self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
                bucket_exists = True
                logger.info(f"✓ S3 bucket exists: {self.config.s3_bucket}")
            except self.s3_client.exceptions.ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                if error_code == "404":
                    # Bucket doesn't exist, try to create it
                    logger.info(
                        f"Bucket does not exist. Attempting to create: {self.config.s3_bucket}"
                    )
                    try:
                        if self.config.aws_region == "us-west-2":
                            self.s3_client.create_bucket(Bucket=self.config.s3_bucket)
                        else:
                            self.s3_client.create_bucket(
                                Bucket=self.config.s3_bucket,
                                CreateBucketConfiguration={
                                    "LocationConstraint": self.config.aws_region
                                },
                            )
                        bucket_exists = True
                        logger.info(f"✓ S3 bucket created: {self.config.s3_bucket}")
                    except self.s3_client.exceptions.ClientError as create_error:
                        if "AccessDenied" in str(create_error):
                            logger.error(f"\n{'='*60}")
                            logger.error("❌ S3 BUCKET ACCESS ERROR")
                            logger.error(f"{'='*60}")
                            logger.error(
                                f"Bucket '{self.config.s3_bucket}' does not exist and you don't have permission to create it."
                            )
                            logger.error("\nPossible solutions:")
                            logger.error(
                                "  1. Use an existing S3 bucket you have access to:"
                            )
                            logger.error(
                                f"     export SAGEMAKER_S3_BUCKET=your-existing-bucket-name"
                            )
                            logger.error(
                                "     python deploy_model.py --model-path model_package/model.tar.gz"
                            )
                            logger.error("\n  2. Ask your AWS administrator to:")
                            logger.error(
                                "     - Create the bucket 'psoriasis-ml-sagemaker' for you, OR"
                            )
                            logger.error(
                                "     - Grant you 's3:CreateBucket' permission"
                            )
                            logger.error(f"{'='*60}\n")
                            raise ValueError(
                                f"S3 bucket '{self.config.s3_bucket}' does not exist and cannot be created due to insufficient permissions"
                            )
                        else:
                            raise
                elif error_code == "403":
                    logger.error(f"\n{'='*60}")
                    logger.error("❌ S3 BUCKET ACCESS DENIED")
                    logger.error(f"{'='*60}")
                    logger.error(
                        f"You don't have permission to access bucket '{self.config.s3_bucket}'"
                    )
                    logger.error(
                        "\nPlease ask your AWS administrator to grant you access to this bucket."
                    )
                    logger.error(f"{'='*60}\n")
                    raise ValueError(
                        f"Access denied to S3 bucket '{self.config.s3_bucket}'"
                    )
                else:
                    raise

            # Upload model
            if bucket_exists:
                s3_key = f"{self.config.s3_model_prefix}/model.tar.gz"
                logger.info(
                    f"Uploading {local_model_path} to s3://{self.config.s3_bucket}/{s3_key}"
                )
                self.s3_client.upload_file(
                    local_model_path, self.config.s3_bucket, s3_key
                )
                logger.info(f"✅ Model uploaded to {self.config.s3_model_uri}")
                return self.config.s3_model_uri

        except ValueError:
            # Re-raise ValueError with our custom message
            raise
        except Exception as e:
            logger.error(f"Failed to upload model to S3: {str(e)}")
            raise

    def create_model(self, model_data_url: str, image_uri: str) -> str:
        """
        Create SageMaker model.

        Args:
            model_data_url: S3 URI of model.tar.gz
            image_uri: ECR image URI

        Returns:
            Model name
        """
        try:
            model_name = f"{self.config.model_name}-{int(time.time())}"
            logger.info(f"Creating SageMaker model: {model_name}")

            # Check if using custom container or built-in
            primary_container = {
                "Image": image_uri,
                "ModelDataUrl": model_data_url,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": model_data_url,
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": self.config.aws_region,
                },
            }

            response = self.sm_client.create_model(
                ModelName=model_name,
                PrimaryContainer=primary_container,
                ExecutionRoleArn=self.config.sagemaker_role,
            )

            logger.info(f"✅ Model created: {model_name}")
            return model_name

        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise

    def create_endpoint_config(self, model_name: str) -> str:
        """
        Create endpoint configuration.

        Args:
            model_name: SageMaker model name

        Returns:
            Endpoint config name
        """
        try:
            config_name = f"{self.config.endpoint_config_name}-{int(time.time())}"
            logger.info(f"Creating endpoint configuration: {config_name}")

            response = self.sm_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        "VariantName": "primary",
                        "ModelName": model_name,
                        "InstanceType": self.config.instance_type,
                        "InitialInstanceCount": self.config.initial_instance_count,
                        "InitialVariantWeight": 1.0,
                    }
                ],
            )

            logger.info(f"✅ Endpoint configuration created: {config_name}")
            return config_name

        except Exception as e:
            logger.error(f"Failed to create endpoint configuration: {str(e)}")
            raise

    def create_or_update_endpoint(self, endpoint_config_name: str) -> str:
        """
        Create new endpoint or update existing one.

        Args:
            endpoint_config_name: Endpoint configuration name

        Returns:
            Endpoint name
        """
        action = "init"
        endpoint_name = self.config.endpoint_name

        try:
            # Check if endpoint exists
            try:
                response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
                status = response.get("EndpointStatus")
            except self.sm_client.exceptions.ClientError:
                response = None
                status = None

            # If endpoint does not exist, create it
            if response is None:
                logger.info(f"Creating new endpoint: {endpoint_name}")
                self.sm_client.create_endpoint(
                    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
                )
                action = "create"

            else:
                # Handle existing endpoint states
                if status == "Failed":
                    logger.warning(
                        f"Endpoint {endpoint_name} is in Failed state. Deleting..."
                    )
                    self.sm_client.delete_endpoint(EndpointName=endpoint_name)
                    waiter = self.sm_client.get_waiter("endpoint_deleted")
                    waiter.wait(EndpointName=endpoint_name)
                    logger.info("✓ Failed endpoint deleted")
                    logger.info(f"Creating new endpoint: {endpoint_name}")
                    self.sm_client.create_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=endpoint_config_name,
                    )
                    action = "create"

                elif status in [
                    "Creating",
                    "Updating",
                    "RollingBack",
                    "SystemUpdating",
                    "Deleting",
                ]:
                    logger.info(
                        f"Endpoint {endpoint_name} is currently in status '{status}'. Waiting for it to become stable..."
                    )
                    # Wait until endpoint is in service or deleted/failed
                    waiter = self.sm_client.get_waiter("endpoint_in_service")
                    try:
                        waiter.wait(
                            EndpointName=endpoint_name,
                            WaiterConfig={"Delay": 30, "MaxAttempts": 120},
                        )
                        logger.info(
                            f"Endpoint {endpoint_name} is now InService. Proceeding to update."
                        )
                        self.sm_client.update_endpoint(
                            EndpointName=endpoint_name,
                            EndpointConfigName=endpoint_config_name,
                        )
                        action = "update"
                    except Exception:
                        # If waiting fails (e.g., endpoint deleted or moved to Failed), re-describe and decide
                        resp = self.sm_client.describe_endpoint(
                            EndpointName=endpoint_name
                        )
                        new_status = resp.get("EndpointStatus")
                        if new_status == "Failed":
                            logger.warning(
                                f"Endpoint {endpoint_name} moved to Failed. Deleting and recreating..."
                            )
                            self.sm_client.delete_endpoint(EndpointName=endpoint_name)
                            self.sm_client.get_waiter("endpoint_deleted").wait(
                                EndpointName=endpoint_name
                            )
                            self.sm_client.create_endpoint(
                                EndpointName=endpoint_name,
                                EndpointConfigName=endpoint_config_name,
                            )
                            action = "create"
                        else:
                            # Try update as a last resort
                            logger.info(
                                f"Attempting to update endpoint {endpoint_name} after wait"
                            )
                            self.sm_client.update_endpoint(
                                EndpointName=endpoint_name,
                                EndpointConfigName=endpoint_config_name,
                            )
                            action = "update"

                else:
                    # status == InService or other stable states: update
                    logger.info(f"Updating existing endpoint: {endpoint_name}")
                    self.sm_client.update_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=endpoint_config_name,
                    )
                    action = "update"

            # Wait for endpoint to be in service
            logger.info(
                f"Waiting for endpoint to be in service (this may take 5-10 minutes)..."
            )

            waiter = self.sm_client.get_waiter("endpoint_in_service")
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={"Delay": 30, "MaxAttempts": 60},
            )

            logger.info(f"✅ Endpoint {action}d successfully: {endpoint_name}")
            return endpoint_name

        except Exception as e:
            logger.error(f"Failed to {action} endpoint: {str(e)}")
            raise

    def get_endpoint_status(self) -> dict:
        """
        Get current endpoint status.

        Returns:
            Endpoint description
        """
        try:
            response = self.sm_client.describe_endpoint(
                EndpointName=self.config.endpoint_name
            )
            return response
        except self.sm_client.exceptions.ClientError:
            return None

    def deploy(self, local_model_path: str, image_uri: str):
        """
        Complete deployment workflow.

        Args:
            local_model_path: Local path to model.tar.gz
            image_uri: ECR image URI
        """
        try:
            logger.info("\n" + "=" * 60)
            logger.info("Starting SageMaker Deployment")
            logger.info("=" * 60 + "\n")

            # Step 1: Upload model to S3
            logger.info("Step 1/4: Uploading model to S3...")
            model_data_url = self.upload_model_to_s3(local_model_path)

            # Step 2: Create SageMaker model
            logger.info("\nStep 2/4: Creating SageMaker model...")
            model_name = self.create_model(model_data_url, image_uri)

            # Step 3: Create endpoint configuration
            logger.info("\nStep 3/4: Creating endpoint configuration...")
            endpoint_config_name = self.create_endpoint_config(model_name)

            # Step 4: Create or update endpoint
            logger.info("\nStep 4/4: Creating/updating endpoint...")
            endpoint_name = self.create_or_update_endpoint(endpoint_config_name)

            logger.info("\n" + "=" * 60)
            logger.info("Deployment Complete!")
            logger.info("=" * 60)
            logger.info(f"Endpoint Name: {endpoint_name}")
            logger.info(f"Region: {self.config.aws_region}")
            logger.info("\nYou can now invoke the endpoint using:")
            logger.info(f"  aws sagemaker-runtime invoke-endpoint \\")
            logger.info(f"    --endpoint-name {endpoint_name} \\")
            logger.info(f"    --body '{{...}}' \\")
            logger.info(f"    --content-type application/json \\")
            logger.info(f"    response.json")
            logger.info("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"\n❌ Deployment failed: {str(e)}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Deploy model to SageMaker")
    parser.add_argument(
        "--model-path",
        default="../model_package/model.tar.gz",
        help="Path to model.tar.gz file",
    )
    parser.add_argument(
        "--image-uri", help="ECR image URI (if not using built-in container)"
    )
    parser.add_argument(
        "--use-builtin",
        action="store_true",
        help="Use SageMaker built-in scikit-learn container",
    )
    parser.add_argument(
        "--status", action="store_true", help="Check endpoint status only"
    )

    args = parser.parse_args()

    # Load configuration
    config = SageMakerConfig()

    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("\nPlease set the following environment variables:")
        logger.info("  - AWS_ACCOUNT_ID")
        logger.info("  - SAGEMAKER_EXECUTION_ROLE")
        logger.info("  - AWS_REGION (optional, default: us-west-2)")
        logger.info(
            "  - SAGEMAKER_S3_BUCKET (optional, default: psoriasis-ml-sagemaker)"
        )
        return

    deployer = SageMakerDeployer(config)

    # Check status only
    if args.status:
        status = deployer.get_endpoint_status()
        if status:
            logger.info(f"Endpoint Status: {status['EndpointStatus']}")
            logger.info(f"Endpoint ARN: {status['EndpointArn']}")
        else:
            logger.info(f"Endpoint '{config.endpoint_name}' does not exist")
        return

    # Validate model path
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        logger.info("Please run model_package.py first to create model.tar.gz")
        return

    # Determine image URI
    if args.use_builtin:
        # Use SageMaker built-in scikit-learn container
        # Note: This may not work perfectly as we're using LightGBM
        image_uri = f"683313688378.dkr.ecr.{config.aws_region}.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3"
        logger.warning(
            "Using built-in scikit-learn container. This may not work with LightGBM."
        )
        logger.warning("Consider using a custom container with --image-uri")
    elif args.image_uri:
        image_uri = args.image_uri
    else:
        try:
            image_uri = config.ecr_image_uri
        except ValueError:
            logger.error("Image URI not specified and AWS_ACCOUNT_ID not set")
            logger.info(
                "Please specify --image-uri or set AWS_ACCOUNT_ID environment variable"
            )
            return

    logger.info(f"Using image: {image_uri}")

    # Deploy
    try:
        deployer.deploy(args.model_path, image_uri)
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
