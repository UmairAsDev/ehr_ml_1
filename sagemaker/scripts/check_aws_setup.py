#!/usr/bin/env python3
"""
Check AWS setup and list available S3 buckets.
This helps identify which bucket to use for SageMaker deployment.
"""

import boto3
import sys
from botocore.exceptions import ClientError, NoCredentialsError


def check_s3_buckets(region="us-west-2"):
    """List all S3 buckets the user has access to."""
    try:
        s3_client = boto3.client("s3", region_name=region)

        print("\n" + "=" * 60)
        print("Checking S3 Buckets")
        print("=" * 60)

        # List all buckets
        response = s3_client.list_buckets()
        buckets = response.get("Buckets", [])

        if not buckets:
            print("❌ No S3 buckets found")
            print("\nYou need to either:")
            print("  1. Ask your AWS admin to create a bucket for you")
            print("  2. Request s3:CreateBucket permission")
            return None

        print(f"\n✓ Found {len(buckets)} S3 bucket(s):\n")

        accessible_buckets = []
        for bucket in buckets:
            bucket_name = bucket["Name"]

            # Check if we can access this bucket
            try:
                s3_client.head_bucket(Bucket=bucket_name)

                # Try to get bucket location
                try:
                    location_response = s3_client.get_bucket_location(
                        Bucket=bucket_name
                    )
                    location = (
                        location_response.get("LocationConstraint") or "us-east-1"
                    )
                except:
                    location = "unknown"

                print(f"  ✓ {bucket_name} (region: {location})")
                accessible_buckets.append((bucket_name, location))

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                print(f"  ✗ {bucket_name} (access denied)")

        if accessible_buckets:
            print(f"\n{'='*60}")
            print("RECOMMENDED ACTION")
            print("=" * 60)
            print("\nYou can use one of the accessible buckets above.")
            print("Set the environment variable:\n")

            # Recommend a bucket in the same region if possible
            same_region_buckets = [b for b in accessible_buckets if b[1] == region]
            if same_region_buckets:
                recommended = same_region_buckets[0][0]
            else:
                recommended = accessible_buckets[0][0]

            print(f"  export SAGEMAKER_S3_BUCKET={recommended}")
            print("\nThen run deployment again:")
            print("  python deploy_model.py --model-path model_package/model.tar.gz")
            print("=" * 60 + "\n")

            return accessible_buckets
        else:
            print("\n❌ No accessible buckets found")
            return None

    except NoCredentialsError:
        print("❌ No AWS credentials found")
        print("Please configure AWS credentials first")
        return None
    except Exception as e:
        print(f"❌ Error checking S3 buckets: {str(e)}")
        return None


def check_iam_permissions():
    """Check current IAM user and permissions."""
    try:
        sts_client = boto3.client("sts")

        print("\n" + "=" * 60)
        print("Checking IAM Identity")
        print("=" * 60)

        identity = sts_client.get_caller_identity()

        print(f"\nUser ARN: {identity['Arn']}")
        print(f"Account ID: {identity['Account']}")
        print(f"User ID: {identity['UserId']}")

    except Exception as e:
        print(f"❌ Error checking IAM identity: {str(e)}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Check AWS setup for SageMaker")
    parser.add_argument(
        "--region", default="us-west-2", help="AWS region (default: us-west-2)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AWS SageMaker Setup Checker")
    print("=" * 60)

    # Check IAM identity
    check_iam_permissions()

    # Check S3 buckets
    buckets = check_s3_buckets(args.region)

    if not buckets:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\nContact your AWS administrator to:")
        print("  1. Create an S3 bucket for SageMaker, OR")
        print("  2. Grant you access to an existing bucket, OR")
        print("  3. Grant you s3:CreateBucket permission")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
