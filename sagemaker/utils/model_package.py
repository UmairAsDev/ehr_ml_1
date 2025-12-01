"""
Model packaging utility for AWS SageMaker deployment.

This script packages MLflow model artifacts into the SageMaker-compatible format (model.tar.gz).
"""

import os
import sys
import tarfile
import shutil
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def package_model_from_mlflow(
    mlflow_run_id=None,
    model_name="flare_detector_v1",
    output_dir="./model_package",
    include_code=False
):
    """
    Package model artifacts from MLflow for SageMaker deployment.
    
    Args:
        mlflow_run_id: Specific MLflow run ID (if None, uses latest from model registry)
        model_name: Registered model name in MLflow
        output_dir: Output directory for model.tar.gz
        include_code: Whether to include inference code in the package
        
    Returns:
        Path to model.tar.gz file
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        logger.info(f"Packaging model: {model_name}")
        
        # Set MLflow tracking URI
        mlflow_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "file://" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
        )
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow tracking URI: {mlflow_uri}")
        
        client = MlflowClient()
        
        # Get run ID from model registry or use provided one
        if mlflow_run_id is None:
            versions = client.get_latest_versions(model_name, stages=["None"])
            if not versions:
                raise RuntimeError(f"No model versions found for {model_name}")
            mlflow_run_id = versions[0].run_id
            logger.info(f"Using latest model version from run: {mlflow_run_id}")
        
        # Download artifacts from MLflow
        logger.info("Downloading artifacts from MLflow...")
        artifacts_dir = mlflow.artifacts.download_artifacts(
            run_id=mlflow_run_id,
            artifact_path=None
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        temp_model_dir = os.path.join(output_dir, "model")
        os.makedirs(temp_model_dir, exist_ok=True)
        
        # Copy required artifacts
        logger.info("Copying model artifacts...")
        
        # Model file
        model_src = os.path.join(artifacts_dir, "model_files", "lgbm_model.pkl")
        if os.path.exists(model_src):
            shutil.copy(model_src, os.path.join(temp_model_dir, "lgbm_model.pkl"))
            logger.info("✓ Copied lgbm_model.pkl")
        else:
            raise FileNotFoundError(f"Model file not found: {model_src}")
        
        # Preprocessing artifacts
        preprocessing_files = ["tfidf.joblib", "svd.joblib", "scaler.joblib"]
        for filename in preprocessing_files:
            src = os.path.join(artifacts_dir, "preprocessing", filename)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(temp_model_dir, filename))
                logger.info(f"✓ Copied {filename}")
            else:
                logger.warning(f"⚠ Preprocessing file not found: {src}")
        
        # Optionally include inference code
        if include_code:
            inference_src = os.path.join(os.path.dirname(__file__), "inference.py")
            if os.path.exists(inference_src):
                shutil.copy(inference_src, os.path.join(temp_model_dir, "inference.py"))
                logger.info("✓ Copied inference.py")
        
        # Create model.tar.gz
        output_tarball = os.path.join(output_dir, "model.tar.gz")
        logger.info(f"Creating tarball: {output_tarball}")
        
        with tarfile.open(output_tarball, "w:gz") as tar:
            tar.add(temp_model_dir, arcname=".")
        
        # Cleanup temp directory
        shutil.rmtree(temp_model_dir)
        
        # Get file size
        file_size_mb = os.path.getsize(output_tarball) / (1024 * 1024)
        logger.info(f"✅ Model package created: {output_tarball} ({file_size_mb:.2f} MB)")
        
        return output_tarball
        
    except Exception as e:
        logger.error(f"Error packaging model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def package_model_from_directory(
    model_dir,
    output_dir="./model_package"
):
    """
    Package model artifacts from a local directory.
    
    Args:
        model_dir: Directory containing model artifacts
        output_dir: Output directory for model.tar.gz
        
    Returns:
        Path to model.tar.gz file
    """
    try:
        logger.info(f"Packaging model from directory: {model_dir}")
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create model.tar.gz
        output_tarball = os.path.join(output_dir, "model.tar.gz")
        logger.info(f"Creating tarball: {output_tarball}")
        
        with tarfile.open(output_tarball, "w:gz") as tar:
            tar.add(model_dir, arcname=".")
        
        file_size_mb = os.path.getsize(output_tarball) / (1024 * 1024)
        logger.info(f"✅ Model package created: {output_tarball} ({file_size_mb:.2f} MB)")
        
        return output_tarball
        
    except Exception as e:
        logger.error(f"Error packaging model: {str(e)}")
        raise


def verify_package(tarball_path):
    """
    Verify the contents of the model package.
    
    Args:
        tarball_path: Path to model.tar.gz file
    """
    try:
        logger.info(f"Verifying package: {tarball_path}")
        
        required_files = ["lgbm_model.pkl", "tfidf.joblib", "svd.joblib", "scaler.joblib"]
        
        with tarfile.open(tarball_path, "r:gz") as tar:
            members = tar.getnames()
            logger.info(f"Package contains {len(members)} files:")
            
            for member in members:
                logger.info(f"  - {member}")
            
            # Check required files
            missing_files = []
            for required in required_files:
                # Check if file exists in root or any subdirectory
                found = any(required in m for m in members)
                if found:
                    logger.info(f"✓ Found required file: {required}")
                else:
                    missing_files.append(required)
                    logger.warning(f"✗ Missing required file: {required}")
            
            if missing_files:
                logger.warning(f"⚠ Package is missing {len(missing_files)} required files")
                return False
            else:
                logger.info("✅ Package verification successful")
                return True
                
    except Exception as e:
        logger.error(f"Error verifying package: {str(e)}")
        return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Package model for SageMaker deployment")
    parser.add_argument(
        "--source",
        choices=["mlflow", "directory"],
        default="mlflow",
        help="Source of model artifacts"
    )
    parser.add_argument(
        "--model-name",
        default="flare_detector_v1",
        help="MLflow model name (for mlflow source)"
    )
    parser.add_argument(
        "--run-id",
        help="Specific MLflow run ID (optional)"
    )
    parser.add_argument(
        "--model-dir",
        help="Model directory path (for directory source)"
    )
    parser.add_argument(
        "--output-dir",
        default="./model_package",
        help="Output directory for model.tar.gz"
    )
    parser.add_argument(
        "--include-code",
        action="store_true",
        help="Include inference code in package"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify package after creation"
    )
    
    args = parser.parse_args()
    
    try:
        # Package model
        if args.source == "mlflow":
            tarball = package_model_from_mlflow(
                mlflow_run_id=args.run_id,
                model_name=args.model_name,
                output_dir=args.output_dir,
                include_code=args.include_code
            )
        else:
            if not args.model_dir:
                raise ValueError("--model-dir is required when using directory source")
            tarball = package_model_from_directory(
                model_dir=args.model_dir,
                output_dir=args.output_dir
            )
        
        # Verify package
        if args.verify:
            verify_package(tarball)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Model package ready for SageMaker deployment:")
        logger.info(f"  {tarball}")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Packaging failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
