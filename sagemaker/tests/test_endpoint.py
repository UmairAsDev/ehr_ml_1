"""
Test SageMaker endpoint with sample patient data.

This script tests the deployed SageMaker endpoint with various test cases.
"""

import boto3
import json
import logging
import argparse
from typing import Dict, Any
from config import SageMakerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample test data
SAMPLE_PATIENT_NOTE = {
    "patientId": "TEST001",
    "noteId": "NOTE001",
    "noteDate": "2024-01-15",
    "patientSummary": "45 year old Male patient",
    "complaints": "Patient reports increased itching and redness on elbows and knees. Symptoms worsening over past 2 weeks.",
    "assesment": "Psoriasis flare-up. Patient experiencing increased scaling and inflammation.",
    "examination": "Multiple erythematous plaques with silvery scale on bilateral elbows. Hyperpigmentation noted.",
    "reviewofsystem": "Reports persistent itching and dry skin. No fever.",
    "currentmedication": "Triamcinolone cream 0.1% applied twice daily",
    "pastHistory": "Non-smoker. Alcohol use: yes, social. Family history: melanoma in father.",
    "diagnoses": "L40.0 Psoriasis vulgaris, Plaque psoriasis",
    "procedure": "",
    "allergy": "No known drug allergies",
}

SAMPLE_BATCH_REQUEST = {
    "notes": [
        {
            "patientId": "TEST001",
            "noteId": "NOTE001",
            "noteDate": "2024-01-15",
            "patientSummary": "45 year old Male patient",
            "complaints": "Mild itching on elbows",
            "assesment": "Stable psoriasis",
            "examination": "Few small plaques on elbows",
            "reviewofsystem": "Minimal symptoms",
            "currentmedication": "Moisturizer daily",
            "pastHistory": "Non-smoker",
            "diagnoses": "L40.0 Psoriasis vulgaris",
        },
        {
            "patientId": "TEST001",
            "noteId": "NOTE002",
            "noteDate": "2024-02-15",
            "patientSummary": "45 year old Male patient",
            "complaints": "Severe itching and scaling. No relief from current treatment.",
            "assesment": "Psoriasis flare. Consider starting steroid therapy.",
            "examination": "Multiple erythematous plaques with thick silvery scale on elbows and knees",
            "reviewofsystem": "Intense itching. Dry skin.",
            "currentmedication": "Starting triamcinolone ointment",
            "pastHistory": "Non-smoker",
            "diagnoses": "L40.0 Psoriasis vulgaris, Plaque psoriasis",
        },
    ]
}


class EndpointTester:
    """Test SageMaker endpoint."""

    def __init__(self, endpoint_name: str, region: str):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client("sagemaker-runtime", region_name=region)

    def invoke_endpoint(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke SageMaker endpoint.

        Args:
            payload: Input data

        Returns:
            Endpoint response
        """
        try:
            logger.info(f"Invoking endpoint: {self.endpoint_name}")

            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )

            result = json.loads(response["Body"].read().decode())

            logger.info("✅ Endpoint invoked successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to invoke endpoint: {str(e)}")
            raise

    def test_single_prediction(self):
        """Test single patient note prediction."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 1: Single Patient Note Prediction")
        logger.info("=" * 60)

        result = self.invoke_endpoint(SAMPLE_PATIENT_NOTE)

        logger.info("\nInput:")
        logger.info(f"  Patient ID: {SAMPLE_PATIENT_NOTE['patientId']}")
        logger.info(f"  Complaints: {SAMPLE_PATIENT_NOTE['complaints'][:60]}...")

        logger.info("\nOutput:")
        logger.info(json.dumps(result, indent=2))

        if "error" not in result:
            logger.info(f"\n✅ Prediction successful!")
            logger.info(f"  Risk Level: {result.get('risk_level')}")
            logger.info(f"  Flare Probability: {result.get('flare_probability')}")
        else:
            logger.error(f"\n❌ Prediction failed: {result['error']}")

        return result

    def test_batch_prediction(self):
        """Test batch prediction."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 2: Batch Prediction")
        logger.info("=" * 60)

        result = self.invoke_endpoint(SAMPLE_BATCH_REQUEST)

        logger.info(f"\nInput: {len(SAMPLE_BATCH_REQUEST['notes'])} patient notes")

        logger.info("\nOutput:")
        logger.info(json.dumps(result, indent=2))

        if "predictions" in result:
            logger.info(f"\n✅ Batch prediction successful!")
            for i, pred in enumerate(result["predictions"]):
                logger.info(f"\n  Note {i+1}:")
                logger.info(f"    Risk Level: {pred.get('risk_level')}")
                logger.info(f"    Probability: {pred.get('flare_probability')}")
        else:
            logger.error(f"\n❌ Batch prediction failed")

        return result

    def test_edge_cases(self):
        """Test edge cases."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 3: Edge Cases")
        logger.info("=" * 60)

        # Test with minimal data
        minimal_note = {
            "patientId": "TEST_MIN",
            "noteId": "MIN001",
            "complaints": "Itching",
        }

        logger.info("\nTest 3a: Minimal data")
        try:
            result = self.invoke_endpoint(minimal_note)
            logger.info("✅ Handled minimal data successfully")
            logger.info(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"❌ Failed with minimal data: {str(e)}")

        # Test with empty fields
        empty_note = {
            "patientId": "TEST_EMPTY",
            "noteId": "EMPTY001",
            "complaints": "",
            "assesment": "",
            "examination": "",
        }

        logger.info("\nTest 3b: Empty fields")
        try:
            result = self.invoke_endpoint(empty_note)
            logger.info("✅ Handled empty fields successfully")
            logger.info(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"❌ Failed with empty fields: {str(e)}")

    def test_performance(self, num_requests: int = 10):
        """Test endpoint performance."""
        logger.info("\n" + "=" * 60)
        logger.info(f"Test 4: Performance ({num_requests} requests)")
        logger.info("=" * 60)

        import time

        times = []
        for i in range(num_requests):
            start = time.time()
            try:
                self.invoke_endpoint(SAMPLE_PATIENT_NOTE)
                elapsed = time.time() - start
                times.append(elapsed)
                logger.info(f"Request {i+1}/{num_requests}: {elapsed:.3f}s")
            except Exception as e:
                logger.error(f"Request {i+1} failed: {str(e)}")

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            logger.info(f"\n✅ Performance Test Complete")
            logger.info(f"  Average: {avg_time:.3f}s")
            logger.info(f"  Min: {min_time:.3f}s")
            logger.info(f"  Max: {max_time:.3f}s")
            logger.info(f"  Requests/sec: {1/avg_time:.2f}")

    def run_all_tests(self, include_performance: bool = False):
        """Run all tests."""
        logger.info("\n" + "=" * 60)
        logger.info("Running All Tests")
        logger.info("=" * 60)

        try:
            self.test_single_prediction()
            self.test_batch_prediction()
            self.test_edge_cases()

            if include_performance:
                self.test_performance()

            logger.info("\n" + "=" * 60)
            logger.info("All Tests Complete!")
            logger.info("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"\nTests failed: {str(e)}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Test SageMaker endpoint")
    parser.add_argument(
        "--endpoint-name", help="SageMaker endpoint name (default from config)"
    )
    parser.add_argument("--region", help="AWS region (default from config)")
    parser.add_argument(
        "--test",
        choices=["single", "batch", "edge", "performance", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests for performance test",
    )
    parser.add_argument(
        "--custom-payload", help="Path to JSON file with custom payload"
    )

    args = parser.parse_args()

    # Load configuration
    config = SageMakerConfig()
    endpoint_name = args.endpoint_name or config.endpoint_name
    region = args.region or config.aws_region

    logger.info(f"Testing endpoint: {endpoint_name}")
    logger.info(f"Region: {region}")

    tester = EndpointTester(endpoint_name, region)

    # Custom payload test
    if args.custom_payload:
        with open(args.custom_payload, "r") as f:
            payload = json.load(f)

        logger.info(f"\nTesting with custom payload from: {args.custom_payload}")
        result = tester.invoke_endpoint(payload)
        logger.info("\nResult:")
        logger.info(json.dumps(result, indent=2))
        return

    # Run tests
    try:
        if args.test == "single":
            tester.test_single_prediction()
        elif args.test == "batch":
            tester.test_batch_prediction()
        elif args.test == "edge":
            tester.test_edge_cases()
        elif args.test == "performance":
            tester.test_performance(args.num_requests)
        elif args.test == "all":
            tester.run_all_tests(include_performance=False)

    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
