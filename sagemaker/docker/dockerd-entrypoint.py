import sys
from sagemaker_inference import model_server

if __name__ == "__main__":
    try:
        model_server.start_model_server()
    except Exception as e:
        print(f"Failed to start model server: {e}")
        sys.exit(1)
