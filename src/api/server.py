# src/api/server.py
import argparse
import os

import uvicorn

from src.api.main import app  # Import the app from main.py


def main():
    """Run the API server"""
    parser = argparse.ArgumentParser(description="Start the API server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", # nosec B104
        help="Host to run the server on"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/comparison/production/production_model.pkl",
        help="Path to the model file",
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ["API_PORT"] = str(args.port)
    os.environ["MODEL_PATH"] = args.model_path

    # Run the server
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
