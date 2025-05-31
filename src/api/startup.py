"""Startup tasks for the API"""
import subprocess  # nosec B404
import sys
from pathlib import Path

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def download_production_model():
    """Download production model on startup"""
    model_path = Path("models/comparison/production/production_model.pkl")

    # Check if model already exists
    if model_path.exists():
        logger.info("Production model already exists")
        return

    # Try to download from GitHub releases
    logger.info("Downloading production model...")
    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, "scripts/download_model.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Model downloaded successfully")
        else:
            logger.warning(f"Model download failed: {result.stderr}")
            # Create dummy model as fallback
            subprocess.run(
                [sys.executable, "scripts/create_dummy_model.py"], check=False
            )  # nosec B603
    except Exception as e:
        logger.error(f"Error during model download: {e}")
        # Create dummy model as fallback
        subprocess.run([sys.executable, "scripts/create_dummy_model.py"], check=False)  # nosec B603


def run_startup_tasks():
    """Run all startup tasks"""
    logger.info("Running startup tasks...")

    # Download model if needed
    download_production_model()

    # Add other startup tasks here
    logger.info("Startup tasks completed")
