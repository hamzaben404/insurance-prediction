#!/usr/bin/env python3
"""Download model from GitHub Releases"""
import os
from pathlib import Path

import requests


def download_model(repo, tag="latest", output_dir="models/comparison/production"):
    """Download model from GitHub releases"""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # GitHub API URL
    if tag == "latest":
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    else:
        api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"

    try:
        # Get release info
        response = requests.get(api_url)
        response.raise_for_status()
        release = response.json()

        # Find model asset
        model_asset = None
        for asset in release.get("assets", []):
            if asset["name"].endswith(".pkl"):
                model_asset = asset
                break

        if not model_asset:
            print("No model file found in release")
            return False

        # Download model
        download_url = model_asset["browser_download_url"]
        output_path = os.path.join(output_dir, "production_model.pkl")

        print(f"Downloading model from {download_url}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        # Save model
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Model downloaded successfully to {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    # Get repo from environment or use default
    repo = os.getenv("GITHUB_REPOSITORY", "hamzaben404/insurance-prediction")
    tag = os.getenv("MODEL_VERSION", "latest")

    if download_model(repo, tag):
        print("Model download complete")
    else:
        print("Using dummy model for development")
        # Create dummy model if download fails
        from scripts.create_dummy_model import create_dummy_model

        create_dummy_model()
