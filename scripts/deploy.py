# scripts/deploy.py
import argparse
import os
import subprocess
import yaml
import sys

def load_config(environment):
    """Load configuration for the specified environment"""
    config_path = f"config/environments/{environment}.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Configuration for {environment} not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def deploy(environment, version, dry_run=False):
    """Deploy the application to the specified environment"""
    print(f"Deploying version {version} to {environment} environment")
    
    # Load environment config
    config = load_config(environment)
    
    # Construct deployment command
    cmd = [
        "docker",
        "run",
        "-d",
        "--name", f"insurance-api-{environment}",
        "-p", f"{config['api']['port']}:8000",
        "-e", f"API_PORT={config['api']['port']}",
        "-e", f"DEBUG={config['api']['debug']}",
        "-e", f"LOG_LEVEL={config['logging']['level']}",
        f"yourusername/insurance-prediction-api:{version}"
    ]
    
    # Execute or print command
    if dry_run:
        print("Would execute: " + " ".join(cmd))
    else:
        try:
            subprocess.run(cmd, check=True)
            print(f"Deployment to {environment} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Deployment failed: {e}")
            sys.exit(1)

def rollback(environment, version, dry_run=False):
    """Rollback to a previous version"""
    print(f"Rolling back {environment} to version {version}")
    
    # Stop current deployment
    stop_cmd = ["docker", "stop", f"insurance-api-{environment}"]
    rm_cmd = ["docker", "rm", f"insurance-api-{environment}"]
    
    if dry_run:
        print("Would execute: " + " ".join(stop_cmd))
        print("Would execute: " + " ".join(rm_cmd))
    else:
        try:
            subprocess.run(stop_cmd, check=True)
            subprocess.run(rm_cmd, check=True)
            print(f"Stopped current deployment in {environment}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not stop current deployment: {e}")
    
    # Deploy specified version
    deploy(environment, version, dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy or rollback the Insurance Prediction API")
    parser.add_argument("action", choices=["deploy", "rollback"], help="Action to perform")
    parser.add_argument("environment", choices=["staging", "production"], help="Deployment environment")
    parser.add_argument("version", help="Version to deploy or rollback to")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    
    args = parser.parse_args()
    
    if args.action == "deploy":
        deploy(args.environment, args.version, args.dry_run)
    elif args.action == "rollback":
        rollback(args.environment, args.version, args.dry_run)