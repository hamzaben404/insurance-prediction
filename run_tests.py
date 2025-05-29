# run_tests.py
import argparse
import subprocess
import sys


def run_command(command):
    """Run a command and return exit code"""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    return result.returncode


def run_tests(test_type="all", verbose=False):
    """Run the specified tests"""
    base_command = ["pytest"]

    if verbose:
        base_command.append("-v")

    if test_type == "unit":
        command = base_command + ["tests/unit/"]
    elif test_type == "integration":
        command = base_command + ["tests/integration/"]
    elif test_type == "e2e":
        command = base_command + ["tests/e2e/"]
    elif test_type == "security":
        command = base_command + ["tests/security/"]
    elif test_type == "performance":
        command = base_command + ["tests/performance/"]
    else:  # all
        command = base_command + ["tests/"]

    return run_command(command)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run tests for the insurance prediction API")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "e2e", "security", "performance"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run tests
    exit_code = run_tests(args.type, args.verbose)

    # Exit with the same code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
