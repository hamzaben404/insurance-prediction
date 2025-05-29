# Makefile

.PHONY: test test-unit test-integration test-e2e test-security test-performance security-scan

# Run all tests
test:
	python run_tests.py

# Run unit tests
test-unit:
	python run_tests.py --type unit

# Run integration tests
test-integration:
	python run_tests.py --type integration

# Run end-to-end tests
test-e2e:
	python run_tests.py --type e2e

# Run security tests
test-security:
	python run_tests.py --type security

# Run performance tests
test-performance:
	python run_tests.py --type performance

# Run security scan with Bandit
security-scan:
	./run_security_scan.sh

# Run all tests and security scan
test-all: test security-scan

# Run tests with coverage
test-coverage:
	python -m pytest --cov=src tests/
	coverage html
	@echo "Coverage report generated in htmlcov directory"


# Run all CI steps locally
ci-check: format lint security-check test

# Corrected format target (using TABS for indentation)
format:
	python scripts/format_code.py

# Add this target to check formatting without modifying files
format-check:
	black --check --config pyproject.toml src tests
	isort --check-only --settings pyproject.toml src tests

lint:
	flake8 src tests

security-check:
	bandit -r src -x tests
	safety check

# Run all CI steps locally
ci-check: format lint security-check test
