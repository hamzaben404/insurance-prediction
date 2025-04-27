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


# Add to existing Makefile

# CI/CD commands
.PHONY: ci-lint ci-test ci-security ci-build cd-build cd-release

# Run linting (for CI)
ci-lint:
	black --check src tests
	isort --check src tests
	flake8 src tests

# Run tests (for CI)
ci-test:
	pytest tests/unit
	pytest tests/integration
	pytest tests/security
	pytest --cov=src tests/ --cov-report=xml

# Run security checks (for CI)
ci-security:
	bandit -r src/
	safety check -r requirements.txt

# Build Docker image (for CI)
ci-build:
	docker build -t insurance-prediction-api:test .

# Build and tag Docker image (for CD)
cd-build:
	docker build -t yourusername/insurance-prediction-api:latest -t yourusername/insurance-prediction-api:$(shell cat VERSION) .

# Release a new version
cd-release:
	bumpversion $(type)