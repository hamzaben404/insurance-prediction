# Continuous Integration Documentation

This document explains the CI processes for the Insurance Prediction project.

## CI Workflows

### Code Quality
Runs on push to main/develop branches and on pull requests.
- Black formatting checks
- isort import ordering
- Flake8 linting
- Bandit security scanning
- Safety dependency checking

### Testing
Runs on push to main/develop branches and on pull requests.
- Unit tests
- Integration tests
- Security tests
- Coverage reporting

### Docker Build
Runs on push to main/develop branches and on pull requests.
- Builds Docker image
- Tests Docker container
- Pushes to Container Registry (only on push, not PR)

### Model Validation
Runs when changes affect model code or data processing.
- Processes test data
- Validates model metrics against thresholds

## CI Best Practices

1. **Keep the build green**: Fix failing CI immediately
2. **Test locally before pushing**: Run tests locally to avoid CI failures
3. **Use meaningful commit messages**: Explain why changes were made
4. **Review CI logs**: Check logs for warnings even when CI passes
5. **Maintain test coverage**: Add tests for new features

## Troubleshooting Common Issues

### Failed Formatting Checks
Run `black src tests` and `isort src tests` locally before committing.

### Failed Tests
Run the specific failing test locally for investigation.

### Docker Build Issues
Test Docker build locally with `docker build -t insurance-prediction-api .`
