#!/bin/bash
# Script to fix all issues before committing

echo "Running pre-commit fixes..."

# Run formatting
echo "1. Running Black formatter..."
black src tests

echo "2. Running isort..."
isort --profile black src tests

echo "3. Fixing end-of-file newlines..."
for file in $(find . -name "*.yml" -o -name "*.yaml" -o -name "Dockerfile" -o -name "*.py" -o -name "*.txt" | grep -v ".git"); do
    if [ -f "$file" ] && [ -n "$(tail -c 1 "$file")" ]; then
        echo >> "$file"
        echo "Fixed: $file"
    fi
done

echo "4. Running flake8 check..."
flake8 src tests || echo "Flake8 found issues"

echo "5. Staging all changes..."
git add -A

echo "Pre-commit fixes complete!"
echo "You can now commit your changes."
