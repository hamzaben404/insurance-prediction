#!/bin/bash
# Script to fix all formatting issues

echo "Installing formatting tools..."
pip install black==23.3.0 isort==5.12.0

echo "Running Black formatter..."
black src tests

echo "Running isort..."
isort --profile black src tests

echo "Adding newlines to end of files..."
find . -name "*.yml" -o -name "*.yaml" -o -name "Dockerfile" | while read file; do
    if [ -n "$(tail -c 1 "$file")" ]; then
        echo >> "$file"
    fi
done

echo "Formatting complete!"
