#!/bin/bash
echo "Running security scan with Bandit..."
bandit -r src/ -f html -o security_report.html
echo "Security scan complete. Report saved to security_report.html"
