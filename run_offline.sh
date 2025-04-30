#!/bin/bash

# This script runs exo in offline mode from within a virtual environment

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: Not running in a virtual environment. This may cause issues."
    echo "It's recommended to run Exo inside a virtual environment."
    echo "Create one with: python -m venv venv"
    echo "And activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
    echo ""
fi

# Print banner
echo "=============================================="
echo "🔌 Starting Exo in OFFLINE MODE"
echo "Only locally available models will be used"
echo "No downloads will be attempted"
echo "=============================================="

# Run exo with the offline flag and pass any additional arguments
python -m exo.main --offline "$@"

# Note: This script should be run from within the virtual environment
# Usage example: ./run_offline.sh
# Additional arguments can be passed: ./run_offline.sh --default-model mistral-7b
