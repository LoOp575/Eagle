#!/usr/bin/env bash
set -e

# Check if .env exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "WARNING: .env file not found. Copied from .env.example - please edit with your actual values."
    else
        echo "ERROR: No .env or .env.example found. Create a .env file before running."
        exit 1
    fi
fi

# Check if dependencies are installed
if ! python -c "import ccxt" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run main.py with passed arguments or default to scan-only
if [ $# -eq 0 ]; then
    python main.py --mode scan-only --exchange binance
else
    python main.py "$@"
fi
