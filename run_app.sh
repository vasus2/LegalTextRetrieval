#!/bin/bash

# Briefly - Quick Start Script
# This script launches the Streamlit UI for the legal precedent search system

echo "Starting Briefly - Legal Precedent Search System..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not detected."
    echo "   Consider activating it with: source .venv/bin/activate"
    echo ""
fi

# Check if required files exist
if [ ! -f "bm25-files/docs00.json" ]; then
    echo "Error: BM25 corpus file not found!"
    echo ""
    echo "Please run the data preparation pipeline first:"
    echo "  1. python briefly/precedent_data_extraction.py"
    echo "  2. python briefly/bm25_pipeline.py"
    echo ""
    exit 1
fi

# Launch Streamlit
echo "Corpus file found. Launching UI..."
echo ""
streamlit run app.py
