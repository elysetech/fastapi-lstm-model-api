#!/bin/bash

# LSTM Model API Runner Script
# This script sets up and runs the LSTM Model API

# Exit on error
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create model directory if it doesn't exist
if [ ! -d "model" ]; then
    echo "Creating model directory..."
    mkdir -p model
fi

# Check if a model file path is provided
MODEL_PATH=""
if [ "$1" != "" ]; then
    MODEL_PATH=$1
    echo "Using model from: $MODEL_PATH"
fi

# Start the API server
echo "Starting LSTM Model API server..."
if [ "$MODEL_PATH" != "" ]; then
    # Pass model path as environment variable
    MODEL_PATH=$MODEL_PATH uvicorn main:app --host 0.0.0.0 --port 8000 --reload
else
    # Use default sample model
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi
