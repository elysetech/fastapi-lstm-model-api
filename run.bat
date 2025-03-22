@echo off
REM LSTM Model API Runner Script for Windows
REM This script sets up and runs the LSTM Model API

echo LSTM Model API Setup

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create model directory if it doesn't exist
if not exist model (
    echo Creating model directory...
    mkdir model
)

REM Check if a model file path is provided
set MODEL_PATH=
if not "%~1"=="" (
    set MODEL_PATH=%~1
    echo Using model from: %MODEL_PATH%
)

REM Start the API server
echo Starting LSTM Model API server...
if not "%MODEL_PATH%"=="" (
    REM Pass model path as environment variable
    set MODEL_PATH=%MODEL_PATH%
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
) else (
    REM Use default sample model
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
)
