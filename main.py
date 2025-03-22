from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import numpy as np
import json
import logging
from model import LSTMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LSTM Model API",
    description="API for making predictions with a LSTM model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = LSTMModel()

# Input data models
class PredictionInput(BaseModel):
    sequence: List[List[float]]
    sequence_length: Optional[int] = None

class PredictionResponse(BaseModel):
    prediction: Union[List[float], float]
    confidence: Optional[float] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "LSTM Model API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionInput):
    """
    Make a prediction with the LSTM model
    
    Args:
        data: Input sequence data for prediction
        
    Returns:
        Prediction result and confidence score
    """
    try:
        logger.info(f"Received prediction request with sequence length: {len(data.sequence)}")
        
        # Convert input to numpy array
        input_sequence = np.array(data.sequence, dtype=np.float32)
        
        # Make prediction
        prediction, confidence = model.predict(input_sequence)
        
        return {
            "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Make a prediction with the LSTM model using data from a file
    
    Args:
        file: JSON file containing sequence data
        
    Returns:
        Prediction result and confidence score
    """
    try:
        logger.info(f"Received file prediction request: {file.filename}")
        
        # Read and parse file content
        content = await file.read()
        data = json.loads(content)
        
        # Validate input data
        if "sequence" not in data:
            raise HTTPException(status_code=400, detail="Input file must contain a 'sequence' field")
        
        # Convert input to numpy array
        input_sequence = np.array(data["sequence"], dtype=np.float32)
        
        # Make prediction
        prediction, confidence = model.predict(input_sequence)
        
        return {
            "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
            "confidence": confidence
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(data: List[PredictionInput]):
    """
    Make batch predictions with the LSTM model
    
    Args:
        data: List of input sequences for prediction
        
    Returns:
        List of prediction results
    """
    try:
        logger.info(f"Received batch prediction request with {len(data)} items")
        
        results = []
        for item in data:
            # Convert input to numpy array
            input_sequence = np.array(item.sequence, dtype=np.float32)
            
            # Make prediction
            prediction, confidence = model.predict(input_sequence)
            
            results.append({
                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                "confidence": confidence
            })
        
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    try:
        info = model.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
