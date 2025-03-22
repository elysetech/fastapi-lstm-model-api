import os
import numpy as np
import logging
import math
import random
from typing import Tuple, List, Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMModel:
    """
    Simplified LSTM model implementation using NumPy for demonstration purposes
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the LSTM model
        
        Args:
            model_path: Path to the saved model file. If None, a sample model will be created.
        """
        self.input_shape = (10, 5)  # 10 time steps, 5 features
        self.output_shape = (1,)    # Single output value
        self.model_path = model_path
        self.is_sample_model = True
        
        # Create weights for our simplified model
        self.weights = {
            'input_weights': np.random.randn(5, 8) * 0.01,
            'recurrent_weights': np.random.randn(8, 8) * 0.01,
            'output_weights': np.random.randn(8, 1) * 0.01,
            'bias': np.zeros(8),
            'output_bias': np.zeros(1)
        }
        
        logger.info("Created simplified LSTM model for demonstration")
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data before prediction
        
        Args:
            data: Input data as numpy array
            
        Returns:
            Preprocessed data ready for model input
        """
        # Check if data needs reshaping
        if len(data.shape) == 2:
            expected_timesteps, expected_features = self.input_shape
            
            # If data doesn't match expected shape, try to reshape
            if data.shape[0] != expected_timesteps or data.shape[1] != expected_features:
                logger.warning(f"Input shape {data.shape} doesn't match expected shape {self.input_shape}")
                
                # If we have more timesteps than expected, truncate
                if data.shape[0] > expected_timesteps:
                    data = data[-expected_timesteps:, :]
                    logger.info(f"Truncated input to last {expected_timesteps} timesteps")
                
                # If we have fewer timesteps, pad with zeros
                elif data.shape[0] < expected_timesteps:
                    padding = np.zeros((expected_timesteps - data.shape[0], data.shape[1]))
                    data = np.vstack((padding, data))
                    logger.info(f"Padded input to {expected_timesteps} timesteps")
                
                # If feature dimensions don't match, log error
                if data.shape[1] != expected_features:
                    raise ValueError(f"Input has {data.shape[1]} features but model expects {expected_features}")
        
        return data
    
    def predict(self, data: np.ndarray) -> Tuple[Union[float, np.ndarray], float]:
        """
        Make a prediction with the simplified LSTM model
        
        Args:
            data: Input data as numpy array
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Preprocess the data
            processed_data = self.preprocess(data)
            
            # Simple forward pass (not a real LSTM, just for demonstration)
            # Sum features with weights and apply activation
            hidden_state = np.zeros(8)
            
            for timestep in range(processed_data.shape[0]):
                x = processed_data[timestep]
                
                # Simple recurrent calculation
                hidden_input = np.dot(x, self.weights['input_weights']) + \
                               np.dot(hidden_state, self.weights['recurrent_weights']) + \
                               self.weights['bias']
                
                # Apply activation
                hidden_state = self.tanh(hidden_input)
            
            # Final output
            output = np.dot(hidden_state, self.weights['output_weights']) + self.weights['output_bias']
            
            # Add some randomness based on input to simulate real predictions
            seed_value = np.sum(processed_data) % 1000
            random.seed(seed_value)
            trend = np.mean(processed_data[:, 0])  # Use first feature for trend
            
            # Generate prediction with some randomness but following the trend
            prediction = float(output[0]) + trend + (random.random() - 0.5) * 0.2
            
            # Fixed confidence for demo
            confidence = 0.85 + random.random() * 0.1
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model information
        """
        # Collect model information
        info = {
            "model_type": "LSTM (Simplified)",
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "layers": [
                {
                    "name": "input_layer",
                    "type": "Input",
                    "units": None,
                    "activation": None
                },
                {
                    "name": "lstm_layer",
                    "type": "SimplifiedLSTM",
                    "units": 8,
                    "activation": "tanh"
                },
                {
                    "name": "output_layer",
                    "type": "Dense",
                    "units": 1,
                    "activation": "linear"
                }
            ],
            "total_params": sum(w.size for w in self.weights.values()),
            "is_sample_model": True,
            "note": "This is a simplified model for demonstration purposes only"
        }
        
        return info
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model to a file (simplified version)
        
        Args:
            save_path: Path where to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the weights as a numpy array
            np.savez(save_path, **self.weights)
            self.model_path = save_path
            logger.info(f"Model weights saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")
