import unittest
import requests
import subprocess
import time
import numpy as np
import os
import signal
import json
from typing import List, Dict, Any, Optional

class LSTMAPITest(unittest.TestCase):
    """Test case for the LSTM Model API"""
    
    BASE_URL = "http://localhost:8000"
    server_process = None
    
    @classmethod
    def setUpClass(cls):
        """Start the API server before running tests"""
        print("Starting API server...")
        # Start the server in a separate process
        cls.server_process = subprocess.Popen(
            ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Used to kill the process group
        )
        
        # Wait for the server to start
        max_retries = 10
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(f"{cls.BASE_URL}/")
                if response.status_code == 200:
                    print("API server started successfully")
                    break
            except requests.exceptions.ConnectionError:
                pass
            
            retries += 1
            time.sleep(1)
            
        if retries == max_retries:
            raise RuntimeError("Failed to start API server")
    
    @classmethod
    def tearDownClass(cls):
        """Stop the API server after running tests"""
        if cls.server_process:
            print("Stopping API server...")
            # Kill the process group
            os.killpg(os.getpgid(cls.server_process.pid), signal.SIGTERM)
            cls.server_process.wait()
            print("API server stopped")
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["message"], "LSTM Model API is running")
    
    def test_model_info(self):
        """Test the model info endpoint"""
        response = requests.get(f"{self.BASE_URL}/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_type"], "LSTM")
        self.assertIn("input_shape", data)
        self.assertIn("output_shape", data)
        self.assertIn("layers", data)
        self.assertIn("total_params", data)
    
    def generate_sample_data(self, timesteps: int = 10, features: int = 5, samples: int = 1) -> List:
        """Generate sample time series data for testing"""
        data = []
        for _ in range(samples):
            sample = []
            for t in range(timesteps):
                feature_values = []
                for f in range(features):
                    value = np.sin(t/5 + f/2) + np.random.normal(0, 0.1)
                    feature_values.append(float(value))
                sample.append(feature_values)
            data.append(sample)
        
        return data[0] if samples == 1 else data
    
    def test_predict_endpoint(self):
        """Test the predict endpoint"""
        # Get model info to determine input shape
        response = requests.get(f"{self.BASE_URL}/model-info")
        self.assertEqual(response.status_code, 200)
        model_info = response.json()
        
        input_shape = model_info["input_shape"]
        timesteps, features = input_shape if len(input_shape) > 1 else (10, 5)
        
        # Generate sample data
        sequence = self.generate_sample_data(timesteps, features)
        
        # Make prediction
        payload = {"sequence": sequence}
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        
        # Check prediction type
        if isinstance(data["prediction"], list):
            self.assertTrue(all(isinstance(x, (int, float)) for x in data["prediction"]))
        else:
            self.assertTrue(isinstance(data["prediction"], (int, float)))
        
        # Check confidence
        self.assertTrue(isinstance(data["confidence"], (int, float)))
    
    def test_batch_predict_endpoint(self):
        """Test the batch predict endpoint"""
        # Get model info to determine input shape
        response = requests.get(f"{self.BASE_URL}/model-info")
        self.assertEqual(response.status_code, 200)
        model_info = response.json()
        
        input_shape = model_info["input_shape"]
        timesteps, features = input_shape if len(input_shape) > 1 else (10, 5)
        
        # Generate batch data (3 sequences)
        batch_sequences = self.generate_sample_data(timesteps, features, samples=3)
        
        # Make batch prediction
        payload = [{"sequence": seq} for seq in batch_sequences]
        response = requests.post(f"{self.BASE_URL}/batch-predict", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 3)
        
        for result in data:
            self.assertIn("prediction", result)
            self.assertIn("confidence", result)
    
    def test_predict_from_file(self):
        """Test the predict from file endpoint"""
        # Get model info to determine input shape
        response = requests.get(f"{self.BASE_URL}/model-info")
        self.assertEqual(response.status_code, 200)
        model_info = response.json()
        
        input_shape = model_info["input_shape"]
        timesteps, features = input_shape if len(input_shape) > 1 else (10, 5)
        
        # Generate sample data
        sequence = self.generate_sample_data(timesteps, features)
        
        # Save to file
        filename = "test_sequence.json"
        with open(filename, "w") as f:
            json.dump({"sequence": sequence}, f)
        
        # Make prediction from file
        with open(filename, "rb") as f:
            files = {"file": (filename, f, "application/json")}
            response = requests.post(f"{self.BASE_URL}/predict/file", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
    
    def test_invalid_input(self):
        """Test API behavior with invalid input"""
        # Test with empty sequence
        payload = {"sequence": []}
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        self.assertNotEqual(response.status_code, 200)
        
        # Test with invalid JSON
        response = requests.post(
            f"{self.BASE_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        self.assertNotEqual(response.status_code, 200)
        
        # Test with missing required field
        payload = {"not_sequence": [[1, 2, 3]]}
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        self.assertNotEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
