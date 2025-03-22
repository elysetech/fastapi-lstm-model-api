import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print(f"API Status: {response.json()['status']}")
            print(f"Message: {response.json()['message']}")
            return True
        else:
            print(f"API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return False

def get_model_info():
    """Get information about the LSTM model"""
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            model_info = response.json()
            print("\nModel Information:")
            print(f"  Model Type: {model_info['model_type']}")
            print(f"  Input Shape: {model_info['input_shape']}")
            print(f"  Output Shape: {model_info['output_shape']}")
            print(f"  Total Parameters: {model_info['total_params']}")
            print(f"  Is Sample Model: {model_info['is_sample_model']}")
            
            print("\nModel Layers:")
            for i, layer in enumerate(model_info['layers']):
                print(f"  Layer {i+1}: {layer['name']} ({layer['type']})")
                if layer['units']:
                    print(f"    Units: {layer['units']}")
                if layer['activation']:
                    print(f"    Activation: {layer['activation']}")
            
            return model_info
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return None

def generate_sample_data(timesteps: int = 10, features: int = 5, samples: int = 1):
    """Generate sample time series data for testing"""
    # Create a sine wave with some noise for each feature
    data = []
    for _ in range(samples):
        sample = []
        for t in range(timesteps):
            # Generate features with some correlation
            feature_values = []
            for f in range(features):
                # Base sine wave with different phase for each feature
                value = np.sin(t/5 + f/2) + np.random.normal(0, 0.1)
                feature_values.append(float(value))
            sample.append(feature_values)
        data.append(sample)
    
    return data[0] if samples == 1 else data

def make_prediction(sequence: List[List[float]]):
    """Make a prediction using the API"""
    try:
        payload = {
            "sequence": sequence
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Result:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']}")
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return None

def make_batch_prediction(sequences: List[List[List[float]]]):
    """Make batch predictions using the API"""
    try:
        # Convert sequences to the expected format
        payload = [{"sequence": seq} for seq in sequences]
        
        response = requests.post(
            f"{BASE_URL}/batch-predict",
            json=payload
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\nBatch Prediction Results:")
            for i, result in enumerate(results):
                print(f"  Sequence {i+1}:")
                print(f"    Prediction: {result['prediction']}")
                print(f"    Confidence: {result['confidence']}")
            return results
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return None

def save_sequence_to_file(sequence: List[List[float]], filename: str = "sequence.json"):
    """Save a sequence to a JSON file"""
    with open(filename, "w") as f:
        json.dump({"sequence": sequence}, f)
    print(f"Sequence saved to {filename}")
    return filename

def predict_from_file(filename: str):
    """Make a prediction using a file upload"""
    try:
        with open(filename, "rb") as f:
            files = {"file": (filename, f, "application/json")}
            response = requests.post(
                f"{BASE_URL}/predict/file",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print("\nFile Prediction Result:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']}")
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return None
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None

def visualize_sequence(sequence: List[List[float]], prediction=None):
    """Visualize the input sequence and prediction"""
    sequence_array = np.array(sequence)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot each feature as a line
    for i in range(sequence_array.shape[1]):
        ax1.plot(sequence_array[:, i], label=f'Feature {i+1}')
    
    ax1.set_title('Input Sequence Features')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the sequence as a heatmap
    im = ax2.imshow(sequence_array.T, aspect='auto', cmap='viridis')
    ax2.set_title('Sequence Heatmap')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Feature')
    plt.colorbar(im, ax=ax2, label='Value')
    
    # Add prediction if provided
    if prediction is not None:
        if isinstance(prediction, list) and len(prediction) == 1:
            prediction = prediction[0]
        
        fig.suptitle(f'Sequence Visualization (Prediction: {prediction:.4f})')
    else:
        fig.suptitle('Sequence Visualization')
    
    plt.tight_layout()
    plt.savefig('sequence_visualization.png')
    print("Visualization saved to sequence_visualization.png")
    plt.close()

def main():
    """Main function to demonstrate API usage"""
    print("LSTM Model API Example")
    print("=====================\n")
    
    # Check if the API is running
    if not check_api_status():
        return
    
    # Get model information
    model_info = get_model_info()
    if not model_info:
        return
    
    # Extract input shape from model info
    input_shape = model_info['input_shape']
    timesteps, features = input_shape if len(input_shape) > 1 else (10, 5)
    
    print(f"\nGenerating sample data with {timesteps} timesteps and {features} features...")
    
    # Generate sample data
    sequence = generate_sample_data(timesteps, features)
    
    # Make a prediction
    result = make_prediction(sequence)
    
    if result:
        # Visualize the sequence and prediction
        visualize_sequence(sequence, result['prediction'])
        
        # Save sequence to file
        filename = save_sequence_to_file(sequence)
        
        # Make prediction from file
        file_result = predict_from_file(filename)
        
        # Generate batch data
        print("\nGenerating batch data...")
        batch_sequences = generate_sample_data(timesteps, features, samples=3)
        
        # Make batch prediction
        batch_results = make_batch_prediction(batch_sequences)
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()
