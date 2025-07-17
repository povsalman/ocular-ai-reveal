#!/usr/bin/env python3
"""
Test script for vessel segmentation model
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from models.vessel_model import VesselModel

def create_test_image(size=(512, 512), pattern='random'):
    """Create a test image with different patterns"""
    if pattern == 'random':
        # Random image
        return np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    elif pattern == 'vessels':
        # Image with vessel-like patterns
        img = np.zeros(size + (3,), dtype=np.uint8)
        # Add some vessel-like lines
        for i in range(0, size[0], 50):
            img[i:i+10, :, 1] = 100  # Green channel
        return img
    elif pattern == 'uniform':
        # Uniform gray image
        return np.full(size + (3,), 128, dtype=np.uint8)
    else:
        return np.random.randint(0, 255, size + (3,), dtype=np.uint8)

def test_vessel_model():
    """Test the vessel segmentation model with different images"""
    print("Testing Vessel Segmentation Model...")
    
    try:
        # Initialize model
        model = VesselModel()
        print("✓ Model loaded successfully")
        
        # Test with different image patterns
        test_patterns = ['random', 'vessels', 'uniform']
        
        for i, pattern in enumerate(test_patterns):
            print(f"\n--- Test {i+1}: {pattern} pattern ---")
            
            # Create test image
            test_image = create_test_image(pattern=pattern)
            
            # Run prediction
            result = model.predict(test_image)
            
            # Print results
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Predicted Class: {result.get('predicted_class', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
            print(f"Dataset Used: {result.get('dataset_used', 'unknown')}")
            
            # Print metrics
            metrics = result.get('metrics', {})
            if metrics:
                print("Metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.3f}")
            
            print("-" * 50)
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vessel_model() 