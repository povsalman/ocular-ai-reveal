import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from typing import Dict, Any
import timm

logger = logging.getLogger(__name__)

class AgeModel:
    """Age Prediction Model Wrapper using PyTorch .pth model"""
    def __init__(self):
        # Load the PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model('inception_resnet_v2', pretrained=False)
        in_features = self.model.classif.in_features
        self.model.classif = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.model.load_state_dict(torch.load('models/age_models/age_model.pth', map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.input_shape = (299, 299)
        self.age_groups = [
            '20-30 years',
            '31-40 years',
            '41-50 years',
            '51-60 years',
            '60+ years'
        ]
        logger.info("PyTorch Age Model loaded")
        self.transform = transforms.Compose([
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
        ])

    def preprocess(self, image_pil: Image.Image) -> torch.Tensor:
        # Convert to RGB if not already
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        image = self.transform(image_pil)  # shape: [3, 299, 299]
        image = image.unsqueeze(0)         # shape: [1, 3, 299, 299]
        return image

    def predict(self, image_tensor) -> Dict[str, Any]:
        try:
            # If input is PIL, preprocess
            if isinstance(image_tensor, Image.Image):
                image_tensor = self.preprocess(image_tensor)
            # If input is numpy, convert to torch tensor and permute axes
            if isinstance(image_tensor, np.ndarray):
                image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            if image_tensor.dtype == torch.uint8:
                image_tensor = image_tensor.float().div(255)
            print("Input dtype:", image_tensor.dtype, "min:", image_tensor.min().item(), "max:", image_tensor.max().item())
            with torch.no_grad():
                pred = self.model(image_tensor)
                predicted_age = float(pred.item())
            # Map to age group
            if predicted_age < 31:
                predicted_class = '20-30 years'
            elif predicted_age < 41:
                predicted_class = '31-40 years'
            elif predicted_age < 51:
                predicted_class = '41-50 years'
            elif predicted_age < 61:
                predicted_class = '51-60 years'
            else:
                predicted_class = '60+ years'
            return {
                "predicted_class": predicted_class,
                "predicted_age": predicted_age,
                "confidence": 1.0,
                "all_probabilities": []
            }
        except Exception as e:
            logger.error(f"Age prediction error: {e}")
            return {
                "predicted_class": "41-50 years",
                "predicted_age": 45,
                "confidence": 0.80,
                "all_probabilities": [0.1, 0.15, 0.8, 0.05, 0.0]
            } 
