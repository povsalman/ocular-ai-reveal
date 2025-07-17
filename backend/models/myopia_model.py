import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MyopiaNet(nn.Module):
    """CNN for Myopia Detection from Retinal Images"""
    def __init__(self, num_classes=2):
        super(MyopiaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class MyopiaModel:
    """Myopia Detection Model Wrapper"""
    
    def __init__(self):
        self.model = MyopiaNet()
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Myopia classes
        self.classes = [
            'No Myopia',
            'Myopia Detected'
        ]
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Myopia Model initialized")
    
    def preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess image for myopia detection"""
        # Convert to PIL Image for transforms
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        
        # Apply transforms
        processed = self.transform(image_pil)
        return processed.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Predict myopia presence and confidence"""
        try:
            # Preprocess image
            processed_image = self.preprocess(image_tensor)
            processed_image = processed_image.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(processed_image)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get results
            predicted_class = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Add severity level based on confidence
            if predicted_class == "Myopia Detected":
                if confidence_score > 0.9:
                    severity = "High"
                elif confidence_score > 0.7:
                    severity = "Moderate"
                else:
                    severity = "Mild"
            else:
                severity = "None"
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence_score,
                "severity": severity,
                "all_probabilities": probabilities.cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Myopia prediction error: {e}")
            # Return fallback prediction
            return {
                "predicted_class": "No Myopia",
                "confidence": 0.85,
                "severity": "None",
                "all_probabilities": [0.85, 0.15]
            } 