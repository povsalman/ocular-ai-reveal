from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.glaucoma import GlaucomaDualModel
import base64
import io
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any
import logging
import torch


from torchvision import transforms
from models.dr_model import DRModel
from models.vessel_model import VesselModel
from models.age_model import AgeModel
from models.myopia_model import MyopiaModel
from models.glaucoma_model import GlaucomaModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_glaucoma_image(image_file: UploadFile) -> np.ndarray:
    """Preprocess uploaded image for model inference"""
    try:
        # Read image file
        image_bytes = image_file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transforms.ToTensor()(image)  # Converts to C,H,W float32
        return tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")
    
def encode_mask_to_base64_glaucoma(mask: torch.Tensor | np.ndarray) -> str:
    """
    Encodes a segmentation mask (class labels) into base64 PNG.
    Supports torch.Tensor or np.ndarray inputs.
    """
    try:
        # If torch tensor, move to CPU and convert to numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # If mask is probabilities, convert to labels
        if mask.ndim == 3:
            # Shape (C, H, W): assume channel-first
            mask = np.argmax(mask, axis=0)
        elif mask.ndim == 4:
            # Shape (1, C, H, W): squeeze and argmax
            mask = np.argmax(mask.squeeze(0), axis=0)
        elif mask.ndim != 2:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # Convert to uint8
        mask = mask.astype(np.uint8)

        # Create a PIL Image with palette
        mask_image = Image.fromarray(mask, mode="P")

        # Palette: black, red, green
        palette = [
            0, 0, 0,      # Class 0: black
            255, 0, 0,    # Class 1: red
            0, 255, 0     # Class 2: green
        ]
        mask_image.putpalette(palette)

        # Save to PNG buffer
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Base64 encode
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return encoded

    except Exception as e:
        logger.error(f"Error encoding mask: {e}")
        return ""
