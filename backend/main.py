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

# Import model modules
from torchvision import transforms
from models.dr_model import DRModel
from models.vessel_model import VesselModel
from models.age_model import AgeModel
from models.myopia_model import MyopiaModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Healthcare Backend",
    description="Backend API for AI-powered retinal image analysis",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
models: Dict[str, Any] = {}

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    logger.info("Loading AI models...")
    try:
        models['dr'] = DRModel()
        models['vessel'] = VesselModel()
        models['age'] = AgeModel()
        models['myopia'] = MyopiaModel()
        models['glaucoma'] = GlaucomaDualModel(
            segmentation_checkpoint="/Users/Adeena/Downloads/unet_origa2.pth",
            classification_checkpoint="/Users/Adeena/Downloads/best_model1.pth",
            device="cpu"
        )
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

def preprocess_image(image_file: UploadFile) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    try:
        image_bytes = image_file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transforms.ToTensor()(image)
        return tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def encode_mask_to_base64(mask: torch.Tensor | np.ndarray) -> str:
    """
    Encodes a segmentation mask (class labels) into base64 PNG.
    """
    try:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        if mask.ndim == 3:
            mask = np.argmax(mask, axis=0)
        elif mask.ndim == 4:
            mask = np.argmax(mask.squeeze(0), axis=0)
        elif mask.ndim != 2:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        mask = mask.astype(np.uint8)
        mask_image = Image.fromarray(mask, mode="P")

        palette = [
            0, 0, 0,      # Class 0: black
            255, 0, 0,    # Class 1: red
            0, 255, 0     # Class 2: green
        ]
        mask_image.putpalette(palette)

        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return encoded

    except Exception as e:
        logger.error(f"Error encoding mask: {e}")
        return ""

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    cdr: Optional[float] = Form(0.5)
):
    """
    Predict using the specified model
    """
    valid_models = ['dr', 'vessel', 'age', 'myopia', 'glaucoma']
    if model_type not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_type. Must be one of: {valid_models}"
        )

    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        model = models.get(model_type)
        if not model:
            raise HTTPException(
                status_code=500,
                detail=f"Model {model_type} not loaded"
            )
        
        image_array = preprocess_image(file)
        result = model.predict(image_array)

        response = {
            "status": result.get("status", "success"),
            "predicted_class": result.get("predicted_class"),
            "model_type": model_type
        }

        if model_type == 'glaucoma':
            response["cdr"] = result.get("cdr")

        else:
            response["confidence"] = result.get("confidence")
            if model_type == 'age' and "predicted_age" in result:
                response["predicted_age"] = result["predicted_age"]

        if model_type == 'dr':
            response["gradcam_image"] = result.get("gradcam_image")
            response["model_used"] = result.get("model_used")
            response["all_probabilities"] = result.get("all_probabilities")

        if model_type == 'vessel':
            if "dataset_used" in result:
                response["dataset_used"] = result["dataset_used"]
            if "metrics" in result:
                response["metrics"] = result["metrics"]

        if model_type in ['vessel', 'glaucoma'] and "mask" in result:
            response["mask_image"] = encode_mask_to_base64(result["mask"])

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Healthcare Backend API",
        "version": "1.0.0",
        "available_models": list(models.keys()),
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
