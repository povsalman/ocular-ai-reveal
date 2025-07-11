from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging
import numpy as np

# Import model modules
from models.dr_model import DRModel
from models.vessel_model import VesselModel
from models.age_model import AgeModel
from models.myopia_model import MyopiaModel
from models.glaucoma_model import GlaucomaModel

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
        # Load all models
        models['dr'] = DRModel()
        models['vessel'] = VesselModel()
        models['age'] = AgeModel()
        models['myopia'] = MyopiaModel()
        models['glaucoma'] = GlaucomaModel()
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

def preprocess_image(image_file: UploadFile) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    try:
        # Read image
        image_data = image_file.file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC to CHW and normalize
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def encode_mask_to_base64(mask: torch.Tensor) -> str:
    """Convert segmentation mask to base64 encoded PNG"""
    try:
        # Convert tensor to PIL Image
        mask_np = mask.cpu().numpy()
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
        
        # Convert to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        
        return encoded
        
    except Exception as e:
        logger.error(f"Error encoding mask: {e}")
        return ""

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    """
    Predict using the specified model
    
    - **file**: Uploaded retinal fundus image
    - **model_type**: One of 'dr', 'vessel', 'age', 'myopia', 'glaucoma'
    """
    
    # Validate model type
    valid_models = ['dr', 'vessel', 'age', 'myopia', 'glaucoma']
    if model_type not in valid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_type. Must be one of: {valid_models}"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Get the model
        model = models.get(model_type)
        if not model:
            raise HTTPException(
                status_code=500,
                detail=f"Model {model_type} not loaded"
            )
        
        # Preprocess image
        image_tensor = preprocess_image(file)
        
        # Run inference
        result = model.predict(image_tensor)
        
        # Prepare response
        response = {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "model_type": model_type
        }
        
        # Add mask image for segmentation models
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