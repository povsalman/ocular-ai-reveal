# AI Healthcare Backend

This is the FastAPI backend for the AI Healthcare application, providing endpoints for various retinal image analysis models.

## Features

- **Vessel Segmentation**: Multi-dataset R2U-Net model for retinal blood vessel segmentation
- **DR Classification**: Diabetic retinopathy classification
- **Glaucoma Detection**: Glaucoma detection model
- **Age Prediction**: Retinal age prediction
- **Myopia Detection**: Myopia detection model

## Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Create and activate virtual environment**:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Ensure the model files are in the correct location**:
   - `python/Vessel Segmentation/models/` contains 12 model files:
     - `r2unet_DRIVE_checkpoint_dice.keras` (preferred)
     - `r2unet_DRIVE_checkpoint.weights.h5` (fallback)
     - `r2unet_DRIVE_final.keras` (fallback)
     - Similar files for CHASEDB1, HRF, and STARE datasets

### Running the Backend

1. **Start the FastAPI server**:

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

2. The server will start on `http://localhost:8000`

3. API documentation will be available at `http://localhost:8000/docs`

## API Endpoints

### POST /predict/

Predict using the specified model.

**Parameters:**

- `file`: Uploaded retinal fundus image
- `model_type`: One of 'dr', 'vessel', 'age', 'myopia', 'glaucoma'

**Response:**

```json
{
  "status": "success",
  "predicted_class": "Vessels Detected",
  "confidence": 0.85,
  "model_type": "vessel",
  "dataset_used": "DRIVE",
  "metrics": {
    "dice_coefficient": 0.77,
    "sensitivity": 0.77,
    "specificity": 0.98,
    "f1_score": 0.76,
    "accuracy": 0.95,
    "jaccard_similarity": 0.62,
    "auc": 0.88
  },
  "mask_image": "base64_encoded_png_image"
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": ["dr", "vessel", "age", "myopia", "glaucoma"]
}
```

### GET /

Root endpoint with API information.

**Response:**

```json
{
  "message": "AI Healthcare Backend API",
  "version": "1.0.0",
  "available_models": ["dr", "vessel", "age", "myopia", "glaucoma"],
  "docs": "/docs"
}
```

## Testing

Run the test script to verify the vessel segmentation model:

```bash
python test_vessel.py
```

## Model Architecture

### Vessel Segmentation (Multi-Dataset R2U-Net)

The vessel segmentation system uses multiple R2U-Net models trained on different datasets:

- **Architecture**: R2U-Net with t=3 (1 forward convolution + 3 recurrent convolutions)
- **Input**: 48x48 patches from retinal fundus images
- **Output**: Binary segmentation mask
- **Training**: Trained on DRIVE, CHASEDB1, HRF, and STARE datasets
- **Preprocessing**: Green channel extraction with CLAHE enhancement
- **Selection**: Best prediction selected based on Dice Coefficient (DC), with Sensitivity (SE) and Specificity (SP) as tie-breakers

### Model Files

The system expects the following files in `python/Vessel Segmentation/models/`:

**DRIVE Dataset:**

1. `r2unet_DRIVE_checkpoint_dice.keras` (preferred) - Best model based on Dice coefficient
2. `r2unet_DRIVE_checkpoint.weights.h5` (fallback) - Weights-only checkpoint
3. `r2unet_DRIVE_final.keras` (fallback) - Final training model

**CHASEDB1 Dataset:**

1. `r2unet_CHASEDB1_checkpoint_dice.keras` (preferred)
2. `r2unet_CHASEDB1_checkpoint.weights.h5` (fallback)
3. `r2unet_CHASEDB1_final.keras` (fallback)

**HRF Dataset:**

1. `r2unet_HRF_checkpoint_dice.keras` (preferred)
2. `r2unet_HRF_checkpoint.weights.h5` (fallback)
3. `r2unet_HRF_final.keras` (fallback)

**STARE Dataset:**

1. `r2unet_STARE_checkpoint_dice.keras` (preferred)
2. `r2unet_STARE_checkpoint.weights.h5` (fallback)
3. `r2unet_STARE_final.keras` (fallback)

## CORS Configuration

The backend is configured to allow CORS from any origin for development. In production, you should specify your frontend URL:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

The backend includes comprehensive error handling with status codes:

- **success**: Prediction completed successfully
- **model_not_loaded**: One or more models failed to load
- **prediction_failed**: Model inference failed
- **invalid_input**: Invalid file or parameters
- **no_models_available**: No models are available for inference

All errors return appropriate HTTP status codes and descriptive error messages.

## Development

### Adding New Models

1. Create a new model class in `models/`
2. Implement the `predict()` method
3. Add the model to the startup loading in `main.py`
4. Update the API documentation

### Testing

- Use the provided test scripts
- Test with various image formats and sizes
- Verify mask encoding works correctly
- Check CORS configuration with frontend

## Troubleshooting

### Common Issues

1. **Model not loading**: Check that the model files exist in the correct location
2. **Import errors**: Ensure all dependencies are installed in the virtual environment
3. **Memory issues**: The models are loaded on startup and kept in memory
4. **CORS errors**: Check the CORS configuration for your frontend URL
5. **Virtual environment not activated**: Make sure to activate the virtual environment before running

### Logs

The backend uses Python logging. Check the console output for detailed error messages and model loading status.

### Performance

- Models are loaded once at startup for optimal performance
- Inference runs on CPU (can be modified for GPU acceleration)
- Multi-model prediction may take longer but provides better results
