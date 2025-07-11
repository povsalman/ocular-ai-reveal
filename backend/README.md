# AI Healthcare Backend

A FastAPI-based backend for AI-powered retinal image analysis with 5 different deep learning models.

## 🚀 Features

- **5 AI Models**: DR Classification, Vessel Segmentation, Age Prediction, Myopia Detection, and Glaucoma Detection
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **CORS Support**: Full CORS enabled for frontend integration
- **Image Processing**: Support for multipart/form-data image uploads
- **Segmentation Masks**: Base64-encoded PNG masks for vessel and glaucoma models
- **Swagger Docs**: Automatic API documentation at `/docs`

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── models/                 # AI model implementations
│   ├── dr_model.py        # Diabetic Retinopathy Classification
│   ├── vessel_model.py    # Vessel Segmentation
│   ├── age_model.py       # Age Prediction
│   ├── myopia_model.py    # Myopia Detection
│   └── glaucoma_model.py  # Glaucoma Detection & Segmentation
├── utils/                  # Utility functions
├── saved_models/          # Pre-trained model weights (not included)
└── README.md              # This file
```

## 🛠 Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Server

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔌 API Endpoints

### POST `/predict/`
Main prediction endpoint for all models.

**Parameters**:
- `file`: Image file (multipart/form-data)
- `model_type`: String - one of `dr`, `vessel`, `age`, `myopia`, `glaucoma`

**Response**:
```json
{
  "predicted_class": "string",
  "confidence": 0.95,
  "model_type": "dr",
  "mask_image": "base64_string"  // Only for vessel and glaucoma models
}
```

### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": ["dr", "vessel", "age", "myopia", "glaucoma"]
}
```

### GET `/`
Root endpoint with API information.

## 🤖 AI Models

### 1. DR Classification (`dr`)
- **Purpose**: Detect and classify diabetic retinopathy stages
- **Classes**: No DR, Mild NPDR, Moderate NPDR, Severe NPDR, Proliferative DR
- **Architecture**: Custom CNN with 5 output classes

### 2. Vessel Segmentation (`vessel`)
- **Purpose**: Segment blood vessels in retinal images
- **Output**: Binary segmentation mask + vessel density
- **Architecture**: U-Net with skip connections

### 3. Age Prediction (`age`)
- **Purpose**: Predict age group from retinal images
- **Classes**: 20-30, 31-40, 41-50, 51-60, 60+ years
- **Architecture**: CNN with age group classification

### 4. Myopia Detection (`myopia`)
- **Purpose**: Detect myopia presence and severity
- **Classes**: No Myopia, Myopia Detected
- **Architecture**: CNN with severity assessment

### 5. Glaucoma Detection (`glaucoma`)
- **Purpose**: Detect glaucoma and segment optic disc
- **Classes**: No Glaucoma, Glaucoma Detected
- **Architecture**: U-Net with classification head

## 🔧 Model Details

All models:
- Run on CPU (can be modified for GPU)
- Use PyTorch framework
- Include fallback predictions for error handling
- Support standard image formats (PNG, JPG, JPEG)
- Normalize images using ImageNet statistics

## 🌐 CORS Configuration

The backend is configured with full CORS support:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🔒 Security Notes

- CORS is set to allow all origins (`*`) for development
- In production, specify your frontend URL
- No authentication implemented (add as needed)
- Input validation for file types and model types

## 🐛 Error Handling

The API includes comprehensive error handling:
- Invalid model types return 400 error
- Invalid file types return 400 error
- Model loading errors return 500 error
- Fallback predictions for model inference errors

## 📊 Performance

- Models are loaded once at startup
- Inference runs on CPU (modify for GPU acceleration)
- Image preprocessing optimized for each model
- Base64 encoding for segmentation masks

## 🔄 Integration with Frontend

The backend is designed to work with the React frontend:
- Accepts multipart/form-data uploads
- Returns JSON responses
- Includes CORS headers
- Provides segmentation masks as base64 strings

## 🚀 Deployment

For production deployment:
1. Set up proper CORS origins
2. Add authentication if needed
3. Use a production WSGI server (Gunicorn)
4. Configure proper logging
5. Add model weight files to `saved_models/`

## 📝 License

This project is part of the AI Healthcare application. 