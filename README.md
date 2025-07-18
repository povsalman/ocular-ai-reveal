# Ocular AI Reveal - Retinal Fundus Analysis Platform!

A comprehensive web application for AI-powered retinal fundus image analysis, featuring multiple deep learning models for various ophthalmological assessments.

## 🧪 Features

### Current Modules

- **✅ Vessel Segmentation** - Multi-dataset R2U-Net for retinal blood vessel segmentation
- **🔄 DR Classification** - Diabetic retinopathy classification (in development)
- **🔄 Glaucoma Detection** - Glaucoma detection and optic disc segmentation (in development)
- **🔄 Age Prediction** - Retinal age prediction from fundus images (in development)
- **🔄 Myopia Detection** - Myopia detection and severity assessment (in development)

### Key Features

- **Multi-Dataset AI Models**: Vessel segmentation uses models trained on DRIVE, CHASEDB1, HRF, and STARE datasets
- **Intelligent Model Selection**: Automatically selects the best performing model based on Dice Coefficient
- **Real-time Analysis**: Upload images and get instant results with detailed metrics
- **Responsive UI**: Modern, medical-themed interface with clean design
- **Comprehensive Metrics**: Display of Dice Coefficient, Sensitivity, Specificity, F1 Score, Accuracy, Jaccard Similarity, and AUC

## 📂 Project Structure

```
ocular-ai-reveal/
├── frontend/           # React + TypeScript + Tailwind CSS + shadcn/ui
│   ├── src/
│   │   ├── components/ # UI components
│   │   ├── pages/      # Application pages
│   │   └── types/      # TypeScript type definitions
│   └── package.json
├── backend/            # FastAPI backend
│   ├── models/         # AI model implementations
│   ├── main.py         # FastAPI application
│   └── requirements.txt
└── python/             # Model files and training code
    └── Vessel Segmentation/
        ├── models/     # Trained model files
        └── inference.py # Model inference utilities
```

## 🛠️ Setup Guide

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ocular-ai-reveal
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## ✅ Testing Vessel Segmentation

1. **Start both servers** (backend and frontend)
2. **Navigate to** `http://localhost:3000`
3. **Click on "Vessel Segmentation"** in the navigation
4. **Upload a retinal fundus image** (PNG, JPG, JPEG)
5. **Click "Start Analysis"**
6. **View results** including:
   - Original image and segmentation mask
   - Detailed metrics (Dice Coefficient, Sensitivity, etc.)
   - Dataset used for prediction
   - Confidence warnings for low-quality predictions

### Expected Results

- **High-quality images**: Dice Coefficient > 0.7, clear vessel segmentation
- **Low-quality images**: Warning message suggesting medical consultation
- **Failed predictions**: Fallback to mock data with error notification

## 🔧 API Endpoints

### POST `/predict/`

Main prediction endpoint for all models.

**Parameters:**

- `file`: Image file (multipart/form-data)
- `model_type`: String - one of `vessel`, `dr`, `glaucoma`, `age`, `myopia`

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

### GET `/health`

Health check endpoint.

### GET `/`

API information and available models.

## 🧠 Model Architecture

### Vessel Segmentation (R2U-Net)

- **Architecture**: R2U-Net with t=3 (1 forward convolution + 3 recurrent convolutions)
- **Input**: 48x48 patches from retinal fundus images
- **Output**: Binary segmentation mask
- **Training**: Trained on DRIVE, CHASEDB1, HRF, and STARE datasets
- **Preprocessing**: Green channel extraction with CLAHE enhancement
- **Selection**: Best prediction selected based on Dice Coefficient (DC), with Sensitivity (SE) and Specificity (SP) as tie-breakers

## 🚀 Deployment

### Backend Deployment

```bash
# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend Deployment

```bash
# Build for production
npm run build

# Serve static files
npm run preview
```

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**

   - Ensure virtual environment is activated
   - Check that all dependencies are installed
   - Verify model files exist in `python/Vessel Segmentation/models/`

2. **Frontend can't connect to backend**

   - Ensure backend is running on `http://localhost:8000`
   - Check CORS configuration
   - Verify network connectivity

3. **Model loading errors**

   - Check that TensorFlow is properly installed
   - Verify model file paths are correct
   - Ensure sufficient memory for model loading

4. **Image upload issues**
   - Check file format (PNG, JPG, JPEG supported)
   - Verify file size (recommended < 10MB)
   - Ensure image is a valid retinal fundus image

### Logs

- **Backend logs**: Check console output for detailed error messages
- **Frontend logs**: Open browser developer tools (F12) for client-side errors

## 📊 Performance

- **Model Loading**: Models loaded once at startup for optimal performance
- **Inference Time**: ~2-5 seconds per image (depending on image size)
- **Memory Usage**: ~2GB RAM for all models loaded
- **CPU Usage**: Optimized for CPU inference (GPU support available)

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow TypeScript best practices
- Use consistent code formatting
- Add proper error handling
- Include comprehensive tests
- Update documentation for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: DRIVE, CHASEDB1, HRF, STARE for vessel segmentation
- **Architecture**: R2U-Net implementation based on research papers
- **UI Framework**: Built with React, TypeScript, and Tailwind CSS
- **Backend**: FastAPI for high-performance API development

---

##  Diabetic Retinopathy (DR) Classification

The **DR Classification** module uses two state-of-the-art deep learning models — **DenseNet** and a **Vision Transformer (ViT)** — to classify the severity of diabetic retinopathy from retinal fundus images.

###  Folder Structure & Setup

Due to GitHub's file size restrictions, model files are not included in the repository.

To enable DR classification:

1. **Download the model files** from the following Google Drive folder:
   📎 **[Download DR Models](https://drive.google.com/drive/folders/1HZ0aZblCraztVov_o9V3EM0rvR8aXCrT?usp=sharing)**

2. **Place the models** in the following directory structure inside your backend:

```
backend/
└── models/
    └── dr_models/
        ├── denseNet.h5              # TensorFlow/Keras DenseNet model
        └── vit.pth                  # PyTorch Vision Transformer model
```

>  The folder must be named `dr_models` exactly and placed under `backend/models/`.

---

###  How It Works

* The system takes a **retinal fundus image** as input.
* Both models make predictions independently.
* The prediction with the **higher confidence score** is selected as the final result.
* A **Grad-CAM heatmap** is generated to visually explain which regions of the retina influenced the prediction.

---

###  DR Classification Stages

| Stage                | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| **No DR**            | No signs of diabetic retinopathy were detected.                                     |
| **Mild NPDR**        | Microaneurysms are present. Regular monitoring is recommended.                      |
| **Moderate NPDR**    | Blood vessel damage is visible. Closer monitoring and treatment may be needed.      |
| **Severe NPDR**      | Extensive damage and blocked vessels are present. Urgent care may be required.      |
| **Proliferative DR** | Abnormal blood vessel growth is observed. Immediate medical attention is necessary. |

---

###  Grad-CAM Explanation

The Grad-CAM visualization highlights the regions of the retinal image that contributed most to the model's decision:

* 🔴 **Bright red/yellow regions**: High attention areas
* 🔵 **Cooler or dark regions**: Low attention areas

This helps users and practitioners understand why a certain prediction was made.

---

###  Example Workflow

1. Navigate to `http://localhost:3000`
2. Click on **"DR Classification"**
3. Upload a **retinal fundus image**
4. Click **"Start Analysis"**
5. View:

   * Predicted **DR stage**
   * **Confidence score** of the prediction
   * Model used (DenseNet or ViT)
   * **Grad-CAM heatmap** for model interpretability
   * Textual explanation for the predicted DR stage

---

### 🔍 API Support for DR Classification

#### POST `/predict/`

* `file`: Image file
* `model_type`: `"dr"`

**Response Format:**

```json
{
  "status": "success",
  "predicted_class": "Moderate NPDR",
  "confidence": 0.91,
  "model_used": "ViT",
  "gradcam_image": "<base64_encoded_image>"
}
```

---

###  Model Details

| Model                    | Framework        | Type              | Description                                       |
| ------------------------ | ---------------- | ----------------- | ------------------------------------------------- |
| DenseNet                 | TensorFlow/Keras | Convolutional     | Lightweight, high-accuracy CNN for classification |
| Vision Transformer (ViT) | PyTorch          | Transformer-based | Excels in capturing global image context          |

---

###  Expected Results

* For clear fundus images, confidence scores ≥ 80% are expected.
* For low-quality or blurry images, confidence may drop and users may be shown a warning.
* Grad-CAM is available for most successful predictions to improve interpretability.

---


## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Note**: This application is for research and educational purposes. Medical decisions should always be made by qualified healthcare professionals.
