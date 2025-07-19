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

### 1. Clo0ne the Repository

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

## Diabetic Retinopathy (DR) Classification

The **DR Classification** module uses two state-of-the-art deep learning models — **DenseNet** and a **Vision Transformer (ViT)** — to classify the severity of diabetic retinopathy from retinal fundus images.

### Folder Structure & Setup

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

> The folder must be named `dr_models` exactly and placed under `backend/models/`.

---

### How It Works

- The system takes a **retinal fundus image** as input.
- Both models make predictions independently.
- The prediction with the **higher confidence score** is selected as the final result.
- A **Grad-CAM heatmap** is generated to visually explain which regions of the retina influenced the prediction.

---

### DR Classification Stages

| Stage                | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| **No DR**            | No signs of diabetic retinopathy were detected.                                     |
| **Mild NPDR**        | Microaneurysms are present. Regular monitoring is recommended.                      |
| **Moderate NPDR**    | Blood vessel damage is visible. Closer monitoring and treatment may be needed.      |
| **Severe NPDR**      | Extensive damage and blocked vessels are present. Urgent care may be required.      |
| **Proliferative DR** | Abnormal blood vessel growth is observed. Immediate medical attention is necessary. |

---

### Grad-CAM Explanation

The Grad-CAM visualization highlights the regions of the retinal image that contributed most to the model's decision:

- 🔴 **Bright red/yellow regions**: High attention areas
- 🔵 **Cooler or dark regions**: Low attention areas

This helps users and practitioners understand why a certain prediction was made.

---

### Example Workflow

1. Navigate to `http://localhost:3000`
2. Click on **"DR Classification"**
3. Upload a **retinal fundus image**
4. Click **"Start Analysis"**
5. View:

   - Predicted **DR stage**
   - **Confidence score** of the prediction
   - Model used (DenseNet or ViT)
   - **Grad-CAM heatmap** for model interpretability
   - Textual explanation for the predicted DR stage

---

### 🔍 API Support for DR Classification

#### POST `/predict/`

- `file`: Image file
- `model_type`: `"dr"`

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

### Model Details

| Model                    | Framework        | Type              | Description                                       |
| ------------------------ | ---------------- | ----------------- | ------------------------------------------------- |
| DenseNet                 | TensorFlow/Keras | Convolutional     | Lightweight, high-accuracy CNN for classification |
| Vision Transformer (ViT) | PyTorch          | Transformer-based | Excels in capturing global image context          |

---

### Expected Results

- For clear fundus images, confidence scores ≥ 80% are expected.
- For low-quality or blurry images, confidence may drop and users may be shown a warning.
- Grad-CAM is available for most successful predictions to improve interpretability.

---

# 👁️ Myopia Detection Module

## 📝 Overview

This module provides AI-powered myopia risk assessment and feature analysis from retinal fundus images. It consists of a React frontend and a FastAPI backend, with a machine learning model for myopia detection and feature extraction.

---

## 🚀 How to Run

### 🐍 Backend

1. **Install Python dependencies:**

   ```bash
   cd backend
   python -m venv venv
   # Activate the virtual environment:
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Ensure model files are present:**

   - Place the following files in `python/Myopia Detection/`:
     - `enhanced_myopia_model.pkl`
     - `scaler.pkl`
     - `pca.pkl`
     - `feature_names.pkl`

3. **Run the backend server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   - The API will be available at [http://localhost:8000](http://localhost:8000)
   - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### ⚛️ Frontend

1. **Install Node.js dependencies:**

   ```bash
   cd frontend
   npm install
   ```

2. **Run the frontend:**
   ```bash
   npm run dev
   ```
   - The app will be available at [http://localhost:5173](http://localhost:5173) (or as shown in your terminal)

---

## 📦 Libraries Used

### Backend

- **fastapi**: API framework
- **uvicorn**: ASGI server
- **numpy, opencv-python, scikit-image**: Image processing
- **scikit-learn**: ML preprocessing and PCA
- **tensorflow, torch**: Deep learning (for other modules)
- **pillow**: Image handling

See `backend/requirements.txt` for full list.

### Frontend

- **React**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **shadcn-ui**: UI components
- **recharts**: Data visualization
- **react-markdown**: Markdown rendering
- **react-router-dom**: Routing

See `frontend/package.json` for full list.

---

## 🔌 API

### Endpoint

**POST** `/predict/`

#### Parameters

- `file`: Retinal fundus image (PNG, JPG, JPEG)
- `model_type`: `"myopia"`

#### Example Request (using `curl`)

```bash
curl -X POST -F "file=@your_image.png" -F "model_type=myopia" http://localhost:8000/predict/
```

#### Example Response

```json
{
  "status": "success",
  "predicted_class": "High Myopia Risk",
  "confidence": 0.92,
  "model_type": "myopia",
  "features": {
    "avg_vessel_length": 32.23,
    "vessel_length_std": 15.94,
    "max_vessel_length": 87.2
  }
}
```

---

## 🛠️ Troubleshooting

- **⚪ White screen on frontend:**

  - Ensure all required files (especially `ModuleAnalysis.tsx` or `ModuleAnalysisMyopia.tsx`) exist and are correctly imported.
  - Check the browser console for errors.
  - Restart the frontend after any file changes.

- **🐍 Backend errors:**

  - Check that all model files are present in `python/Myopia Detection/`.
  - Ensure the virtual environment is activated and dependencies are installed.
  - Check the backend terminal for error logs.

- **🌐 CORS errors:**

  - Make sure the backend allows requests from your frontend URL (see CORS settings in `main.py`).

- **🔗 API not responding:**
  - Confirm the backend is running and accessible at the correct port.

---

## 📝 Additional Notes

- The clinical summary and key features are displayed with up to 2 decimal points for clarity.
- For development, both frontend and backend must be running simultaneously.

---

# 🧓 Retinal Age Prediction Module

### 📝 Overview

The **Age Prediction** module uses deep learning to estimate a patient’s age from retinal fundus images. It leverages a pretrained **InceptionResNetV2** architecture, fine-tuned on the [ODIR-5K Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k), enabling non-invasive retinal age estimation.

This module follows the same **FastAPI + React** architecture as the rest of the Ocular AI Reveal platform and integrates cleanly into the shared backend/frontend structure.

---

### 🚀 How to Run

#### 🐍 Backend

1. **Ensure model file is present**:
   Place the trained model in:

   ```
   backend/models/age_models/age_model.pth
   ```

   Available for download at: https://drive.google.com/drive/folders/1w3WJFu4v93rpW5hcrBKfmDq9QukJgcFL?usp=sharing

2. **Install Python dependencies**:

   ```bash
   cd backend
   python -m venv venv

   # Activate the virtual environment:
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Run the backend server**:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

#### ⚛️ Frontend

1. **Install frontend dependencies**:

   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**:

   ```bash
   npm run dev
   ```

   The app will be available at:
   `http://localhost:8000`

---

### 📦 Required Libraries

#### Backend

The following libraries are required for the Age Prediction module (included in `requirements.txt`):

- `torch` (PyTorch)
- `timm` (for InceptionResNetV2 model)
- `Pillow` (image handling)
- `numpy` (array operations)
- `torchvision.transforms` (image preprocessing)
- `fastapi`, `uvicorn` (API development)
- `logging`, `typing` (standard library utilities)

#### Frontend

- `React`, `Tailwind CSS`, `shadcn/ui`
- Part of the existing unified React/Vite frontend

---

### 🔌 API Endpoint

#### POST `/predict/`

**Parameters:**

| Name         | Type     | Description                           |
| ------------ | -------- | ------------------------------------- |
| `file`       | `File`   | Retinal fundus image (PNG, JPG, JPEG) |
| `model_type` | `String` | Must be `"age"`                       |

**Example Request (cURL):**

```bash
curl -X POST -F "file=@your_image.jpg" -F "model_type=age" http://localhost:8000/predict/
```

**Example Response:**

```json
{
  "status": "success",
  "predicted_age": 57.34,
  "confidence": 0.89,
  "model_type": "age"
}
```

---

### 🧠 Model Details

- **Architecture**: InceptionResNetV2 (via `timm`)
- **Input**: 299x299 RGB image
- **Output**: Continuous predicted age (float)
- **Preprocessing**: Resize, normalize, convert to tensor
- **Loss Function**: Mean Squared Error during training
- **Inference**: Optimized for CPU; <5s per image

---

### 💡 Expected Results

| Input Quality      | Expected Output                                   |
| ------------------ | ------------------------------------------------- |
| High-quality image | Accurate age prediction within ±5 years (usually) |
| Low-quality image  | Lower confidence and possible warning             |
| Invalid input      | Returns error with appropriate message            |

---

### 🛠️ Troubleshooting

- **Backend doesn’t predict**:

  - Ensure `age_model.pth` exists at `https://drive.google.com/drive/folders/1w3WJFu4v93rpW5hcrBKfmDq9QukJgcFL?usp=sharing`
  - Verify all dependencies are installed (especially `torch` and `timm`)

- **Frontend blank or broken**:

  - Ensure `ModuleAnalysisAge.tsx` exists and is routed
  - Check console for import errors or missing components

- **CORS issues**:

  - Ensure FastAPI is configured with proper CORS middleware

- **Low confidence predictions**:

  - Recommend using high-resolution, centered fundus images

---

### 🔍 Integration Note

This module plugs into the unified `/predict/` API using `model_type: "age"`, following the same interface and workflow as the other modules.


## Glaucoma Segmentation and Detection

The Glaucoma Detection module combines a **U-Net-based segmentation model** with a **DenseNet201 classifier** to analyze retinal fundus images for signs of glaucoma.

---

### Folder Structure & Setup

To enable glaucoma segmentation and classification:

1. **Download the model files** from the following Google Drive folder:  
📎 **Download Glaucoma Models**

2. **Place the models** in the following directory inside your backend:

```
backend/
└── models/
    └── glaucoma_models/
        ├── best_model1.pth          # DenseNet model for classification
        └── unet_origa2.pth          # Unet segmentation model
```

> The folder must be named `glaucoma_models` exactly and placed under `backend/models/`.

---

### How It Works

- A retinal fundus image is uploaded by the user.
- The **U-Net segmentation model** detects:
  - **Optic Disc** (label 1)
  - **Optic Cup** (label 2)
- From this segmentation, the **Cup-to-Disc Ratio (CDR)** is computed:  

CDR = Cup Area / Disc Area

- The segmented **Region of Interest (ROI)** is extracted.
- The **DenseNet201 classifier** receives the ROI and CDR as input to predict glaucoma presence.
- The final output includes:
- Predicted class
- CDR value
- Segmentation mask

---

### Segmentation Labels

| Region      | Label | Color  |
|-------------|-------|--------|
| Optic Disc  | 1     | 🔴 Red |
| Optic Cup   | 2     | 🟢 Green |

---

### CDR Threshold Interpretation

| CDR Value Range | Interpretation                |
|------------------|------------------------------|
| 0.1 – 0.5        | Normal                       |
| > 0.5            | Clinical review for Glaucoma |

---

### Example Workflow

1. Go to [http://localhost:3000](http://localhost:3000)
2. Click on **“Glaucoma Detection”**
3. Upload a retinal image
4. Click **“Start Analysis”**
5. View:
 - Segmentation mask
 - CDR ratio
 - Final classification result

---

### API Support for Glaucoma Detection

**Endpoint:**

**Payload:**

- `file`: Image file  
- `model_type`: `"glaucoma"`

**Response Format:**

```json
{
  "status": "success",
  "predicted_class": "Glaucoma Detected",
  "cdr": 0.65,
  "mask": "<base64_encoded_mask>"
}
```

### Model Details

| Model        | Framework | Type           | Description                                        |
|--------------|-----------|----------------|----------------------------------------------------|
| U-Net        | PyTorch   | Segmentation   | Identifies optic disc and cup structures in the retina |
| DenseNet201  | PyTorch   | Classification | Receives ROI and CDR for glaucoma prediction       |

---

### Expected Results

- For clear fundus images, it is expected to correctly detect Glaucoma.
- Segmentation mask is available to improve interpretability regarding ROI and CDR.

---

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Note**: This application is for research and educational purposes. Medical decisions should always be made by qualified healthcare professionals.

Read file: frontend/README.md
Read file: backend/README.md
Read file: backend/requirements.txt
Read file: frontend/package.json
Read file: backend/models/myopia_model.py
Here is a sample README file for the **Myopia Detection** module, covering setup, libraries, API, and troubleshooting:

---
