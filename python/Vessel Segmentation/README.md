# Vessel Segmentation Module - Backend Documentation

## Overview

This module provides multi-dataset retinal vessel segmentation using R2U-Net models, integrated into the FastAPI backend. It supports inference on DRIVE, CHASEDB1, HRF, and STARE datasets, and returns static, pre-evaluated metrics for each dataset.

---

## File Structure

```
python/Vessel Segmentation/
├── models/           # Trained model files (.keras, .h5)
├── inference.py      # Model inference utilities (VesselSegmentationModel class)
├── results.txt       # Static evaluation metrics for each dataset
└── README.md         # This documentation
```

---

## Backend Flow

1. **API Route**: The FastAPI backend exposes a `/predict/` endpoint (see `backend/main.py`).
2. **Model Loading**: On startup, all available vessel segmentation models are loaded from the `models/` directory, preferring `*_dice.keras` files.
3. **Image Upload**: The user uploads a retinal fundus image via the frontend.
4. **Inference**: The backend runs the image through all loaded models (one per dataset).
5. **Best Model Selection**: The model with the **highest confidence score** (lowest average entropy) is selected as the best. Confidence is computed as:
   - For each model, the sigmoid output (probability map) is used to compute per-pixel entropy:
     ```python
     entropy = - (p * log2(p + eps) + (1 - p) * log2(1 - p + eps))
     avg_entropy = np.mean(entropy)
     confidence = 1 - avg_entropy  # log2(2) = 1
     ```
   - The model with the highest confidence is selected.
6. **Static Metrics**: Instead of computing metrics dynamically, the backend reads the pre-saved metrics for the selected dataset from `results.txt` and returns them in the API response.
7. **Result Sending**: The API response includes:
   - The predicted mask (base64 PNG)
   - The dataset used
   - The static metrics (Dice, Sensitivity, Specificity, F1, Accuracy, Jaccard, AUC)
   - Status and confidence

---

## results.txt Structure

- Each line contains the dataset name and a Python dictionary of metrics:

```
DRIVE Metrics: {'AC': 0.966..., 'SE': 0.775..., 'SP': 0.984..., 'F1': 0.801..., 'DC': 0.801..., 'JS': 0.669..., 'AUC': 0.972...}
STARE Metrics: {...}
CHASEDB1 Metrics: {...}
HRF Metrics: {...}
```

- **Key Mapping:**
  - `AC` → `accuracy`
  - `SE` → `sensitivity`
  - `SP` → `specificity`
  - `F1` → `f1_score`
  - `DC` → `dice_coefficient`
  - `JS` → `jaccard_similarity`
  - `AUC` → `auc`

---

## Dataset-Specific Preprocessing

Before inference, each uploaded image is preprocessed for each dataset as follows:

- **Green Channel Extraction**: Only the green channel is used, as vessels are most visible there.
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization is applied with `clipLimit=2.0` and `tileGridSize=(8, 8)`.
- **DRIVE**: The image is cropped to `img[:, 9:574]`.
- **STARE, CHASEDB1, HRF**: The full image is used.
- **Resizing**: All images are resized to 512x512 for model input.

---

## Involved Files

- `inference.py`: Contains the `VesselSegmentationModel` class for model loading and prediction.
- `results.txt`: Stores static metrics for each dataset.
- `backend/models/vessel_model.py`: Orchestrates model selection (entropy-based), mask generation, and static metric retrieval.
- `backend/main.py`: FastAPI app and `/predict/` route.

---

## Example API Response

```
{
  "status": "success",
  "predicted_class": "Vessels Detected",
  "confidence": 0.85,
  "model_type": "vessel",
  "dataset_used": "DRIVE",
  "metrics": {
    "dice_coefficient": 0.8015,
    "sensitivity": 0.7750,
    "specificity": 0.9849,
    "f1_score": 0.8015,
    "accuracy": 0.9660,
    "jaccard_similarity": 0.6692,
    "auc": 0.9724
  },
  "mask_image": "base64_encoded_png_image"
}
```

---

## Notes

- The backend now uses entropy-based confidence for model selection, not Dice or pseudo-ground truth.
- All metrics are static and dataset-specific, loaded from `results.txt`.
- If a model or metrics are missing, the backend will log an error and return a fallback response.
- For more details, see the main project README and backend/README.md.
