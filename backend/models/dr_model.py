# backend/models/dr_model.py

import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16
import torch.nn as nn
from utils.preprocessing_dr import preprocess_image
from pathlib import Path
from utils.gradcam_dr import generate_gradcam
from utils.gradcam_densenet_dr import generate_gradcam_densenet

import cv2
from io import BytesIO
import base64

MODEL_DIR = Path(__file__).resolve().parent / "dr_models"

class DRModel:
    def __init__(self):
        self.device = torch.device("cpu")

        # Load PyTorch ViT model with custom head
        self.torch_model = vit_b_16(weights=None)
        self.torch_model.heads = nn.Linear(in_features=768, out_features=5)
        state_dict = torch.load(MODEL_DIR / "vit.pth", map_location=self.device)
        self.torch_model.load_state_dict(state_dict)
        self.torch_model.to(self.device)
        self.torch_model.eval()

        # Load Keras model
        self.keras_model = tf.keras.models.load_model(MODEL_DIR / "denseNet.h5")

        # Class labels
        self.classes = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']

        # Torch transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def generate_overlay_gradcam(self, gradcam: np.ndarray, preprocessed_rgb: np.ndarray) -> str:
        try:
            heatmap = np.uint8(255 * gradcam)
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Blend with original preprocessed image
            overlay = cv2.addWeighted(preprocessed_rgb, 0.6, heatmap_color, 0.4, 0)

            # Convert to PIL and encode as base64
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(overlay_rgb)
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return b64_str

        except Exception as e:
            print(f"Grad-CAM overlay generation failed: {e}")
            return ""

    def predict(self, image_array: np.ndarray) -> dict:
        # Step 1: Preprocess using CLAHE + Resize (same as training)
        preprocessed = preprocess_image(image_array)  # Output: 224x224 RGB CLAHE-enhanced

        # Step 2: Torch prediction
        torch_tensor = self.transform(Image.fromarray(preprocessed)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            torch_output = self.torch_model(torch_tensor)
            torch_probs = F.softmax(torch_output, dim=1)[0].cpu().numpy()
            torch_pred = int(torch_probs.argmax())
            torch_conf = float(torch_probs[torch_pred])

        # Step 3: Keras prediction
        keras_input = np.expand_dims(preprocessed.astype("float32") / 255.0, axis=0)
        keras_output = self.keras_model.predict(keras_input)[0]
        keras_pred = int(keras_output.argmax())
        keras_conf = float(keras_output[keras_pred])

        # Step 4: Grad-CAM (on torch model using preprocessed image)

        # Step 5: Pick better model
        # if keras_conf > torch_conf:
        #     final_pred = keras_pred
        #     final_conf = keras_conf
        #     model_used = "Keras DenseNet"

            
        #     # Generate Grad-CAM from DenseNet
        #     try:
        #         heatmap = generate_gradcam_densenet(
        #             self.keras_model,
        #             tf.convert_to_tensor(keras_input),
        #             keras_pred
        #         )
        #         gradcam_image_b64 = self.generate_overlay_gradcam(heatmap, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
        #     except Exception as e:
        #         gradcam_image_b64 = ""
        #         print(f"[Warning] DenseNet Grad-CAM failed: {e}")
            
        # else:
        final_pred = torch_pred
        final_conf = torch_conf
        model_used = "Vision Transformer"

        try:
            gradcam_map = generate_gradcam(self.torch_model, torch_tensor, torch_pred)
            gradcam_image_b64 = self.generate_overlay_gradcam(gradcam_map, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
        except Exception as e:
            gradcam_image_b64 = ""
            print(f"[Warning] Grad-CAM failed: {e}")
                
        return {
            "status": "success",
            "predicted_class": self.classes[final_pred],
            "confidence": round(final_conf, 4),
            "model_used": model_used,
            "all_probabilities": {
                #"keras": keras_output.tolist(),
                "torch": torch_probs.tolist()
            },
            "gradcam_image": gradcam_image_b64
        }
