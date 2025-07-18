import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any

def extract_roi(img_path, mask):
    """
    Extracts the region of interest from the image, masks the background to black,
    and returns a tight crop around the cup and disc.
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    W, H = img.size
    mask_h, mask_w = mask.shape

    # Mask for cup and disc
    roi_mask = (mask == 1) | (mask == 2)
    if not np.any(roi_mask):
        # No cup/disc detected: return resized original image
        return img

    # Bounding box
    y_min = np.min(np.where(roi_mask)[0])
    y_max = np.max(np.where(roi_mask)[0])
    x_min = np.min(np.where(roi_mask)[1])
    x_max = np.max(np.where(roi_mask)[1])

    # Rescale coordinates to original image
    x_min = int(x_min / mask_w * W)
    x_max = int(x_max / mask_w * W)
    y_min = int(y_min / mask_h * H)
    y_max = int(y_max / mask_h * H)

    # Add margin
    margin = 40
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, W)
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, H)

    # Create a resized binary mask for original image size
    mask_pil = Image.fromarray((roi_mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    mask_resized = np.array(mask_pil) > 0

    # Mask the image: set background to black
    img_np[~mask_resized] = 0

    # Crop the masked image
    roi_np = img_np[y_min:y_max, x_min:x_max]

    # Convert back to PIL Image
    roi_pil = Image.fromarray(roi_np)

    return roi_pil

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

def build_unet():
    m = nn.ModuleDict({
        "enc1": double_conv(3, 64),
        "enc2": double_conv(64, 128),
        "enc3": double_conv(128, 256),
        "enc4": double_conv(256, 512),
        "bottleneck": double_conv(512, 1024),
        "dec4": double_conv(1024 + 512, 512),
        "dec3": double_conv(512 + 256, 256),
        "dec2": double_conv(256 + 128, 128),
        "dec1": double_conv(128 + 64, 64),
        "final": nn.Conv2d(64, 3, 1),
        "pool": nn.MaxPool2d(2),
        "up": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    })
    return m

def unet_forward(m, x):
    e1 = m["enc1"](x)
    e2 = m["enc2"](m["pool"](e1))
    e3 = m["enc3"](m["pool"](e2))
    e4 = m["enc4"](m["pool"](e3))
    b = m["bottleneck"](m["pool"](e4))
    d4 = m["dec4"](torch.cat([m["up"](b), e4], dim=1))
    d3 = m["dec3"](torch.cat([m["up"](d4), e3], dim=1))
    d2 = m["dec2"](torch.cat([m["up"](d3), e2], dim=1))
    d1 = m["dec1"](torch.cat([m["up"](d2), e1], dim=1))
    out = m["final"](d1)
    return out

# DenseNet-201 feature extractor
densenet = models.densenet201(pretrained=True)
feature_extractor = densenet.features
feature_extractor.eval()  # Important for inference

# Global pooling
global_pool = nn.AdaptiveAvgPool2d((1, 1))

# Compute flattened feature size
with torch.no_grad():
    dummy = torch.zeros(1, 3, 256, 256)
    out = feature_extractor(dummy)
    out = global_pool(out)
    feat_dim = out.view(1, -1).shape[1]
print("Feature dimension:", feat_dim)

# Classifier head (must exactly match training)
classifier_head = nn.Sequential(
    nn.Linear(feat_dim + 1, 1024),  # Note +1 for CDR
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1)
)

# Load saved weights
checkpoint = torch.load("models/glaucoma_models/best_model1.pth", map_location="cpu")
feature_extractor.load_state_dict(checkpoint["feature_extractor"])
classifier_head.load_state_dict(checkpoint["classifier_head"])

# Set to eval mode
feature_extractor.eval()
classifier_head.eval()

# Wrapper
class GlaucomaDualModel:
    """
    Wrapper to load segmentation model (U-Net) + DenseNet201 classification model with custom head.
    """
    def __init__(self, segmentation_checkpoint: str, classification_checkpoint: str, device: str = "cpu"):
        self.device = torch.device(device)

        # Build your U-Net
        self.segmentation_model = build_unet().to(self.device)
        self.segmentation_model.load_state_dict(torch.load(segmentation_checkpoint, map_location=self.device))
        self.segmentation_model.eval()

        # DenseNet feature extractor
        densenet = models.densenet201(pretrained=True)
        self.feature_extractor = densenet.features.eval().to(self.device)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        # Compute feat_dim
        with torch.no_grad():
            dummy = torch.zeros(1,3,256,256).to(self.device)
            out = self.feature_extractor(dummy)
            out = self.global_pool(out)
            feat_dim = out.view(1,-1).shape[1]

        # Classifier head
        self.classifier_head = nn.Sequential(
            nn.Linear(feat_dim + 1, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        ).to(self.device)

        # Load classifier head weights
        checkpoint = torch.load(classification_checkpoint, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.classifier_head.load_state_dict(checkpoint["classifier_head"])

        # Set to eval mode
        self.feature_extractor.eval()
        self.classifier_head.eval()

        self.classes = ["No Glaucoma", "Glaucoma Detected"]

        # Transforms
        IMG_SIZE = (256,256)
        self.transform_img = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])
        self.transform_roi_val = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def preprocess_segmentation(self, image_tensor):
        image_np = image_tensor.permute(1,2,0).numpy()
        pil = Image.fromarray((image_np*255).astype(np.uint8))
        processed = self.transform_img(pil)
        return processed.unsqueeze(0).to(self.device)

    def preprocess_classification(self, image_tensor):
        image_np = image_tensor.permute(1,2,0).numpy()
        pil = Image.fromarray((image_np*255).astype(np.uint8))
        processed = self.transform_roi_val(pil)
        return processed.unsqueeze(0).to(self.device)

    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Predicts glaucoma from an input tensor by:
        - Computing segmentation mask
        - Computing CDR
        - Extracting ROI
        - Running DenseNet classifier
        """
        try:
            # 1️⃣ Convert input tensor to PIL
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))

            # 2️⃣ Preprocess for segmentation
            seg_input = self.transform_img(pil_image).unsqueeze(0).to(self.device)

            # 3️⃣ Run U-Net segmentation
            with torch.no_grad():
                logits = unet_forward(self.segmentation_model, seg_input)
            label_mask = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy()

            # Debug
            unique_labels = np.unique(label_mask)
            print("Unique labels in mask:", unique_labels)

            # 4️⃣ Compute CDR
            cup_area = np.sum(label_mask == 2)
            disc_area = np.sum(label_mask == 1)
            cdr_value = cup_area / disc_area if disc_area > 0 else 0.0
            print(f"CDR value computed: {cdr_value:.4f}")

            # 5️⃣ Save the input tensor to a temporary image for extract_roi
            # (because your extract_roi expects a path)
            import tempfile
            tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_image.save(tmp_file.name)

            # 6️⃣ Extract ROI
            roi_pil = extract_roi(tmp_file.name, label_mask)

            # 7️⃣ Transform ROI for DenseNet
            clf_input = self.transform_roi_val(roi_pil).unsqueeze(0).to(self.device)

            # 8️⃣ Classify
            with torch.no_grad():
                features = self.feature_extractor(clf_input)
                features = self.global_pool(features)
                features = features.view(features.size(0), -1)

                cdr_tensor = torch.tensor([[cdr_value]], dtype=torch.float32).to(self.device)
                combined = torch.cat([features, cdr_tensor], dim=1)

                logits = self.classifier_head(combined)
                prob = torch.sigmoid(logits).item()

            predicted_idx = 1 if prob > 0.5 else 0

            return {
                "predicted_class": self.classes[predicted_idx],
                "confidence": prob,
                "cdr": cdr_value,
                "mask": label_mask,
                "all_probabilities": [1 - prob, prob]
            }

        except Exception as e:
            print("Prediction error:", e)
            return {
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "cdr": 0.0,
                "mask": torch.zeros((256, 256), dtype=torch.int64),
                "all_probabilities": [0.0, 0.0]
            }

    