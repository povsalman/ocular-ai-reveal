# backend/utils/preprocessing.py
import cv2
import numpy as np

def apply_clahe_rgb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_img

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (224, 224))
    image = apply_clahe_rgb(image)
    return image
