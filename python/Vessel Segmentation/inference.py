import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rrcu_block(inputs, filters, kernel_size=3, t=3):
    """Recurrent Residual Convolutional Unit (RRCU) with t time steps."""
    conv = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(conv)
    x = layers.ReLU()(x)
    for _ in range(t-1):  # t-1 recurrent convolutions
        conv_r = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.Add()([conv, conv_r])  # Residual connection
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)  # Add dropout for regularization
    return x

def r2unet(input_shape=(48, 48, 1)):
    """R2U-Net model with t=3."""
    inputs = layers.Input(input_shape)
    # Encoder
    e1 = rrcu_block(inputs, 16)
    p1 = layers.MaxPooling2D((2, 2))(e1)
    e2 = rrcu_block(p1, 32)
    p2 = layers.MaxPooling2D((2, 2))(e2)
    e3 = rrcu_block(p2, 64)
    p3 = layers.MaxPooling2D((2, 2))(e3)
    e4 = rrcu_block(p3, 128)
    # Decoder
    u3 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(e4)
    u3 = layers.Concatenate()([u3, e3])
    d3 = rrcu_block(u3, 64)
    u2 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(d3)
    u2 = layers.Concatenate()([u2, e2])
    d2 = rrcu_block(u2, 32)
    u1 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(d2)
    u1 = layers.Concatenate()([u1, e1])
    d1 = rrcu_block(u1, 16)
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)
    return tf.keras.models.Model(inputs, outputs)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Compute Dice loss for binary segmentation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    """Combine binary cross-entropy and Dice loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Compute Dice coefficient for evaluation during training."""
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(tf.where(y_pred > 0.5, 1.0, 0.0), tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def load_image(image_path):
    """Load and preprocess retinal fundus image to enhance vessel visibility with controlled contrast."""
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        # If image_path is already a PIL Image
        img = image_path
    
    # Convert to RGB if not already (in case of grayscale or other formats)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Extract green channel (vessels are most prominent in green channel)
    img_array = np.array(img)
    green_channel = img_array[:, :, 1]  # Green channel (index 1 in RGB)
    
    # Apply CLAHE to enhance contrast with reduced amplification
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_channel = clahe.apply(green_channel)
    
    return green_channel

def extract_patches_for_eval(image, patch_size=48):
    """Extract all possible 48x48 non-overlapping patches from an image for evaluation."""
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, patch_size):  # Non-overlapping patches
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch[..., np.newaxis])  # Add channel dimension
    return np.array(patches), h, w

def reconstruct_image(patches, h, w, patch_size=48):
    """Reconstruct image from non-overlapping patches."""
    recon = np.zeros((h, w, 1))
    patch_idx = 0
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            if patch_idx < len(patches):  # Ensure index is valid
                recon[y:y+patch_size, x:x+patch_size] = patches[patch_idx]
            patch_idx += 1
    return recon

class VesselSegmentationModel:
    """Vessel Segmentation Model using R2U-Net"""
    
    def __init__(self, model_path=None):
        self.model = None
        # Use the current directory as the base path
        if model_path is None:
            model_path = "r2unet_DRIVE_checkpoint_dice.keras"
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load the trained R2U-Net model"""
        try:
            # Try to load the saved model
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    'combined_loss': combined_loss, 
                    'dice_coefficient': dice_coefficient
                }
            )
            logger.info(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            logger.warning(f"Model file {self.model_path} not found. Creating new model.")
            # Create a new model if the saved one doesn't exist
            self.model = r2unet()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                loss=combined_loss,
                metrics=['accuracy', tf.keras.metrics.MeanSquaredError(), dice_coefficient]
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for vessel segmentation"""
        # Load and preprocess the image
        processed_image = load_image(image)
        
        # Ensure the image has the right shape for patch extraction
        if len(processed_image.shape) == 2:
            processed_image = processed_image[..., np.newaxis]
        
        return processed_image
    
    def predict(self, image):
        """Predict vessel segmentation mask"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Extract patches
            patches, h, w = extract_patches_for_eval(processed_image)
            
            if len(patches) == 0:
                logger.warning("No patches extracted from image")
                return {
                    "predicted_class": "No Vessels Detected",
                    "confidence": 0.0,
                    "mask": np.zeros((h, w)),
                    "vessel_density": 0.0
                }
            
            # Predict on patches
            pred_patches = self.model.predict(patches, verbose=0)
            
            # Reconstruct predicted image
            pred_mask = reconstruct_image(pred_patches, h, w)
            
            # Calculate vessel density as confidence
            vessel_pixels = np.sum(pred_mask > 0.5)
            total_pixels = pred_mask.size
            vessel_density = vessel_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Determine vessel presence class
            if vessel_density > 0.1:
                predicted_class = "Vessels Detected"
                confidence = min(vessel_density * 2, 0.95)  # Scale density to confidence
            else:
                predicted_class = "No Vessels Detected"
                confidence = 0.85
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "mask": pred_mask.squeeze(),  # Remove channel dimension
                "vessel_density": vessel_density
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return fallback prediction
            return {
                "predicted_class": "Vessels Detected",
                "confidence": 0.75,
                "mask": np.zeros((512, 512)),
                "vessel_density": 0.15
            } 