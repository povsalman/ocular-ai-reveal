

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import filters, measure, morphology, segmentation
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from scipy import ndimage
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings
import os
import joblib

# Enhanced imports for improvements
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    precision_recall_curve, roc_curve, auc, roc_auc_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

class EnhancedMyopiaFeatureExtractor:
    def __init__(self):
        self.features = []
        
    def preprocess_image(self, image_path):
        """Load and preprocess the fundus image with enhanced preprocessing"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply circular mask to focus on fundus area
        height, width = img_rgb.shape[:2]
        center = (width // 2, height // 2)
        radius = min(center[0], center[1]) - 20
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply mask
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        return masked_img, mask
    
    def frangi_filter(self, image, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15):
        """Apply Frangi filter for better vessel detection with version compatibility"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        try:
            # Try new parameter names (scikit-image >= 0.16)
            filtered = filters.frangi(gray, scale_range=scale_range, scale_step=scale_step, 
                                beta1=beta1, beta2=beta2)
        except TypeError:
            # Fallback to old parameter names (scikit-image < 0.16)
            filtered = filters.frangi(gray, scale_range=scale_range, scale_step=scale_step, 
                                beta=beta1, gamma=beta2)
        
        # Normalize to 0-255 range
        filtered = ((filtered - filtered.min()) / (filtered.max() - filtered.min()) * 255).astype(np.uint8)
        
        return filtered
    
    def extract_vessel_features(self, image, mask):
        """Fixed vessel feature extraction with proper width calculation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
        # Enhanced vessel detection using Frangi filter
        frangi_filtered = self.frangi_filter(image)
    
        # Extract green channel (vessels are most visible here)
        green_channel = image[:, :, 1]
    
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
    
        # Combine traditional and Frangi-based vessel detection
        combined = cv2.addWeighted(blurred, 0.5, frangi_filtered, 0.5, 0)  # More balanced weighting
    
        # Vessel segmentation using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(combined, cv2.MORPH_TOPHAT, kernel)
    
        # FIXED: Better thresholding with adaptive method
        vessel_mask = cv2.adaptiveThreshold(
            tophat, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15, 2  # Increased block size and constant
        )
    
        # FIXED: Clean up vessel mask
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel)
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel)
    
        # FIXED: Only process if we have significant vessels
        if np.count_nonzero(vessel_mask) < 100:  # Skip if too few vessel pixels
            return {
                'avg_vessel_width': 0.0,
                'vessel_width_std': 0.0,
                'max_vessel_width': 0.0,
                'min_vessel_width': 0.0,
                'branch_density': 0.0
            }
    
        # === VESSEL WIDTH CALCULATION ===
        # Use label to identify connected components
        labeled, num_features = ndimage.label(vessel_mask)
        vessel_widths = []
    
        # Process each vessel component separately
        for i in range(1, num_features + 1):
            component = (labeled == i)
        
            # Skip small components
            if np.sum(component) < 10:
                continue
            
            # Compute distance transform for this component
            dist_transform = ndimage.distance_transform_edt(component)
            skeleton = morphology.skeletonize(component)
        
            if np.any(skeleton):
                # Get widths at skeleton points
                skeleton_coords = np.argwhere(skeleton)
                widths = 2 * dist_transform[tuple(skeleton_coords.T)]
                vessel_widths.extend(widths.tolist())
    
        # Calculate width statistics
        if vessel_widths:
            avg_vessel_width = np.mean(vessel_widths)
            vessel_width_std = np.std(vessel_widths)
            max_vessel_width = np.max(vessel_widths)
            min_vessel_width = np.min(vessel_widths)
        else:
            avg_vessel_width = vessel_width_std = max_vessel_width = min_vessel_width = 0.0
    
        # === BRANCH POINT DETECTION ===
        if np.any(vessel_mask):
            # Use cleaned mask for better skeletonization
            skeleton = morphology.skeletonize(vessel_mask > 0)
            branch_points = self.detect_branch_points(skeleton)
        else:
            branch_points = []
    
        total_area = np.sum(mask > 0)
        branch_density = len(branch_points) / total_area if total_area > 0 else 0.0
    
        return {
            'avg_vessel_width': avg_vessel_width,
            'vessel_width_std': vessel_width_std,
            'max_vessel_width': max_vessel_width,
            'min_vessel_width': min_vessel_width,
            'branch_density': branch_density
        }
     
    def detect_branch_points(self, skeleton):
        """Fixed branch point detection"""
        # Create kernel for branch point detection
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
        # Convert to uint8 for OpenCV operations
        skeleton_uint8 = skeleton.astype(np.uint8) * 255
        
        # Count neighbors
        neighbor_count = cv2.filter2D(skeleton_uint8, -1, kernel) // 255  # Integer division
        
        # Find branch points (3 or more connections)
        branch_points = np.where((skeleton_uint8 > 0) & (neighbor_count >= 3))
        return list(zip(branch_points[0], branch_points[1]))
    
    def extract_cup_to_disc_ratio(self, disc_contour, image):
        """Extract cup-to-disc ratio - key feature for glaucoma/myopia detection"""
        if disc_contour is None or len(disc_contour) < 5:
            return 0
        
        # Extract ROI around disc
        x, y, w, h = cv2.boundingRect(disc_contour)
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        roi = image[y_start:y_end, x_start:x_end]
        
        # Segment cup within disc
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to segment darker regions (cup)
        cup_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 15, 2)
        
        # Calculate areas
        cup_area = np.sum(cup_mask > 0)
        disc_area = cv2.contourArea(disc_contour)
        
        # Calculate cup-to-disc ratio
        cdr = cup_area / disc_area if disc_area > 0 else 0
        
        return min(cdr, 1.0)  # Cap at 1.0 for realistic values
    
    def extract_optic_disc_features(self, image, mask):
        """Extract optic disc related features - only selected features"""
        # Convert to LAB color space for better disc detection
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
        
        # Enhanced disc detection using multiple approaches
        # Method 1: Brightness-based
        _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, kernel)
        _, morph_mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(bright_mask, morph_mask)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize features with default values
        features = {
            'optic_disc_area_ratio': 0,
            'disc_circularity': 0,
            'disc_displacement': 0,
            'disc_eccentricity': 0,
            'cup_to_disc_ratio': 0,
            'disc_aspect_ratio': 0,
            'disc_ellipse_angle': 0
        }
        
        # Find the largest bright region (likely optic disc)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate optic disc features
            disc_area = cv2.contourArea(largest_contour)
            total_area = np.sum(mask > 0)
            disc_area_ratio = disc_area / total_area if total_area > 0 else 0
            
            # Disc shape analysis
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * disc_area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Enhanced cup-to-disc ratio calculation
            cup_to_disc_ratio = self.extract_cup_to_disc_ratio(largest_contour, image)
            
            # Disc center and eccentricity
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Distance from image center (disc displacement)
                img_center_x, img_center_y = image.shape[1] // 2, image.shape[0] // 2
                disc_displacement = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
                
                # Fit ellipse to get eccentricity
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
                    
                    # Additional ellipse features
                    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
                    ellipse_angle = ellipse[2]
                else:
                    eccentricity = 0
                    aspect_ratio = 0
                    ellipse_angle = 0
            else:
                disc_displacement = 0
                eccentricity = 0
                aspect_ratio = 0
                ellipse_angle = 0
            
            features = {
                'optic_disc_area_ratio': disc_area_ratio,
                'disc_circularity': circularity,
                'disc_displacement': disc_displacement,
                'disc_eccentricity': eccentricity,
                'cup_to_disc_ratio': cup_to_disc_ratio,
                'disc_aspect_ratio': aspect_ratio,
                'disc_ellipse_angle': ellipse_angle
            }
        
        return features
    
    def extract_macular_features(self, image, mask):
        """Extract macular region features - only mean intensity for r0"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Get image center (approximate macular region)
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        
        # Use only the first macular radius
        radius = min(width, height) // 8
        
        macular_mask = np.zeros_like(gray)
        cv2.circle(macular_mask, (center_x, center_y), radius, 255, -1)
        
        # Extract macular region
        macular_region = cv2.bitwise_and(gray, gray, mask=macular_mask)
        macular_pixels = macular_region[macular_region > 0]
        
        if len(macular_pixels) > 0:
            mean_intensity = np.mean(macular_pixels)
        else:
            mean_intensity = 0
        
        return {
            'macular_mean_intensity_r0': mean_intensity
        }
    
    def extract_features_from_image(self, image_path, label):
        """Extract selected features from a single image"""
        try:
            # Preprocess image
            image, mask = self.preprocess_image(image_path)
            
            # Extract selected types of features
            vessel_features = self.extract_vessel_features(image, mask)
            optic_disc_features = self.extract_optic_disc_features(image, mask)
            macular_features = self.extract_macular_features(image, mask)
            
            # Combine all features
            all_features = {
                'filename': os.path.basename(image_path),
                'label': label,
                **vessel_features,
                **optic_disc_features,
                **macular_features
            }
            
            return all_features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# In the main execution block after training (replace the existing save section):
# Save model AND preprocessing objects
# Remove all code related to joblib.dump, model saving, and test dataset processing at the bottom of the file.


class MyopiaCheckerBackend:
    def __init__(self, model_path='../python/Myopia Detection/enhanced_myopia_model.pkl', scaler_path='../python/Myopia Detection/scaler.pkl', pca_path='../python/Myopia Detection/pca.pkl', feature_names_path='../python/Myopia Detection/feature_names.pkl'):
        self.model = joblib.load(os.path.abspath(model_path))
        self.scaler = joblib.load(os.path.abspath(scaler_path))
        self.pca = joblib.load(os.path.abspath(pca_path))
        self.feature_names = joblib.load(os.path.abspath(feature_names_path))
        self.extractor = EnhancedMyopiaFeatureExtractor()
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['normal', 'myopia'])

    def predict_from_image_array(self, image_array):
        import tempfile
        from PIL import Image
        # Save the numpy array as a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.fromarray(image_array)
            img.save(tmp.name)
            tmp_path = tmp.name
        try:
            # Extract features
            features = self.extractor.extract_features_from_image(tmp_path, label=None)
            if features is None:
                return {"status": "error", "message": "Feature extraction failed"}
            # Prepare feature vector
            X = np.array([features[f] for f in self.feature_names]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            # Predict
            y_pred = self.model.predict(X_pca)
            y_proba = self.model.predict_proba(X_pca)
            pred_label = self.label_encoder.inverse_transform(y_pred)[0]
            confidence = float(np.max(y_proba))
            # Remove filename and label from features for frontend display
            display_features = {k: v for k, v in features.items() if k not in ('filename', 'label')}
            return {
                "status": "success",
                "predicted_class": pred_label,
                "confidence": confidence,
                "probabilities": {
                    "normal": float(y_proba[0,0]),
                    "myopia": float(y_proba[0,1])
                },
                "features": display_features
            }
        finally:
            os.remove(tmp_path)