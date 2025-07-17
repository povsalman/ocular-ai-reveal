import numpy as np
from models.myopia_checker import MyopiaCheckerBackend

class MyopiaModel:
    def __init__(self):
        self.checker = MyopiaCheckerBackend(
            model_path='../python/Myopia Detection/enhanced_myopia_model.pkl',
            scaler_path='../python/Myopia Detection/scaler.pkl',
            pca_path='../python/Myopia Detection/pca.pkl',
            feature_names_path='../python/Myopia Detection/feature_names.pkl'
        )

    def predict(self, image_array: np.ndarray):
        # image_array is a numpy array (H, W, 3) in RGB
        result = self.checker.predict_from_image_array(image_array)
        # Map to backend API response format
        if result["status"] == "success":
            response = {
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "status": "success"
            }
            if "features" in result:
                response["features"] = result["features"]
            return response
        else:
            return {
                "predicted_class": "error",
                "confidence": 0.0,
                "probabilities": {},
                "status": "error",
                "message": result.get("message", "Unknown error")
            } 