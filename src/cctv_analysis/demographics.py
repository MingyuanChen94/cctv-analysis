import insightface
import cv2
import numpy as np

class DemographicAnalyzer:
    def __init__(self, model_path=None, device="cuda"):
        """
        Initialize demographic analyzer using InsightFace
        Args:
            model_path: Optional path to custom model
            device: Device to run inference on
        """
        self.model = insightface.app.FaceAnalysis(
            allowed_modules=['detection', 'genderage'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.model.prepare(ctx_id=0 if device == "cuda" else -1)
        
    def analyze(self, img):
        """
        Analyze demographics from face image
        Args:
            img: numpy array of BGR image
        Returns:
            List of dictionaries containing gender and age for each detected face
        """
        faces = self.model.get(img)
        results = []
        
        for face in faces:
            gender = "female" if face.sex == 0 else "male"
            # Define age groups
            age = face.age
            if age < 18:
                age_group = "under_18"
            elif age < 30:
                age_group = "18-29"
            elif age < 50:
                age_group = "30-49"
            else:
                age_group = "50+"
                
            results.append({
                "gender": gender,
                "age_group": age_group,
                "confidence": face.det_score
            })
            
        return results
