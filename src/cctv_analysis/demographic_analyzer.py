import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace


class DemographicAnalyzer:
    """
    Analyze demographic attributes (age, gender) of detected persons
    using DeepFace framework.
    """

    def __init__(self, config: dict):
        """
        Initialize the demographic analyzer.

        Args:
            config (dict): Configuration dictionary containing:
                - models_path: Path to pre-trained models
                - batch_size: Batch size for processing
                - device: Device to run inference on ('cpu' or 'gpu')
                - backend: Backend model for face analysis ('opencv', 'ssd', 'dlib', 'mtcnn')
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.models_path = Path(config.get("models_path", "models"))
        self.batch_size = config.get("batch_size", 32)
        self.device = config.get("device", "cpu")
        self.backend = config.get("backend", "opencv")

        # Configure GPU memory growth if using GPU
        if self.device == "gpu":
            self._configure_gpu()

        self._initialize_models()

    def _configure_gpu(self):
        """Configure TensorFlow GPU memory growth."""
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            self.logger.warning(f"Error configuring GPU: {str(e)}")

    def _initialize_models(self):
        """Initialize face detection and analysis models."""
        try:
            # Initialize face detector
            if self.backend == "opencv":
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            else:
                # DeepFace will handle other backends
                pass

            # Warm up DeepFace models
            self.logger.info("Initializing DeepFace models...")
            sample_img = np.zeros((224, 224, 3), dtype=np.uint8)
            _ = DeepFace.analyze(
                sample_img,
                actions=["age", "gender"],
                enforce_detection=False,
                silent=True,
            )

        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def analyze(self, image: np.ndarray) -> Dict[str, any]:
        """
        Analyze demographic attributes in the given image.

        Args:
            image (np.ndarray): Input image (cropped person ROI)

        Returns:
            Dict: Dictionary containing:
                - age: Estimated age
                - gender: Predicted gender
                - confidence: Confidence scores for predictions
        """
        if image is None or image.size == 0:
            return self._get_default_results()

        try:
            # Ensure image is in BGR format (DeepFace requirement)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Detect face in the image
            face_img = self._detect_face(image)
            if face_img is None:
                return self._get_default_results()

            # Analyze demographics using DeepFace
            results = DeepFace.analyze(
                face_img,
                actions=["age", "gender"],
                enforce_detection=False,
                silent=True,
            )

            # Process results
            return {
                "age": results[0]["age"],
                "gender": results[0]["gender"],
                "confidence": {
                    "gender": results[0]["gender_probability"],
                    "age": self._calculate_age_confidence(results[0]["age"]),
                },
            }

        except Exception as e:
            self.logger.error(f"Error analyzing demographics: {str(e)}")
            return self._get_default_results()

    def _detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract face from the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            Optional[np.ndarray]: Cropped face image or None if no face detected
        """
        try:
            if self.backend == "opencv":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if len(faces) > 0:
                    x, y, w, h = faces[0]  # Use the first detected face
                    return image[y : y + h, x : x + w]

            return image  # Return full image if no face detected or using different backend

        except Exception as e:
            self.logger.error(f"Error detecting face: {str(e)}")
            return None

    def _calculate_age_confidence(self, age: float) -> float:
        """
        Calculate confidence score for age prediction.
        Simple heuristic based on typical age estimation errors.

        Args:
            age (float): Predicted age

        Returns:
            float: Confidence score between 0 and 1
        """
        # Age prediction tends to be less reliable for very young and very old ages
        if age < 0 or age > 100:
            return 0.0
        elif age < 12 or age > 75:
            return 0.7
        else:
            return 0.85

    def _get_default_results(self) -> Dict[str, any]:
        """Return default results when analysis fails."""
        return {"age": None, "gender": None, "confidence": {"gender": 0.0, "age": 0.0}}
