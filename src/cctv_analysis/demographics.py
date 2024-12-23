import insightface
import cv2
import numpy as np
import os
from pathlib import Path
import onnxruntime

class DemographicAnalyzer:
    def __init__(self, model_path=None, device="cuda"):
        """
        Initialize demographic analyzer using InsightFace
        Args:
            model_path: Optional path to custom model
            device: Device to run inference on
        """
        # Set environment variables
        home = str(Path.home())
        self.model_dir = os.path.join(home, '.insightface/models/buffalo_l')
        
        # Force creation of model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize ONNX Runtime session
        if device == "cuda" and self._check_cuda_available():
            self.providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            print("Using CUDA for demographic analysis")
        else:
            self.providers = [('CPUExecutionProvider', {})]
            print("Using CPU for demographic analysis")

        try:
            # Initialize the face analyzer with explicit paths and configuration
            self.model = insightface.app.FaceAnalysis(
                name="buffalo_l",
                root=os.path.dirname(self.model_dir),
                providers=self.providers,
                allowed_modules=['detection', 'genderage']
            )

            # Set model files explicitly
            model_files = {
                'det_10g.onnx': 'detection',
                'genderage.onnx': 'genderage'
            }

            # Prepare model with specific configuration
            self.model.prepare(
                ctx_id=0 if device == "cuda" else -1,
                det_size=(640, 640),
                det_thresh=0.5
            )

            print("Model preparation successful")
            
            # Verify models are loaded
            if not hasattr(self.model, 'models') or len(self.model.models) == 0:
                raise RuntimeError("Models not properly loaded")

        except Exception as e:
            print(f"Error during model initialization: {e}")
            self._manual_model_init()

    def _manual_model_init(self):
        """Manual initialization of InsightFace models if automatic init fails"""
        try:
            print("Attempting manual model initialization...")
            
            # Initialize detection model
            det_path = os.path.join(self.model_dir, 'det_10g.onnx')
            self.det_session = onnxruntime.InferenceSession(
                det_path,
                providers=self.providers
            )

            # Initialize gender-age model
            genderage_path = os.path.join(self.model_dir, 'genderage.onnx')
            self.genderage_session = onnxruntime.InferenceSession(
                genderage_path,
                providers=self.providers
            )

            print("Manual model initialization successful")
            
        except Exception as e:
            print(f"Error during manual model initialization: {e}")
            print(f"Please ensure models are downloaded to: {self.model_dir}")
            raise

    def _check_cuda_available(self):
        """Check if CUDA is available for inference"""
        try:
            return 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        except:
            return False

    def analyze(self, img):
        """
        Analyze demographics from face image
        Args:
            img: numpy array of BGR image
        Returns:
            List of dictionaries containing gender and age for each detected face
        """
        if img is None or img.size == 0:
            print("Invalid input image")
            return []

        try:
            # Ensure image is in BGR format
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # Resize image if too large
            max_size = 1920
            scale = min(max_size / max(img.shape[0], img.shape[1]), 1.0)
            if scale < 1.0:
                new_size = tuple(int(dim * scale) for dim in img.shape[:2][::-1])
                img = cv2.resize(img, new_size)

            # Process image
            try:
                faces = self.model.get(img)
            except AttributeError:
                print("Using manual detection fallback...")
                faces = self._manual_detect(img)

            results = []
            for face in faces:
                if hasattr(face, 'det_score') and face.det_score < 0.5:
                    continue

                # Extract gender and age
                gender = "female" if getattr(face, 'sex', 0) == 0 else "male"
                age = int(getattr(face, 'age', 25))
                
                # Define age group
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
                    "age": age,
                    "age_group": age_group,
                    "confidence": float(getattr(face, 'det_score', 0.0)),
                    "bbox": face.bbox.tolist() if hasattr(face, 'bbox') else None
                })

            return results

        except Exception as e:
            print(f"Error during face analysis: {e}")
            return []

    def _manual_detect(self, img):
        """Manual detection fallback if model.get() fails"""
        # This is a simplified implementation
        try:
            # Prepare image for detection
            input_size = (640, 640)
            im_ratio = float(img.shape[0]) / img.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio > model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)
            det_scale = float(new_height) / img.shape[0]
            resized_img = cv2.resize(img, (new_width, new_height))
            
            # Add padding
            det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
            det_img[:new_height, :new_width, :] = resized_img

            # Normalize and transpose image
            blob = cv2.dnn.blobFromImage(
                det_img, 
                1.0/128, 
                input_size, 
                (127.5, 127.5, 127.5), 
                swapRB=True
            )

            # Run detection
            self.det_session.run(None, {'input.1': blob})
            
            # For simplicity, return empty list as this is just a fallback
            return []
            
        except Exception as e:
            print(f"Error in manual detection: {e}")
            return []

    def get_model_info(self):
        """Get information about loaded models and configuration"""
        return {
            "providers": self.providers,
            "model_dir": self.model_dir,
            "detection_size": (640, 640)
        }
