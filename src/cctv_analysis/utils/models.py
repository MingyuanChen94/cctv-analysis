import torch
import torchvision.transforms as T
from yolox.exp import get_exp
from yolox.utils import postprocess
from tracker.byte_tracker import BYTETracker
from torchreid.utils import FeatureExtractor
from typing import Optional

from cctv_analysis.utils.config import Config
from cctv_analysis.utils.logger import setup_logger

class ModelManager:
    """Manages all models used in the CCTV analysis pipeline"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("ModelManager")
        self.device = torch.device(config.processing.device)
        
        # Initialize models
        self.logger.info("Initializing models...")
        self.detector = self._init_detector()
        self.tracker = self._init_tracker()
        self.reid_extractor = self._init_reid()
        
        # Initialize transforms
        self.reid_transform = self._init_reid_transform()
        self.logger.info("Model initialization complete")
    
    def _init_detector(self):
        """Initialize YOLOX detector"""
        try:
            exp = get_exp(self.config.detector.exp_file)
            model = exp.get_model()
            model.eval()
            
            self.logger.info(f"Loading detector weights from {self.config.detector.weights}")
            ckpt = torch.load(
                self.config.detector.weights,
                map_location=self.device
            )
            model.load_state_dict(ckpt["model"])
            model.to(self.device)
            
            return model
        except Exception as e:
            self.logger.error(f"Error initializing detector: {e}")
            raise
    
    def _init_tracker(self):
        """Initialize ByteTracker"""
        try:
            return BYTETracker(
                track_thresh=self.config.tracker.track_thresh,
                track_buffer=self.config.tracker.track_buffer,
                match_thresh=self.config.tracker.match_thresh
            )
        except Exception as e:
            self.logger.error(f"Error initializing tracker: {e}")
            raise
    
    def _init_reid(self):
        """Initialize ReID model"""
        try:
            self.logger.info(f"Loading ReID model from {self.config.reid.weights}")
            return FeatureExtractor(
                model_name=self.config.reid.model,
                model_path=str(self.config.reid.weights),
                device=self.device
            )
        except Exception as e:
            self.logger.error(f"Error initializing ReID model: {e}")
            raise
    
    def _init_reid_transform(self):
        """Initialize transforms for ReID"""
        return T.Compose([
            T.Resize(size=self.config.reid.input_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def detect(self, batch: torch.Tensor):
        """Run detection on a batch of frames"""
        try:
            outputs = self.detector(batch)
            return postprocess(
                outputs,
                1,  # num_classes (person only)
                self.config.detector.confidence_threshold
            )
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            raise
    
    def track(self, detections, image_size):
        """Run tracking on detections"""
        try:
            return self.tracker.update(
                detections.cpu().numpy(),
                [image_size[0], image_size[1]],
                [image_size[0], image_size[1]]
            )
        except Exception as e:
            self.logger.error(f"Error during tracking: {e}")
            raise
    
    def extract_reid_features(self, frame: torch.Tensor,
                            bbox: torch.Tensor) -> torch.Tensor:
        """Extract ReID features for a person detection"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            person_img = frame[:, y1:y2, x1:x2].cpu().numpy().transpose(1, 2, 0)
            
            # Prepare for ReID model
            img_tensor = self.reid_transform(person_img).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Extract features
            return self.reid_extractor(img_tensor)
        except Exception as e:
            self.logger.error(f"Error extracting ReID features: {e}")
            raise
