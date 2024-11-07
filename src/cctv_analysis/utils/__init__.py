# src/cctv_analysis/utils/init.py
from ..detector import PersonDetector
from ..reid import PersonReID
from ..demographics import DemographicAnalyzer
from ..matcher import PersonMatcher
from .config import GPUConfig, ModelPaths

class ModelInitializer:
    def __init__(self, model_paths=None):
        """
        Initialize all models with proper GPU configuration
        Args:
            model_paths: Optional ModelPaths instance
        """
        # Configure device
        self.device = GPUConfig.get_device()
        
        # Get model paths
        if model_paths is None:
            model_paths = ModelPaths()
        self.model_paths = model_paths
        
        # Initialize models
        self.detector = None
        self.reid_model = None
        self.demographic_analyzer = None
        self.matcher = None
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all models with proper error handling"""
        try:
            self.detector = PersonDetector(
                model_path=str(self.model_paths.detector_path),
                model_size='l',
                device=self.device
            )
            
            self.reid_model = PersonReID(
                model_path=str(self.model_paths.reid_path),
                device=self.device
            )
            
            self.demographic_analyzer = DemographicAnalyzer(
                device=self.device
            )
            
            self.matcher = PersonMatcher(
                similarity_threshold=0.75
            )
            
        except Exception as e:
            raise RuntimeError(f"Error initializing models: {e}")
