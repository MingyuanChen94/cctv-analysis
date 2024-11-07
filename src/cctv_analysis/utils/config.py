# src/cctv_analysis/utils/config.py
import torch
import os
from pathlib import Path

class GPUConfig:
    @staticmethod
    def get_device():
        """
        Configure GPU settings and return appropriate device
        """
        if not torch.cuda.is_available():
            print("GPU not available, using CPU")
            return "cpu"
            
        # Get the GPU device
        device = torch.device("cuda:0")
        
        # Print GPU information
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory Available: {gpu_memory:.2f} GB")
        
        # Enable TF32 on Ampere GPUs for better performance
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            print("Enabling TF32 for better performance")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmark mode
        torch.backends.cudnn.benchmark = True
        
        return "cuda"

class ModelPaths:
    def __init__(self, base_path=None):
        """
        Initialize model paths
        Args:
            base_path: Base path to models directory, defaults to project root/models
        """
        if base_path is None:
            # Get the project root directory
            project_root = Path(__file__).resolve().parents[3]
            base_path = project_root / "models"
            
        self.base_path = Path(base_path)
        
        # Define model paths
        self.detector_path = self.base_path / "detector" / "yolox_l.pth"
        self.reid_path = self.base_path / "reid" / "osnet_x1_0.pth"
        
        # Verify model files exist
        self._verify_paths()
        
    def _verify_paths(self):
        """Verify all model paths exist"""
        missing_files = []
        for path_attr in [self.detector_path, self.reid_path]:
            if not path_attr.exists():
                missing_files.append(path_attr)
                
        if missing_files:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(str(f) for f in missing_files)}\n"
                "Please download the required model weights using scripts/download_models.py"
            )
