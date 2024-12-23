import os
import yaml
from pathlib import Path
import torch

class Config:
    def __init__(self, config_path=None):
        """
        Initialize configuration
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or os.path.join('configs', 'models.yaml')
        self.config = self._load_config()
        self.device = self._setup_device()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _setup_device(self):
        """Setup compute device and optimize CUDA if available"""
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return "cpu"

        # CUDA is available, optimize settings
        device = "cuda"
        
        # Enable TF32 on Ampere GPUs
        if torch.cuda.get_device_capability()[0] >= 8:
            print("Enabling TF32 on Ampere GPU")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Set cudnn to benchmark mode for faster convolutions
        torch.backends.cudnn.benchmark = True

        # Print GPU info
        gpu_name = torch.cuda.get_device_name()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory Available: {mem_gb:.2f} GB")

        return device

    def get_model_path(self, model_type):
        """Get model path from configuration"""
        try:
            return self.config['models'][model_type]['url']
        except KeyError:
            print(f"Model type {model_type} not found in config")
            return None

    def get_model_config(self, model_type):
        """Get model configuration"""
        try:
            return self.config['models'][model_type]['config']
        except KeyError:
            print(f"Configuration for {model_type} not found")
            return {}

    def get_inference_config(self):
        """Get inference configuration"""
        return self.config.get('inference', {})

    def get_device(self):
        """Get compute device"""
        return self.device

# Initialize global configuration
CONFIG = Config()
