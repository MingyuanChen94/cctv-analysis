"""Configuration handler for CCTV analysis system."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

@dataclass
class Config:
    """Configuration handler for CCTV analysis system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None,
                 config_dict: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
        """
        if config_dict is not None:
            self._config = config_dict
        elif config_path is not None:
            self._config = self._load_yaml(config_path)
        else:
            # Load default config from package directory
            default_config = Path(__file__).parent.parent.parent.parent / "configs" / "config.yaml"
            self._config = self._load_yaml(default_config)
            
        self._validate_config()
        self._setup_paths()
        
    def _load_yaml(self, config_path: Union[str, Path]) -> Dict:
        """Load YAML configuration file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing configuration file: {e}")
            
    def _validate_config(self):
        """Validate configuration values."""
        required_sections = [
            'processing', 'paths', 'detector', 'tracker',
            'reid', 'matching', 'demographics', 'visualization'
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigError(f"Missing required configuration section: {section}")
                
        # Validate processing settings
        if self.processing.device not in ['cuda', 'cpu']:
            raise ConfigError("Processing device must be 'cuda' or 'cpu'")
            
        # Validate detector settings
        if self.detector.backend not in ['yolov5', 'ssd', 'hog']:
            raise ConfigError("Unsupported detector backend")
            
        # Validate demographic settings
        if self.demographics.backend not in ['opencv', 'ssd', 'dlib', 'mtcnn']:
            raise ConfigError("Unsupported demographics backend")
            
    def _setup_paths(self):
        """Setup and create required directories."""
        base_dir = Path(__file__).parent.parent.parent.parent
        
        # Update relative paths to absolute paths
        for key, path in self.paths.items():
            if isinstance(path, str) and not os.path.isabs(path):
                self.paths[key] = str(base_dir / path)
                
        # Create directories if they don't exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
    def __getattr__(self, name: str) -> Any:
        """Enable dot notation access to configuration sections."""
        if name in self._config:
            return ConfigSection(self._config[name])
        raise AttributeError(f"Configuration section not found: {name}")
        
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self._config.copy()
        
    def update(self, updates: Dict):
        """Update configuration with new values."""
        def update_recursive(original: Dict, updates: Dict):
            for key, value in updates.items():
                if (key in original and isinstance(original[key], dict) 
                    and isinstance(value, dict)):
                    update_recursive(original[key], value)
                else:
                    original[key] = value
                    
        update_recursive(self._config, updates)
        self._validate_config()
        
        
class ConfigSection:
    """Wrapper for configuration sections enabling dot notation access."""
    
    def __init__(self, section: Dict):
        self._section = section
        
    def __getattr__(self, name: str) -> Any:
        """Enable dot notation access to section items."""
        if name in self._section:
            value = self._section[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"Configuration item not found: {name}")
        
    def to_dict(self) -> Dict:
        """Convert section to dictionary."""
        return self._section.copy()
