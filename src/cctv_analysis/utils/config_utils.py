import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigurationManager:
    """Manages configuration loading and validation."""

    DEFAULT_CONFIG = {
        "detector": {
            "backend": "yolov5",
            "confidence_threshold": 0.5,
            "device": "cuda",
        },
        "tracker": {"max_disappeared": 30, "max_distance": 0.6, "min_confidence": 0.5},
        "demographics": {"batch_size": 32, "backend": "opencv"},
        "logging": {
            "level": "INFO",
            "console": True,
            "file": {"enabled": True, "directory": "logs"},
        },
        "output": {"save_video": True, "save_csv": True, "visualization": True},
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path (Optional[str]): Path to configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path (str): Path to configuration file

        Returns:
            Dict[str, Any]: Loaded configuration
        """
        try:
            with open(config_path) as f:
                file_config = yaml.safe_load(f)

            # Update default config with file config
            self._update_recursive(self.config, file_config)

            return self.config

        except Exception as e:
            raise RuntimeError(f"Error loading config from {config_path}: {str(e)}")

    def _update_recursive(self, base_dict: Dict, update_dict: Dict):
        """Recursively update dictionary while preserving nested structure."""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._update_recursive(base_dict[key], value)
            else:
                base_dict[key] = value

    def save_config(self, output_path: str):
        """
        Save current configuration to file.

        Args:
            output_path (str): Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key (str): Configuration key (supports dot notation)
            default (Any): Default value if key not found

        Returns:
            Any: Configuration value
        """
        try:
            value = self.config
            for k in key.split("."):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key (str): Configuration key (supports dot notation)
            value (Any): Value to set
        """
        keys = key.split(".")
        current = self.config

        # Navigate to the correct nested level
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate detector config
            assert self.get("detector.backend") in ["yolov5", "ssd", "hog"]
            assert 0 < self.get("detector.confidence_threshold") < 1

            # Validate tracker config
            assert self.get("tracker.max_disappeared") > 0
            assert 0 < self.get("tracker.max_distance") < 1
            assert 0 < self.get("tracker.min_confidence") < 1

            # Validate demographics config
            assert self.get("demographics.batch_size") > 0
            assert self.get("demographics.backend") in ["opencv", "dlib", "mtcnn"]

            return True

        except AssertionError:
            return False

    def export_env_vars(self):
        """Export configuration as environment variables."""

        def _flatten_dict(d: Dict, parent_key: str = "") -> Dict[str, str]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key).items())
                else:
                    items.append((new_key.upper(), str(v)))
            return dict(items)

        # Set environment variables
        for key, value in _flatten_dict(self.config).items():
            os.environ[f"CCTV_ANALYSIS_{key}"] = value


# Example usage
if __name__ == "__main__":
    # Create config manager with default configuration
    config_manager = ConfigurationManager()

    # Load custom configuration
    config_manager.load_config("config/custom_config.yaml")

    # Get configuration values
    detector_backend = config_manager.get("detector.backend")
    print(f"Detector backend: {detector_backend}")

    # Validate configuration
    is_valid = config_manager.validate_config()
    print(f"Configuration is valid: {is_valid}")

    # Save configuration
    config_manager.save_config("config/saved_config.yaml")

    # Export as environment variables
    config_manager.export_env_vars()
