#!/usr/bin/env python3
"""
Script to verify model weights for CCTV Analysis.
Checks file existence, size, and integrity.
"""

import os
import hashlib
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import logging

console = Console()

def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('model_verification.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

class ModelVerifier:
    """Verifies the integrity and usability of model weights."""
    
    def __init__(self, models_dir: Path = None, config_path: Path = None):
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parents[1] / "configs" / "models.yaml"
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Set up base directory
        if models_dir:
            self.base_dir = models_dir
        else:
            self.base_dir = Path(self.config['paths']['base_dir']).resolve()
    
    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file with progress bar."""
        file_size = file_path.stat().st_size
        md5_hash = hashlib.md5()
        
        with Progress() as progress:
            task = progress.add_task(
                f"Calculating MD5 for {file_path.name}",
                total=file_size
            )
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    md5_hash.update(chunk)
                    progress.update(task, advance=len(chunk))
        
        return md5_hash.hexdigest()
    
    def check_model_loadable(self, file_path: Path, model_type: str) -> bool:
        """Try to load model file with PyTorch."""
        try:
            if model_type == "detector":
                # Try loading YOLOX model
                state_dict = torch.load(file_path, map_location='cpu')
                return 'model' in state_dict
            elif model_type == "reid":
                # Try loading OSNet model
                state_dict = torch.load(file_path, map_location='cpu')
                return isinstance(state_dict, dict)
            return False
        except Exception as e:
            logger.error(f"Error loading model {file_path}: {e}")
            return False
    
    def verify_model(self, model_name: str) -> dict:
        """Verify a specific model."""
        model_info = self.config['models'][model_name]
        file_path = self.base_dir / model_info['output_path']
        
        result = {
            'name': model_info['name'],
            'version': model_info['version'],
            'path': file_path,
            'exists': file_path.exists(),
            'size_match': False,
            'md5_match': False,
            'loadable': False,
            'required': model_info['required']
        }
        
        if result['exists']:
            # Check file size
            actual_size = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
            expected_size = model_info.get('size_mb', 0)
            result['size_match'] = abs(actual_size - expected_size) < 1.0  # 1MB tolerance
            
            # Check MD5 if provided
            if 'md5' in model_info:
                actual_md5 = self.calculate_md5(file_path)
                result['md5_match'] = actual_md5 == model_info['md5']
            else:
                result['md5_match'] = True  # Skip MD5 check if not provided
            
            # Try loading model
            result['loadable'] = self.check_model_loadable(file_path, model_name)
        
        return result
    
    def verify_all_models(self) -> List[dict]:
        """Verify all models."""
        results = []
        for model_name in self.config['models']:
            console.print(f"\nVerifying {model_name} model...")
            result = self.verify_model(model_name)
            results.append(result)
        return results

def display_results(results: List[dict]):
    """Display verification results in a formatted table."""
    table = Table(
        title="Model Verification Results",
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Model", style="cyan")
    table.add_column("Version", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Path", style="yellow")
    table.add_column("Details", style="white")
    
    for result in results:
        # Determine status and details
        if not result['exists']:
            status = "❌ Missing"
            details = "File not found"
            status_style = "red"
        elif not result['size_match']:
            status = "⚠️ Size Mismatch"
            details = "Incorrect file size"
            status_style = "yellow"
        elif not result['md5_match']:
            status = "⚠️ Hash Mismatch"
            details = "MD5 verification failed"
            status_style = "yellow"
        elif not result['loadable']:
            status = "⚠️ Not Loadable"
            details = "File cannot be loaded"
            status_style = "yellow"
        else:
            status = "✓ Valid"
            details = "All checks passed"
            status_style = "green"
        
        table.add_row(
            result['name'],
            result['version'],
            f"[{status_style}]{status}[/]",
            str(result['path']),
            details
        )
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description="Verify CCTV Analysis model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model',
        choices=['detector', 'reid'],
        help='Specific model to verify'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        help='Directory containing the models'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed verification information'
    )
    
    args = parser.parse_args()
    
    try:
        verifier = ModelVerifier(args.models_dir)
        
        if args.model:
            results = [verifier.verify_model(args.model)]
        else:
            results = verifier.verify_all_models()
        
        display_results(results)
        
        # Check if any required models are invalid
        has_invalid = any(
            not (r['exists'] and r['size_match'] and r['md5_match'] and r['loadable'])
            for r in results if r['required']
        )
        
        if has_invalid:
            console.print("\n[red]❌ Some required models are missing or invalid.")
            console.print(
                "[yellow]Run 'python scripts/download_models.py' to download "
                "or update models."
            )
            sys.exit(1)
        else:
            console.print("\n[green]✓ All model verifications passed!")
            
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
