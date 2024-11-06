#!/usr/bin/env python3
"""
Script to download required model weights for CCTV Analysis.
Supports both direct downloads and Google Drive downloads.
"""

import os
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
import sys
from urllib.parse import urlparse, parse_qs

class GoogleDriveDownloader:
    """Handles downloading files from Google Drive."""
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_confirm_token(self, response):
        """Extract confirmation token from response."""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def download_file(self, url: str, output_path: Path, 
                     progress_callback=None) -> bool:
        """
        Download a file from Google Drive.
        
        Args:
            url: Google Drive download URL
            output_path: Path to save the file
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if download was successful
        """
        try:
            file_id = self._extract_file_id(url)
            if not file_id:
                raise ValueError("Could not extract Google Drive file ID from URL")
            
            response = self.session.get(url, stream=True, allow_redirects=True)
            token = self.get_confirm_token(response)
            
            if token:
                params = {'confirm': token}
                response = self.session.get(url, params=params, stream=True)
            
            file_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                if progress_callback and file_size:
                    with tqdm(
                        total=file_size,
                        unit='iB',
                        unit_scale=True,
                        desc=f"Downloading {output_path.name}"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            pbar.update(size)
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Error downloading from Google Drive: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _extract_file_id(self, url: str) -> str:
        """Extract file ID from Google Drive URL."""
        parsed = urlparse(url)
        if parsed.netloc == 'drive.google.com':
            if 'id=' in url:
                return parse_qs(parsed.query)['id'][0]
            else:
                path_parts = parsed.path.split('/')
                if 'd' in path_parts:
                    idx = path_parts.index('d')
                    if idx + 1 < len(path_parts):
                        return path_parts[idx + 1]
        return None

class ModelDownloader:
    """Handles downloading and verifying model weights."""
    
    def __init__(self, models_dir: Path = None):
        # Load configuration
        config_path = Path(__file__).parents[1] / "configs" / "models.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Set up base directory
        if models_dir:
            self.base_dir = models_dir
        else:
            self.base_dir = Path(self.config['paths']['base_dir']).resolve()
        
        # Initialize downloaders
        self.gdrive_downloader = GoogleDriveDownloader()
        
        # Create necessary directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / ".temp").mkdir(exist_ok=True)
        (self.base_dir / ".backup").mkdir(exist_ok=True)
    
    def download_file(self, url: str, output_path: Path, 
                     source_type: str = "direct") -> bool:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            output_path: Path to save the file
            source_type: Type of source ("direct" or "gdrive")
            
        Returns:
            bool: True if download was successful
        """
        # Ensure parent directories exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if source_type == "gdrive":
                return self.gdrive_downloader.download_file(url, output_path)
            
            # Direct download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_size = int(response.headers.get('content-length', 0))
            
            with tqdm(
                total=file_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {output_path.name}"
            ) as progress_bar:
                with open(output_path, 'wb') as f:
                    for data in response.iter_content(chunk_size=8192):
                        size = f.write(data)
                        progress_bar.update(size)
            
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_models(self, models: list = None) -> bool:
        """
        Download specified models or all models if none specified.
        
        Args:
            models: List of model names to download, or None for all models
            
        Returns:
            bool: True if all downloads were successful
        """
        if models is None:
            models = list(self.config['models'].keys())
        
        success = True
        for model_name in models:
            if model_name not in self.config['models']:
                print(f"Unknown model: {model_name}")
                continue
                
            model_info = self.config['models'][model_name]
            output_path = self.base_dir / model_info['output_path']
            
            # Skip if file exists
            if output_path.exists():
                print(f"✓ {model_info['name']} already downloaded to {output_path}")
                continue
            
            print(f"\nDownloading {model_info['name']} to {output_path}...")
            source_type = model_info.get('source_type', 'direct')
            if not self.download_file(
                model_info['url'],
                output_path,
                source_type
            ):
                success = False
            else:
                print(f"✓ Successfully downloaded {model_info['name']}")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Download model weights for CCTV Analysis")
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to download'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        help='Directory to store models'
    )
    
    args = parser.parse_args()
    
    print("CCTV Analysis Model Downloader")
    print("==============================")
    
    try:
        downloader = ModelDownloader(args.models_dir)
        if downloader.download_models(args.models):
            print("\n✓ All models downloaded successfully!")
        else:
            print("\n✗ Some downloads failed. Please try again.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
