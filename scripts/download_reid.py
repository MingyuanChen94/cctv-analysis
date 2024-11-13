# scripts/download_models.py

import torch
import torchvision.models as models
import os
import requests
from tqdm import tqdm
import hashlib
import yaml

def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar
    Args:
        url: URL to download from
        destination: Local path to save file
        chunk_size: Download chunk size
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    with open(destination, "wb") as f:
        with tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=f"Downloading {os.path.basename(destination)}"
        ) as pbar:
            for data in response.iter_content(chunk_size):
                size = f.write(data)
                pbar.update(size)

def get_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_inception_v3():
    """Download and prepare Inception-v3 model for ReID"""
    # Create models directory if it doesn't exist
    models_dir = os.path.join("models", "reid")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model paths
    model_path = os.path.join(models_dir, "inception_v3_reid.pth")
    
    # Load base Inception-v3 model
    print("Loading pretrained Inception-v3...")
    model = models.inception_v3(pretrained=True)
    
    # Modify for ReID
    model.aux_logits = False
    num_features = model.fc.in_features
    
    # Replace classifier with identity dimension reduction
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 751)  # Market-1501 has 751 identities
    )
    
    # Save modified model
    print(f"Saving modified model to {model_path}")
    torch.save({
        'state_dict': model.state_dict(),
        'num_classes': 751,
        'feature_dim': 2048
    }, model_path)
    
    # Calculate and save hash
    model_hash = get_file_hash(model_path)
    
    # Update config
    config_path = os.path.join("configs", "models.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['models']['person_reid']['checksum'] = model_hash
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Model saved with hash: {model_hash}")
    print("Config updated with new checksum")

def download_yolov8():
    """Download YOLOv8x6 model"""
    models_dir = os.path.join("models", "detector")
    os.makedirs(models_dir, exist_ok=True)
    
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x6.pt"
    model_path = os.path.join(models_dir, "yolov8x6.pt")
    
    print("Downloading YOLOv8x6...")
    download_file(url, model_path)
    
    # Calculate and save hash
    model_hash = get_file_hash(model_path)
    
    # Update config
    config_path = os.path.join("configs", "models.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['models']['person_detection']['checksum'] = model_hash
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLOv8x6 saved with hash: {model_hash}")
    print("Config updated with new checksum")

if __name__ == "__main__":
    # Download both models
    print("Downloading and preparing models...")
    download_inception_v3()
    download_yolov8()
    print("Done!")
