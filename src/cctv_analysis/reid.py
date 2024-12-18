import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import List, Union
import numpy as np
from torchreid.models import build_model
from pathlib import Path

class PersonReID:
    """Person re-identification module using OSNet."""
    
    def __init__(self, model_path: str = "models/reid/osnet_x1_0.pth"):
        """
        Initialize the ReID model.
        
        Args:
            model_path: Path to OSNet model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = build_model(
            name='osnet_x1_0',
            num_classes=1000,  # Number of training identities
            pretrained=False
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup image preprocessing
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, images: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """
        Extract features from person patches.
        
        Args:
            images: Single image or list of images (RGB format)
            
        Returns:
            Tensor of features
        """
        if isinstance(images, np.ndarray):
            images = [images]
            
        # Preprocess images
        processed = []
        for img in images:
            # Convert numpy array to PIL Image
            img = Image.fromarray(img)
            processed.append(self.transform(img))
            
        # Stack all images into a batch
        batch = torch.stack(processed).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            
        return features.cpu()
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two sets of features.
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            Similarity matrix
        """
        # Normalize features
        features1 = nn.functional.normalize(features1, dim=1)
        features2 = nn.functional.normalize(features2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(features1, features2.t())
        
        return similarity
        
    def get_distance_matrix(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance matrix between two sets of features.
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            Distance matrix
        """
        similarity = self.compute_similarity(features1, features2)
        distance = 2 - 2 * similarity  # Convert cosine similarity to euclidean distance
        return distance
