import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import os
import sys

def init_pretrained_weights(model, model_path):
    """Initializes model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading weights from {model_path}: {e}")
        return

    try:
        state_dict = checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded pretrained weights from {model_path}")
    except Exception as e:
        print(f"Error loading state dict: {e}")

class OSBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0

        mid_channels = out_channels // reduction

        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        self.conv2 = torch.nn.ModuleList()
        for t in range(T):
            self.conv2.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1, bias=False),
                    torch.nn.BatchNorm2d(mid_channels),
                    torch.nn.ReLU(inplace=True)
                )
            )
        self.conv3 = torch.nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        for conv2_t in self.conv2:
            out = conv2_t(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class OSNet(torch.nn.Module):
    def __init__(self, num_classes, blocks=[2, 2, 2], channels=[64, 256, 384, 512], feature_dim=512):
        super(OSNet, self).__init__()
        self.feature_dim = feature_dim

        # Conv Layer 1
        self.conv1 = torch.nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels[0])
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)

        # OSNet Blocks
        self.layers = torch.nn.ModuleList([])
        
        # Conv Layer 2
        layer2 = []
        in_channels = channels[0]
        for i in range(blocks[0]):
            layer2.append(OSBlock(in_channels, channels[1]))
            in_channels = channels[1]
        self.layers.append(torch.nn.Sequential(*layer2))
        
        # Conv Layer 3
        layer3 = []
        for i in range(blocks[1]):
            layer3.append(OSBlock(channels[1], channels[2]))
        self.layers.append(torch.nn.Sequential(*layer3))
        
        # Conv Layer 4
        layer4 = []
        for i in range(blocks[2]):
            layer4.append(OSBlock(channels[2], channels[3]))
        self.layers.append(torch.nn.Sequential(*layer4))

        # Global Average Pooling
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.fc = torch.nn.Linear(channels[3], feature_dim)
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_feats=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        feats = self.fc(x)
        
        if return_feats:
            return feats
            
        x = self.classifier(feats)
        return x

class PersonReID:
    def __init__(self, model_path=None, device="cuda"):
        """
        Initialize PersonReID model
        Args:
            model_path: Path to pretrained weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set cudnn to benchmark mode for faster convolutions
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = "cpu"
            print("GPU not available, using CPU")
        
        # Initialize model for Market-1501 dataset (751 identities)
        self.model = OSNet(
            num_classes=751,
            blocks=[2, 2, 2],
            channels=[64, 256, 384, 512],
            feature_dim=512
        )
        
        if model_path:
            init_pretrained_weights(self.model, model_path)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = T.Compose([
            T.Resize((256, 128)),  # Standard size for person ReID
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, img_array):
        """
        Preprocess image from OpenCV format to model input
        Args:
            img_array: numpy array in BGR format (OpenCV default)
        Returns:
            torch tensor
        """
        if img_array is None or img_array.size == 0:
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        img_pil = Image.fromarray(img_rgb)
        # Apply transformations
        img_tensor = self.transform(img_pil)
        return img_tensor
        
    @torch.no_grad()
    def extract_features(self, img_crop):
        """
        Extract ReID features from a cropped person image
        Args:
            img_crop: numpy array of cropped person image (BGR format)
        Returns:
            feature vector (numpy array)
        """
        try:
            if img_crop is None or img_crop.size == 0:
                return None
                
            # Ensure minimum size
            if img_crop.shape[0] < 10 or img_crop.shape[1] < 10:
                return None
                
            # Preprocess image
            img_tensor = self.preprocess_image(img_crop)
            if img_tensor is None:
                return None
                
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Extract features
            features = self.model(img_tensor, return_feats=True)
            
            # Normalize feature vector
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None
        
    def compute_similarity(self, feat1, feat2):
        """
        Compute cosine similarity between two feature vectors
        Args:
            feat1, feat2: numpy arrays of same dimension
        Returns:
            float: similarity score between 0 and 1
        """
        if feat1 is None or feat2 is None:
            return 0.0
            
        # Ensure features are normalized
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        return np.clip(np.dot(feat1_norm, feat2_norm), 0, 1)