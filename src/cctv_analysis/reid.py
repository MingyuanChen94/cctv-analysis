import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import os
import sys

def init_pretrained_weights(model, model_path):
    """Initializes model with pretrained weights."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded pretrained weights from {model_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")

class ConvLayer(torch.nn.Module):
    """Basic convolutional layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                                  padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class OSBlock(torch.nn.Module):
    """Omni-scale block"""
    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        
        mid_channels = out_channels // reduction

        self.conv1 = ConvLayer(in_channels, mid_channels, 1)
        self.conv2 = torch.nn.ModuleList()
        for t in range(T):
            self.conv2.append(
                torch.nn.Sequential(
                    ConvLayer(mid_channels, mid_channels, 3, stride=1, padding=1)
                )
            )
        self.conv3 = torch.nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + x2_t
        x2_norm = x2 / len(self.conv2)
        
        x3 = self.conv3(x2_norm)
        x3 = self.bn3(x3)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = x3 + identity
        out = self.relu(out)

        return out

class OSNet(torch.nn.Module):
    """OSNet architecture"""
    def __init__(self, num_classes, blocks=[2, 2, 2], channels=[64, 256, 384, 512], feature_dim=512):
        super(OSNet, self).__init__()
        self.feature_dim = feature_dim

        # Conv Layer 1
        self.conv1 = torch.nn.Sequential(
            ConvLayer(3, channels[0], 7, stride=2, padding=3),
            torch.nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Conv Layer 2-4
        self.conv2 = self._make_layer(channels[0], channels[1], blocks[0])
        self.conv3 = self._make_layer(channels[1], channels[2], blocks[1])
        self.conv4 = self._make_layer(channels[2], channels[3], blocks[2])

        # Global Average Pooling
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels[3], feature_dim),
            torch.nn.BatchNorm1d(feature_dim),
            torch.nn.ReLU(inplace=True)
        )
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(OSBlock(in_channels, out_channels))
        for i in range(1, blocks):
            layers.append(OSBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x, return_feats=False):
        x = self.featuremaps(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        feats = self.fc(x)
        
        if return_feats:
            return feats
            
        x = self.classifier(feats)
        return x

class PersonReID:
    def __init__(self, model_path=None, device="cuda"):
        """Initialize PersonReID model"""
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        # Initialize OSNet model
        self.model = OSNet(
            num_classes=751,  # Market-1501 dataset
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
    
    def preprocess_image(self, img):
        """Preprocess image from OpenCV format to model input"""
        if img is None or img.size == 0:
            return None
            
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Handle small images
            if img_rgb.shape[0] < 10 or img_rgb.shape[1] < 10:
                return None
                
            # Convert to PIL Image
            img_pil = Image.fromarray(img_rgb)
            
            # Apply transformations
            img_tensor = self.transform(img_pil)
            return img_tensor
            
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return None
        
    @torch.no_grad()
    def extract_features(self, img_crop):
        """Extract ReID features from a cropped person image"""
        try:
            if img_crop is None or img_crop.size == 0:
                return None
                
            # Preprocess image
            img_tensor = self.preprocess_image(img_crop)
            if img_tensor is None:
                return None
                
            # Add batch dimension
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
        """Compute cosine similarity between two feature vectors"""
        if feat1 is None or feat2 is None:
            return 0.0
            
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        return np.clip(np.dot(feat1_norm, feat2_norm), 0, 1)
