import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import cv2

@dataclass
class DemographicInfo:
    """Class to store demographic information."""
    gender: str
    age_group: str
    confidence: float

class DemographicAnalyzer:
    """Analyze demographics of detected persons."""
    
    def __init__(self, gender_model_path: str, age_model_path: str):
        """
        Initialize demographic analyzer with pre-trained models.
        
        Args:
            gender_model_path: Path to gender classification model
            age_model_path: Path to age estimation model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize gender classification model
        self.gender_model = self._load_gender_model(gender_model_path)
        self.gender_model.to(self.device)
        self.gender_model.eval()
        
        # Initialize age estimation model
        self.age_model = self._load_age_model(age_model_path)
        self.age_model.to(self.device)
        self.age_model.eval()
        
        # Define age groups
        self.age_groups = [
            (0, 12, "child"),
            (13, 19, "teenager"),
            (20, 39, "young_adult"),
            (40, 59, "adult"),
            (60, float('inf'), "senior")
        ]
        
        # Setup image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _load_gender_model(self, model_path: str) -> nn.Module:
        """Load gender classification model."""
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 2)  # Binary classification
        model.load_state_dict(torch.load(model_path))
        return model
        
    def _load_age_model(self, model_path: str) -> nn.Module:
        """Load age estimation model."""
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(512, 1)  # Regression
        model.load_state_dict(torch.load(model_path))
        return model
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        image = Image.fromarray(image)
        image = self.transform(image)
        return image.unsqueeze(0)
    
    def analyze_person(self, image: np.ndarray) -> Optional[DemographicInfo]:
        """
        Analyze demographics for a single person.
        
        Args:
            image: Cropped person image
            
        Returns:
            DemographicInfo object or None if analysis fails
        """
        try:
            # Preprocess image
            tensor = self._preprocess_image(image).to(self.device)
            
            with torch.no_grad():
                # Gender classification
                gender_output = self.gender_model(tensor)
                gender_probs = torch.softmax(gender_output, dim=1)
                gender_idx = torch.argmax(gender_probs).item()
                gender = "male" if gender_idx == 0 else "female"
                gender_conf = gender_probs[0, gender_idx].item()
                
                # Age estimation
                age_output = self.age_model(tensor)
                estimated_age = age_output.item()
                
                # Determine age group
                age_group = "unknown"
                for min_age, max_age, group_name in self.age_groups:
                    if min_age <= estimated_age <= max_age:
                        age_group = group_name
                        break
                
            return DemographicInfo(
                gender=gender,
                age_group=age_group,
                confidence=gender_conf
            )
            
        except Exception as e:
            print(f"Error in demographic analysis: {str(e)}")
            return None
    
    def analyze_batch(self, images: List[np.ndarray]) -> List[Optional[DemographicInfo]]:
        """
        Analyze demographics for a batch of person images.
        
        Args:
            images: List of cropped person images
            
        Returns:
            List of DemographicInfo objects
        """
        results = []
        try:
            # Preprocess images
            tensors = torch.stack([self._preprocess_image(img) for img in images])
            tensors = tensors.to(self.device)
            
            with torch.no_grad():
                # Gender classification
                gender_outputs = self.gender_model(tensors)
                gender_probs = torch.softmax(gender_outputs, dim=1)
                gender_indices = torch.argmax(gender_probs, dim=1)
                gender_confs = torch.gather(gender_probs, 1, gender_indices.unsqueeze(1))
                
                # Age estimation
                age_outputs = self.age_model(tensors)
                
                # Process each result
                for i in range(len(images)):
                    gender = "male" if gender_indices[i].item() == 0 else "female"
                    estimated_age = age_outputs[i].item()
                    
                    # Determine age group
                    age_group = "unknown"
                    for min_age, max_age, group_name in self.age_groups:
                        if min_age <= estimated_age <= max_age:
                            age_group = group_name
                            break
                    
                    results.append(DemographicInfo(
                        gender=gender,
                        age_group=age_group,
                        confidence=gender_confs[i].item()
                    ))
                    
        except Exception as e:
            print(f"Error in batch demographic analysis: {str(e)}")
            results.extend([None] * len(images))
            
        return results
    
    def get_demographics_statistics(self, demographics: List[DemographicInfo]) -> Dict:
        """
        Generate statistics from demographic analysis results.
        
        Args:
            demographics: List of DemographicInfo objects
            
        Returns:
            Dictionary containing demographic statistics
        """
        stats = {
            "gender_distribution": defaultdict(int),
            "age_distribution": defaultdict(int),
            "gender_age_distribution": defaultdict(lambda: defaultdict(int))
        }
        
        valid_demographics = [d for d in demographics if d is not None]
        total_count = len(valid_demographics)
        
        if total_count == 0:
            return stats
        
        # Calculate distributions
        for demo in valid_demographics:
            stats["gender_distribution"][demo.gender] += 1
            stats["age_distribution"][demo.age_group] += 1
            stats["gender_age_distribution"][demo.gender][demo.age_group] += 1
        
        # Convert to percentages
        for gender in stats["gender_distribution"]:
            stats["gender_distribution"][gender] /= total_count
            
        for age in stats["age_distribution"]:
            stats["age_distribution"][age] /= total_count
            
        for gender in stats["gender_age_distribution"]:
            for age in stats["gender_age_distribution"][gender]:
                stats["gender_age_distribution"][gender][age] /= total_count
        
        return stats
