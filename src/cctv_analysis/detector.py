import torch
from yolox.exp import Exp
from yolox.data.datasets import COCO_CLASSES
import numpy as np
import cv2

class YOLOXExp(Exp):
    def __init__(self, model_size='l'):
        super().__init__()
        # Model size configurations
        size_configs = {
            's': {'depth': 0.33, 'width': 0.50, 'input_size': (640, 640)},
            'm': {'depth': 0.67, 'width': 0.75, 'input_size': (640, 640)},
            'l': {'depth': 1.0,  'width': 1.0,  'input_size': (640, 640)},
            'x': {'depth': 1.33, 'width': 1.25, 'input_size': (640, 640)}
        }
        
        if model_size not in size_configs:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(size_configs.keys())}")
            
        config = size_configs[model_size]
        
        # Basic exp attributes
        self.num_classes = 80  # COCO has 80 classes
        self.depth = config['depth']
        self.width = config['width']
        self.input_size = config['input_size']
        self.random_size = (10, 20)
        self.test_size = self.input_size
        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1

class PersonDetector:
    def __init__(self, model_path, model_size='l', device="cuda"):
        """
        Initialize YOLOX detector
        Args:
            model_path: Path to model weights
            model_size: YOLOX model size ('s', 'm', 'l', or 'x')
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cudnn benchmark for faster training
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = "cpu"
            print("GPU not available, using CPU")

        # Create experiment and model
        try:
            self.exp = YOLOXExp(model_size)
            self.model = self.exp.get_model()
            
            # Load weights
            ckpt = torch.load(model_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            self.model.load_state_dict(ckpt)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Successfully loaded YOLOX-{model_size.upper()} model")
            
        except Exception as e:
            raise RuntimeError(f"Error initializing YOLOX model: {e}")
        
        # Model settings
        self.input_size = self.exp.input_size
        self.num_classes = self.exp.num_classes
        self.confthre = 0.01  # Lowered confidence threshold
        self.nmsthre = 0.65   # Adjusted NMS threshold
        
    def preprocess(self, img, input_size=(640, 640)):
        """
        Preprocess image for inference
        Args:
            img: OpenCV image in BGR format
            input_size: Model input size
        Returns:
            Preprocessed tensor and resize ratio
        """
        # Convert image to RGB first
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate padding size
        ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        new_h, new_w = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
        
        # Resize image
        resized = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        
        # Create padded image
        padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        padded[:new_h, :new_w] = resized
        
        # Convert to tensor and normalize
        padded = padded.transpose(2, 0, 1)
        padded = torch.from_numpy(padded).float().div(255.0)
        padded = padded.unsqueeze(0)
        
        return padded, ratio
        
    def postprocess(self, prediction, num_classes, conf_thre=0.1, nms_thre=0.45):
        """
        Postprocess the model output
        """
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            if not image_pred.size(0):
                continue
                
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5: 5 + num_classes], 1, keepdim=True
            )
            
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            
            if not detections.size(0):
                continue
                
            nms_out_index = torch.ops.torchvision.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
            detections = detections[nms_out_index]
            output[i] = detections

        return output

    @torch.no_grad()
    def detect(self, img, conf_thresh=0.3):
        """
        Detect persons in image
        Args:
            img: OpenCV image in BGR format
            conf_thresh: Confidence threshold
        Returns:
            List of detections, each in (x1, y1, x2, y2, confidence) format
        """
        try:
            # Check for valid image
            if img is None or img.size == 0:
                print("Invalid image input")
                return []
                
            # Preprocess image
            img_processed, ratio = self.preprocess(img, self.input_size)
            img_processed = img_processed.to(self.device)
            
            # Run inference
            outputs = self.model(img_processed)
            
            # Postprocess
            outputs = self.postprocess(
                outputs,
                self.num_classes,
                conf_thresh,
                self.nmsthre
            )
            
            if outputs[0] is None:
                return []
                
            outputs = outputs[0].cpu().numpy()
            
            # Filter only person class (class 0 in COCO)
            person_dets = outputs[outputs[:, 6] == 0]
            
            # Scale coordinates back to original image size
            person_dets[:, :4] = person_dets[:, :4] / ratio
            
            # Return bounding boxes and confidence scores
            detections = person_dets[:, :5]  # x1, y1, x2, y2, confidence
            
            # Additional filtering for more reliable detections
            valid_dets = []
            for det in detections:
                x1, y1, x2, y2, conf = det
                w = x2 - x1
                h = y2 - y1
                
                # Filter out too small or too large detections
                if w < 20 or h < 40 or w/h > 2 or h/w > 4:
                    continue
                    
                valid_dets.append(det)
            
            return np.array(valid_dets)
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
        
    def draw_detections(self, img, detections, show_conf=True, color=(0, 255, 0)):
        """
        Draw detection boxes on image
        Args:
            img: OpenCV image
            detections: List of detections from detect()
            show_conf: Whether to show confidence scores
            color: BGR color tuple for boxes
        Returns:
            Image with drawn detections
        """
        img_draw = img.copy()
        for det in detections:
            x1, y1, x2, y2, conf = map(float, det)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            if show_conf:
                label = f"person {conf:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    img_draw,
                    (x1, y1 - label_h - baseline),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                cv2.putText(
                    img_draw,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
        
        return img_draw