import os
import cv2
import torch
import torchreid
import datetime
import numpy as np
import psutil
from pathlib import Path
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count, get_context
import platform
from typing import Dict, List, Tuple, Optional, Union
import queue
from threading import Thread

# Try to import GPUtil but handle if not installed
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class DeviceManager:
    """Manages hardware device selection and optimization across platforms."""
    
    def __init__(self):
        self.platform = platform.system()
        self.device = self._select_device()
        self.can_parallelize = self._check_parallelization()
        
    def _select_device(self) -> torch.device:
        """Select the optimal available device for the current platform."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon (MPS) acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU for computation")
        return device
    
    def _check_parallelization(self) -> bool:
        """Check if parallelization is supported on the current platform/device."""
        if self.device.type == "cuda":
            return True  # CUDA supports multi-GPU and CPU parallel processing
        elif self.device.type == "mps":
            # MPS currently has limitations with multiprocessing
            return self.platform != "Darwin"
        return True  # CPU can always use parallel processing
    
    def get_optimal_workers(self) -> int:
        """Return the optimal number of worker processes for the current setup."""
        if not self.can_parallelize:
            return 1
        if self.device.type == "cuda":
            return min(torch.cuda.device_count() * 2, cpu_count() - 1)
        return max(1, cpu_count() - 1)


class ResourceManager:
    """Manages system resource detection and allocation."""
    
    def __init__(self):
        self.platform = platform.system()
        self.memory = self._get_memory_info()
        self.cpu_info = self._get_cpu_info()
        self.gpu_info = self._get_gpu_info()
        self.device = self._select_optimal_device()
        
    def _get_memory_info(self) -> Dict:
        """Get system memory information."""
        vm = psutil.virtual_memory()
        return {
            'total': vm.total,
            'available': vm.available,
            'used': vm.used,
            'percent': vm.percent,
            # Reserve 2GB or 10% of RAM, whichever is larger
            'safe_usage_limit': min(vm.available - max(2 * 1024**3, vm.total * 0.1), vm.available * 0.9)
        }
    
    def _get_cpu_info(self) -> Dict:
        """Get CPU information and determine optimal thread count."""
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Reserve 1 core or 10% of cores, whichever is larger
        reserved_cores = max(1, int(cpu_count * 0.1))
        optimal_workers = max(1, cpu_count - reserved_cores)
        
        return {
            'cpu_count': cpu_count,
            'physical_cores': psutil.cpu_count(logical=False),
            'frequency': cpu_freq.current if cpu_freq else None,
            'usage_percent': cpu_percent,
            'optimal_workers': optimal_workers,
            'reserved_cores': reserved_cores
        }
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information if available."""
        gpu_info = {'available': False, 'devices': []}
        
        try:
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['device_count'] = torch.cuda.device_count()
                
                # Get GPU information using torch if GPUtil is not available
                if GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_info['devices'].append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_total': gpu.memoryTotal,
                            'memory_used': gpu.memoryUsed,
                            'memory_free': gpu.memoryFree,
                            'load': gpu.load
                        })
                else:
                    # Use torch.cuda for basic GPU information
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**2  # Convert to MB
                        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                        memory_free = total_memory - memory_allocated
                        
                        gpu_info['devices'].append({
                            'id': i,
                            'name': props.name,
                            'memory_total': total_memory,
                            'memory_used': memory_allocated,
                            'memory_free': memory_free,
                            'load': memory_allocated / total_memory  # Approximate load based on memory usage
                        })
                        
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['available'] = True
                gpu_info['type'] = 'mps'
                gpu_info['device_count'] = 1
                
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            # Provide fallback GPU information
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['devices'] = [{
                    'id': 0,
                    'name': torch.cuda.get_device_name(0),
                    'memory_total': 0,  # Unknown
                    'memory_free': 0,   # Unknown
                    'memory_used': 0,   # Unknown
                    'load': 0.5         # Assume 50% load as fallback
                }]
            
        return gpu_info
    
    def _select_optimal_device(self) -> torch.device:
        """Select the optimal device based on available resources."""
        if self.gpu_info['available']:
            if 'cuda_version' in self.gpu_info:
                # For CUDA devices, select the GPU with most free memory
                if self.gpu_info['devices']:
                    best_gpu = max(self.gpu_info['devices'], 
                                 key=lambda x: x['memory_free'])
                    torch.cuda.set_device(best_gpu['id'])
                    print(f"Using CUDA GPU {best_gpu['id']}: {best_gpu['name']}")
                return torch.device("cuda")
            else:
                print("Using Apple Silicon (MPS)")
                return torch.device("mps")
        return torch.device("cpu")
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        if self.device.type == "cuda":
            # For CUDA, use GPU memory
            gpu = self.gpu_info['devices'][torch.cuda.current_device()]
            available_memory = gpu['memory_free'] * 0.8  # Use 80% of free memory
            # Approximate memory per frame (adjust based on your model's needs)
            memory_per_frame = 0.1  # GB
            return max(1, int(available_memory / memory_per_frame))
        else:
            # For CPU/MPS, use system memory
            available_memory = self.memory['safe_usage_limit'] / (1024**3)  # Convert to GB
            memory_per_frame = 0.05  # GB
            return max(1, int(available_memory / memory_per_frame))
    
    def get_optimal_workers(self) -> int:
        """Get optimal number of worker processes."""
        if self.device.type == "cuda":
            # For CUDA, optimize for GPU count and CPU cores
            return min(self.gpu_info['device_count'] * 2, self.cpu_info['optimal_workers'])
        elif self.device.type == "mps":
            # For MPS, consider CPU cores but limit due to Metal limitations
            return min(4, self.cpu_info['optimal_workers'])
        else:
            # For CPU, use optimal workers from CPU info
            return self.cpu_info['optimal_workers']
    
    def print_resource_info(self):
        """Print detailed resource information."""
        print("\nSystem Resource Information:")
        print(f"Platform: {self.platform}")
        print("\nMemory:")
        print(f"Total: {self.memory['total'] / (1024**3):.1f} GB")
        print(f"Available: {self.memory['available'] / (1024**3):.1f} GB")
        print(f"Used: {self.memory['percent']}%")
        
        print("\nCPU:")
        print(f"Total Cores: {self.cpu_info['cpu_count']}")
        print(f"Physical Cores: {self.cpu_info['physical_cores']}")
        print(f"Current Usage: {self.cpu_info['usage_percent']}%")
        print(f"Optimal Workers: {self.cpu_info['optimal_workers']}")
        
        if self.gpu_info['available']:
            print("\nGPU:")
            if 'cuda_version' in self.gpu_info:
                print(f"CUDA Version: {self.gpu_info['cuda_version']}")
                for gpu in self.gpu_info['devices']:
                    print(f"\nGPU {gpu['id']}: {gpu['name']}")
                    print(f"Memory: {gpu['memory_free']}/{gpu['memory_total']} MB free")
                    print(f"Load: {gpu['load']*100:.1f}%")
            else:
                print("Type: Apple Silicon (MPS)")


class VideoAnalyzer:
    def __init__(self, output_dir="output", num_workers=None):
        """Initialize the video analysis system with resource optimization."""
        self.device_manager = DeviceManager()
        self.resource_manager = ResourceManager()
        self.resource_manager.print_resource_info()
        
        self.num_workers = num_workers or self.device_manager.get_optimal_workers()
        self.device = self.device_manager.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize models with proper device placement
        self.reid_model, self.yolo_model = self._initialize_models()
        
        # Setup debugging and output directories
        self._setup_directories()
        
        # Configure transform pipeline
        self.transform = self._setup_transform_pipeline()
        
        # Analysis parameters
        self.params = self._setup_analysis_parameters()

    def _setup_directories(self):
        """Setup output and debug directories."""
        self.debug_dir = self.output_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)
        (self.debug_dir / "frames").mkdir(exist_ok=True)
        (self.debug_dir / "detections").mkdir(exist_ok=True)

    def _setup_transform_pipeline(self):
        """Setup the image transform pipeline."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _setup_analysis_parameters(self):
        """Setup analysis parameters with resource-aware optimizations."""
        # Get optimal batch size based on available memory
        optimal_batch = self.resource_manager.get_optimal_batch_size()
        
        base_params = {
            'confidence_threshold': 0.3,
            'similarity_threshold': 0.5,
            'max_time_diff': 600,
            'min_detection_area': 2000,
            'batch_size': optimal_batch
        }
        
        # Adjust frame skipping based on hardware and load
        if self.device.type == "cuda":
            gpu = self.resource_manager.gpu_info['devices'][torch.cuda.current_device()]
            if gpu['load'] > 0.8:  # High GPU load
                base_params['skip_frames'] = 2
            else:
                base_params['skip_frames'] = 1
        elif self.device.type == "mps":
            cpu_load = self.resource_manager.cpu_info['usage_percent']
            base_params['skip_frames'] = 2 if cpu_load > 70 else 1
        else:
            cpu_load = self.resource_manager.cpu_info['usage_percent']
            base_params['skip_frames'] = 4 if cpu_load > 70 else 2
            
        return base_params

    def verify_video(self, video_path: Path) -> Dict:
        """Verify video file and return its properties."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        file_size = video_path.stat().st_size / (1024 * 1024)  # Size in MB
        if file_size < 1:
            print(f"Warning: Video file suspiciously small ({file_size:.2f} MB): {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        props = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'file_size': file_size
        }
        props['duration'] = props['frame_count'] / props['fps'] if props['fps'] > 0 else 0

        cap.release()
        return props

    def _initialize_models(self):
        """Initialize models with platform-specific optimizations."""
        print("\nInitializing models...")
        
        # Initialize ReID model
        reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        reid_model = reid_model.to(self.device)
        reid_model.eval()

        # Initialize YOLO model with platform-specific settings
        yolo_model = YOLO("yolov8x.pt")
        
        # Configure YOLO based on platform
        if self.device.type != "cpu":
            yolo_model.to(self.device)
        
        print("Models initialized successfully")
        return reid_model, yolo_model

    def _process_batch(self, frames: List[np.ndarray], video_name: str) -> Tuple[List, List]:
        """Process a batch of frames in parallel."""
        all_features = []
        all_timestamps = []
        
        for idx, frame in enumerate(frames):
            features, timestamps = self.process_frame(frame, idx, video_name)
            all_features.extend(features)
            all_timestamps.extend(timestamps)
            
        return all_features, all_timestamps

    def analyze_video(self, video_path: Path) -> Dict:
        """Analyze video with platform-optimized parallel processing."""
        print(f"\nAnalyzing video: {video_path}")
        
        # Verify video
        props = self.verify_video(video_path)
        if props['frame_count'] == 0:
            raise RuntimeError("Video appears to be empty")

        # Determine optimal batch size based on platform
        batch_size = self._get_optimal_batch_size()
        
        # Setup parallel processing based on platform capabilities
        if self.device_manager.can_parallelize:
            return self._parallel_video_analysis(video_path, props, batch_size)
        else:
            return self._sequential_video_analysis(video_path, props)

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on hardware."""
        if self.device_manager.device.type == "cuda":
            return 32  # Larger batches for GPU
        elif self.device_manager.device.type == "mps":
            return 16  # Medium batches for Apple Silicon
        return 8  # Smaller batches for CPU

    def process_frame(self, frame: np.ndarray, frame_count: int, video_name: str) -> Tuple[List, List]:
        """Process a single frame for person detection and feature extraction."""
        results = self.yolo_model(frame, conf=self.params['confidence_threshold'])
        detections = results[0].boxes.data

        frame_features = []
        frame_timestamps = []
        debug_frame = frame.copy() if frame_count % 100 == 0 else None

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()

            # Filter for person class
            if cls != 0:
                continue

            # Calculate detection metrics
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0

            # Filter unrealistic detections
            if aspect_ratio < 0.1 or aspect_ratio > 1.0 or area < self.params['min_detection_area']:
                continue

            # Extract person crop
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size > 0:
                features = self.extract_reid_features(crop)
                if features is not None:
                    frame_features.append(features)
                    frame_timestamps.append(frame_count)

                    # Save debug visualization
                    if debug_frame is not None:
                        cv2.rectangle(debug_frame, (int(x1), int(y1)),
                                    (int(x2), int(y2)), (0, 255, 0), 2)

        # Save debug frame
        if debug_frame is not None and len(frame_features) > 0:
            debug_path = self.debug_dir / "frames" / f"{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)

        return frame_features, frame_timestamps

    def extract_reid_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract ReID features from a person detection."""
        if image.shape[0] < 50 or image.shape[1] < 25:
            return None

        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.reid_model(input_tensor)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                features_np = features.cpu().numpy().squeeze()

                # Validate features
                if np.isnan(features_np).any() or np.sum(features_np) == 0:
                    return None

                return features_np
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None

    def _sequential_video_analysis(self, video_path: Path, props: Dict) -> Dict:
        """Process video sequentially when parallel processing is not available."""
        cap = cv2.VideoCapture(str(video_path))
        all_features = []
        all_timestamps = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.params['skip_frames'] != 0:
                continue

            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{props['frame_count']}")

            features, timestamps = self.process_frame(frame, frame_count, video_path.stem)
            all_features.extend(features)
            all_timestamps.extend(timestamps)

        cap.release()
        print(f"Extracted {len(all_features)} feature vectors")

        return {
            'features': all_features,
            'timestamps': [t / props['fps'] for t in all_timestamps],
            'video_props': props
        }

    def _parallel_video_analysis(self, video_path: Path, props: Dict, batch_size: int) -> Dict:
        """Perform parallel video analysis with platform-specific optimizations."""
        frames_queue = queue.Queue(maxsize=batch_size * 2)
        results_queue = queue.Queue()
        
        # Start frame reading thread
        read_thread = Thread(target=self._read_frames_worker,
                           args=(video_path, frames_queue, props, batch_size))
        read_thread.daemon = True
        read_thread.start()
        
        # Process frames in parallel
        with get_context('spawn').Pool(self.num_workers) as pool:
            while True:
                try:
                    batch = frames_queue.get(timeout=60)
                    if batch is None:  # End signal
                        break
                    
                    # Process batch
                    result = pool.apply_async(self._process_batch,
                                           args=(batch, video_path.stem))
                    results_queue.put(result)
                    
                except queue.Empty:
                    print("Timeout waiting for frames")
                    break
        
        # Collect results
        all_features = []
        all_timestamps = []
        
        while not results_queue.empty():
            result = results_queue.get()
            features, timestamps = result.get()
            all_features.extend(features)
            all_timestamps.extend(timestamps)
        
        return {
            'features': all_features,
            'timestamps': [t / props['fps'] for t in all_timestamps],
            'video_props': props
        }

    def _read_frames_worker(self, video_path: Path, frames_queue: queue.Queue,
                          props: Dict, batch_size: int):
        """Worker thread for reading video frames."""
        cap = cv2.VideoCapture(str(video_path))
        batch = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if len(batch) < batch_size:
                batch.append(frame)
            else:
                frames_queue.put(batch)
                batch = []
                
        # Put remaining frames
        if batch:
            frames_queue.put(batch)
            
        # Signal end
        frames_queue.put(None)
        cap.release()

    def analyze_folder(self, folder_path: str) -> Dict:
        """Analyze all videos in a folder with platform-optimized parallel processing."""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        videos = self._get_videos(folder_path)
        camera_data = self._process_videos(videos)
        matches = self._match_across_cameras(camera_data)
        
        return self._save_results(matches, camera_data)

    def _get_videos(self, folder_path: Path) -> Dict[str, List[Path]]:
        """Get and organize videos by camera."""
        videos = list(folder_path.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No .mp4 files found in {folder_path}")

        camera_videos = {"Camera_1": [], "Camera_2": []}
        for video in videos:
            for camera in camera_videos:
                if video.name.startswith(camera):
                    camera_videos[camera].append(video)
                    break
        
        return camera_videos

    def _match_across_cameras(self, camera_data: Dict) -> List[Dict]:
        """Match detections between cameras."""
        all_matches = []
        
        for video1, data1 in camera_data["Camera_1"].items():
            for video2, data2 in camera_data["Camera_2"].items():
                matches = self.match_features(data1, data2, video1, video2)
                all_matches.extend(matches)
                
        return all_matches

    def match_features(self, data1: Dict, data2: Dict, video1_name: str, video2_name: str) -> List[Dict]:
        """Match person features between two videos."""
        features1 = np.array(data1['features'])
        features2 = np.array(data2['features'])
        times1 = np.array(data1['timestamps'])
        times2 = np.array(data2['timestamps'])

        if len(features1) == 0 or len(features2) == 0:
            return []

        print(f"\nMatching features between {video1_name} and {video2_name}")
        print(f"Features: {len(features1)} vs {len(features2)}")

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(features1, features2)
        print(f"Similarity stats - Min: {similarity_matrix.min():.3f}, "
              f"Max: {similarity_matrix.max():.3f}, Mean: {similarity_matrix.mean():.3f}")

        # Find matches
        matches = []
        high_similarity_pairs = np.where(
            similarity_matrix > self.params['similarity_threshold'])

        for idx1, idx2 in zip(*high_similarity_pairs):
            similarity = similarity_matrix[idx1, idx2]
            time_diff = abs(times1[idx1] - times2[idx2])

            if time_diff < self.params['max_time_diff']:
                matches.append({
                    'video1': video1_name,
                    'video2': video2_name,
                    'frame1': float(times1[idx1]),
                    'frame2': float(times2[idx2]),
                    'similarity': float(similarity),
                    'time_difference': float(time_diff)
                })

        matches.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"Found {len(matches)} matches")
        return matches

    def _process_videos(self, camera_videos: Dict[str, List[Path]]) -> Dict:
        """Process videos with platform-specific optimizations."""
        camera_data = {"Camera_1": {}, "Camera_2": {}}
        
        for camera, video_list in camera_videos.items():
            if self.device_manager.can_parallelize:
                with get_context('spawn').Pool(self.num_workers) as pool:
                    results = pool.map(self.analyze_video, video_list)
                    for video, result in zip(video_list, results):
                        camera_data[camera][video.name] = result
            else:
                for video in video_list:
                    camera_data[camera][video.name] = self.analyze_video(video)
                    
        return camera_data

    def _save_results(self, matches: List[Dict], camera_data: Dict) -> Dict:
        """Save analysis results with platform-specific metadata."""
        output_data = {
            "matches": matches,
            "metadata": {
                "total_matches": len(matches),
                "timestamp": str(datetime.datetime.now()),
                "parameters": self.params,
                "platform": {
                    "system": platform.system(),
                    "device": self.device_manager.device.type,
                    "num_workers": self.num_workers,
                    "can_parallelize": self.device_manager.can_parallelize
                },
                "videos_analyzed": {
                    camera: list(data.keys())
                    for camera, data in camera_data.items()
                }
            }
        }

        output_path = self.output_dir / "results.json"
        with open(output_path, 'w') as f:
            import json
            json.dump(output_data, f, indent=4)

        print(f"\nAnalysis complete. Results saved to {output_path}")
        return output_data


def main():
    # Create analyzer instance
    analyzer = VideoAnalyzer(output_dir="reid_output")

    # Set input folder path
    folder_path = os.path.join(os.path.expanduser("~"), "Library", "CloudStorage", 
                              "OneDrive-UniversityofExeter", "Documents", "VISIONARY", 
                              "Durham Experiment", "processed_data_3")

    try:
        results = analyzer.analyze_folder(folder_path)
        print(f"\nFound {len(results['matches'])} total matches")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
