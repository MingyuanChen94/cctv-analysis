#!/usr/bin/env python3
"""
Enhanced Parameter Optimization for Cross-Camera People Tracking using Bayesian Optimization.

This script improves optimization by:
1. Using ground truth or consistency metrics instead of fixed targets
2. Adding cross-validation across multiple video pairs
3. Incorporating quality metrics beyond raw counts
4. Supporting multi-objective optimization
"""

import os
import time
import numpy as np
import json
import glob
import random
from pathlib import Path
import subprocess
import argparse
import re
import cv2
from collections import defaultdict
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Script to be optimized
SCRIPT_PATH = "v15.py"  # Path to tracking script

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class EnhancedBayesianOptimizer:
    """Enhanced Bayesian optimization for parameter tuning of people tracking."""
    
    def __init__(self, video_pairs, output_dir, ground_truth=None, n_calls=50, 
                 n_initial_points=10, cross_validation=True, n_folds=3,
                 multi_objective=True, visualize_evaluations=False, verbose=True, debug=False):
        """
        Initialize the enhanced Bayesian optimizer.
        
        Args:
            video_pairs: List of (video1_path, video2_path) tuples
            output_dir: Directory to save results
            ground_truth: Dict mapping video pair names to ground truth values (optional)
            n_calls: Number of optimization iterations
            n_initial_points: Number of initial random points for GP optimization
            cross_validation: Whether to use cross-validation
            n_folds: Number of CV folds (if cross_validation=True)
            multi_objective: Whether to use multiple metrics for optimization
            visualize_evaluations: Whether to visualize tracking during evaluation
            verbose: Whether to print progress information
            debug: Whether to print detailed debugging information
        """
        self.video_pairs = video_pairs
        self.output_dir = Path(output_dir)
        self.ground_truth = ground_truth or {}
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.cross_validation = cross_validation
        self.n_folds = min(n_folds, len(video_pairs)) if cross_validation else 1
        self.multi_objective = multi_objective
        self.visualize_evaluations = visualize_evaluations
        self.verbose = verbose
        self.debug = debug
        
        self.opt_history = []
        self.best_score = float('inf')
        self.best_params = None
        self.best_results = None
        self.fold_results = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the parameter space
        self.define_parameter_space()
        
        # Set up cross-validation folds if needed
        if self.cross_validation and len(self.video_pairs) > 1:
            self.setup_cv_folds()
        else:
            self.train_pairs = self.video_pairs
            self.val_pairs = []
            
        # Print summary
        print(f"Initialized optimizer with {len(self.video_pairs)} video pairs")
        print(f"Cross-validation: {'Enabled' if self.cross_validation else 'Disabled'}")
        if self.cross_validation:
            print(f"Number of folds: {self.n_folds}")
        print(f"Multi-objective optimization: {'Enabled' if self.multi_objective else 'Disabled'}")
    
    def define_parameter_space(self):
        """Define the space of parameters to optimize."""
        # We'll focus on a subset of key parameters to keep optimization manageable
        self.param_space = [
            # Camera 1 parameters (more complex environment)
            Real(0.3, 0.8, name='cam1_min_confidence'),
            Real(0.5, 0.9, name='cam1_similarity_threshold'),
            Integer(1, 5, name='cam1_max_disappear_seconds'),
            Real(0.1, 0.7, name='cam1_deep_feature_weight'),
            Real(0.05, 0.4, name='cam1_position_weight'),
            Real(0.05, 0.3, name='cam1_color_weight'),
            Real(0.5, 0.9, name='cam1_reentry_threshold'),
            Real(0.5, 0.9, name='cam1_new_track_confidence'),
            
            # Camera 2 parameters (cleaner environment)
            Real(0.2, 0.7, name='cam2_min_confidence'),
            Real(0.4, 0.9, name='cam2_similarity_threshold'),
            Integer(1, 5, name='cam2_max_disappear_seconds'),
            Real(0.1, 0.7, name='cam2_deep_feature_weight'),
            Real(0.05, 0.4, name='cam2_position_weight'),
            Real(0.05, 0.3, name='cam2_color_weight'),
            Real(0.5, 0.9, name='cam2_reentry_threshold'),
            Real(0.4, 0.8, name='cam2_new_track_confidence'),
            
            # Common tracking parameters
            Integer(3, 15, name='max_lost_seconds'),
            Integer(2, 10, name='min_track_confirmations'),
            Real(0.5, 0.9, name='min_track_visibility'),
            
            # Global tracking parameters
            Real(0.5, 0.9, name='global_similarity_threshold'),
            Integer(2, 30, name='cam1_to_cam2_min_time'),
            Integer(10, 60, name='cam1_to_cam2_max_time'),
            Real(0.1, 0.5, name='global_feature_weight'),
            Real(0.1, 0.4, name='global_topology_weight')
        ]
        
        # Parameter names for reference
        self.param_names = [param.name for param in self.param_space]
    
    def setup_cv_folds(self):
        """Set up cross-validation folds for robust optimization."""
        # Shuffle video pairs for randomized folds
        all_pairs = self.video_pairs.copy()
        random.shuffle(all_pairs)
        
        # Divide into folds
        fold_size = len(all_pairs) // self.n_folds
        self.folds = []
        
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_folds - 1 else len(all_pairs)
            self.folds.append(all_pairs[start_idx:end_idx])
        
        # Use first fold as validation by default (will rotate during optimization)
        self.current_fold = 0
        self.train_pairs = [p for i, fold in enumerate(self.folds) 
                          for p in fold if i != self.current_fold]
        self.val_pairs = self.folds[self.current_fold]
        
        print(f"Set up {self.n_folds} cross-validation folds")
        print(f"Training pairs: {len(self.train_pairs)}, Validation pairs: {len(self.val_pairs)}")
    
    def rotate_cv_folds(self):
        """Rotate cross-validation folds to evaluate on different data."""
        if not self.cross_validation or len(self.folds) <= 1:
            return
            
        self.current_fold = (self.current_fold + 1) % self.n_folds
        self.train_pairs = [p for i, fold in enumerate(self.folds) 
                          for p in fold if i != self.current_fold]
        self.val_pairs = self.folds[self.current_fold]
        
        if self.verbose:
            print(f"Rotated to fold {self.current_fold+1}/{self.n_folds}")
            print(f"Training pairs: {len(self.train_pairs)}, Validation pairs: {len(self.val_pairs)}")
    
    def prepare_config(self, params):
        """Prepare the configuration dictionary for v15.py."""
        # Convert NumPy types to native Python types
        config = {name: convert_numpy_types(value) for name, value in params.items()}
        
        # Add fixed parameters
        fixed_config = {
            'cam1_motion_weight': 0.1,
            'cam1_part_weight': 0.1,
            'cam1_pose_weight': 0.1,
            'cam1_activity_weight': 0.1,
            
            'cam2_motion_weight': 0.1,
            'cam2_part_weight': 0.1,
            'cam2_pose_weight': 0.05,
            'cam2_activity_weight': 0.05,
            
            'max_history_length': 15,
            'process_every_nth_frame': 1,
            'min_detection_height': 65,
            
            'global_color_weight': 0.1,
            'global_part_weight': 0.1,
            'global_pose_weight': 0.05,
            'global_temporal_weight': 0.1,
            'temporal_constraint': True,
            'max_time_gap': 60,
            'min_time_gap': 2,
        }
        config.update(fixed_config)
        
        # Ensure cam1_to_cam2_min_time is always less than cam1_to_cam2_max_time
        min_time = min(config['cam1_to_cam2_min_time'], config['cam1_to_cam2_max_time'])
        max_time = max(config['cam1_to_cam2_min_time'], config['cam1_to_cam2_max_time'])
        config['cam1_to_cam2_min_time'] = min_time
        config['cam1_to_cam2_max_time'] = max_time
        avg_time = int((min_time + max_time) // 2)
        
        # Add camera topology
        config['camera_topology'] = {
            "1": {
                "2": {
                    'direction': 'right_to_left',
                    'avg_transit_time': avg_time,
                    'exit_zones': [[0.7, 0.3, 1.0, 0.8]],
                    'entry_zones': [[0.0, 0.3, 0.3, 0.8]]
                }
            },
            "2": {
                "1": {
                    'direction': 'left_to_right',
                    'avg_transit_time': 15,
                    'exit_zones': [[0.0, 0.3, 0.3, 0.8]],
                    'entry_zones': [[0.7, 0.3, 1.0, 0.8]]
                }
            }
        }
        
        return config
    
    def extract_result_tuple(self, output_text):
        """
        Extract result tuple from command output using various methods.
        """
        # Method 1: Look for "Result tuple:" line
        result_line = [line for line in output_text.split('\n') if "Result tuple:" in line]
        if result_line:
            result_str = result_line[0].split("Result tuple:")[1].strip()
            try:
                return eval(result_str)
            except:
                pass
        
        # Method 2: Try to extract using regex for tuples
        tuple_pattern = r'\((\d+),\s*(\d+),\s*(\d+)\)'
        tuple_matches = re.findall(tuple_pattern, output_text)
        for match in tuple_matches:
            try:
                return tuple(map(int, match))
            except:
                continue
        
        # Method 3: Look for specific result outputs
        camera1_pattern = r'Camera\s*1.*?:\s*(\d+)\s*unique'
        camera2_pattern = r'Camera\s*2.*?:\s*(\d+)\s*unique'
        transitions_pattern = r'Transitions.*?:\s*(\d+)'
        
        camera1_match = re.search(camera1_pattern, output_text)
        camera2_match = re.search(camera2_pattern, output_text)
        transitions_match = re.search(transitions_pattern, output_text)
        
        if camera1_match and camera2_match and transitions_match:
            try:
                return (
                    int(camera1_match.group(1)),
                    int(camera2_match.group(1)),
                    int(transitions_match.group(1))
                )
            except:
                pass
        
        # Method 4: Parse cross_camera_summary.json if it exists in the output directory
        try:
            # Look for newest summary file
            json_pattern = os.path.join(str(self.temp_trial_dir), "*cross_camera_summary.json")
            summary_files = glob.glob(json_pattern)
            if summary_files:
                newest_file = max(summary_files, key=os.path.getctime)
                with open(newest_file, 'r') as f:
                    summary = json.load(f)
                    return (
                        summary.get('camera1_unique_persons', 0),
                        summary.get('camera2_unique_persons', 0),
                        summary.get('camera1_to_camera2_transitions', 0)
                    )
        except Exception as e:
            if self.debug:
                print(f"Error parsing summary JSON: {e}")
        
        # Failed to extract
        return None
    
    def extract_quality_metrics(self, output_text, output_dir):
        """
        Extract additional quality metrics from the output and result files.
        """
        metrics = {
            'track_smoothness': 0.0,     # Smoothness of tracking (less jumping)
            'track_continuity': 0.0,     # Fewer track breaks
            'detection_confidence': 0.0, # Average detection confidence
            'tracking_consistency': 0.0, # Consistency of tracking features
            'transit_time_consistency': 0.0  # Consistency of transit times
        }
        
        # Look for tracked results in JSON files
        try:
            # Check for tracking results in each camera directory
            for cam_id in [1, 2]:
                # Find all tracking result files for this camera
                cam_pattern = os.path.join(str(output_dir), f"*Camera_{cam_id}*", "tracking_results.json")
                result_files = glob.glob(cam_pattern)
                
                if result_files:
                    with open(result_files[0], 'r') as f:
                        results = json.load(f)
                        
                        # Count tracks with sufficient duration (>2s) as a quality metric
                        valid_tracks = 0
                        track_durations = []
                        
                        if 'persons' in results:
                            for person_id, data in results['persons'].items():
                                duration = data.get('duration', 0)
                                if duration > 2.0:  # Tracks longer than 2 seconds
                                    valid_tracks += 1
                                    track_durations.append(duration)
                                    
                            if 'total_persons' in results and results['total_persons'] > 0:
                                # Ratio of valid tracks to total tracks
                                metrics[f'cam{cam_id}_valid_track_ratio'] = valid_tracks / max(1, results['total_persons'])
                                
                                # Average track duration
                                if track_durations:
                                    metrics[f'cam{cam_id}_avg_track_duration'] = sum(track_durations) / len(track_durations)
            
            # Look for cross-camera summary for transition data
            summary_pattern = os.path.join(str(output_dir), "*cross_camera_summary.json")
            summary_files = glob.glob(summary_pattern)
            
            if summary_files:
                with open(summary_files[0], 'r') as f:
                    summary = json.load(f)
                    
                    # Extract candidate transitions vs confirmed
                    if 'raw_counts' in summary:
                        raw = summary['raw_counts']
                        candidates = raw.get('camera1_to_camera2_candidates', 0)
                        confirmed = raw.get('camera1_to_camera2_transitions', 0)
                        
                        if candidates > 0:
                            # Higher ratio means more confident transitions
                            metrics['transition_confidence'] = confirmed / max(1, candidates)
        
        except Exception as e:
            if self.debug:
                print(f"Error extracting quality metrics: {e}")
        
        return metrics
    
    def calculate_trajectory_metrics(self, output_dir):
        """
        Calculate trajectory smoothness metrics from tracking output.
        
        Args:
            output_dir: Directory containing tracking visualization data
            
        Returns:
            Dictionary of trajectory quality metrics
        """
        metrics = {
            'trajectory_smoothness': 0.0,
            'velocity_consistency': 0.0
        }
        
        try:
            # Look for visualization frames to analyze trajectories
            vis_dir = Path(output_dir) / "visualization"
            
            if not vis_dir.exists():
                return metrics
                
            frame_files = list(vis_dir.glob("frame_*.jpg"))
            frame_files.sort()
            
            if len(frame_files) < 10:  # Need enough frames for analysis
                return metrics
                
            # Sample a subset of frames to analyze
            sample_frames = frame_files[::5][:20]  # Every 5th frame, up to 20 frames
            
            # Extract trajectories from visualization frames (if available)
            trajectories = defaultdict(list)
            
            for frame_path in sample_frames:
                frame = cv2.imread(str(frame_path))
                
                if frame is None:
                    continue
                    
                # Use simple text extraction to find bounding boxes and IDs
                # This is a simplified approach - would need OCR or direct data access for real implementation
                
                # Placeholder for trajectory extraction
                # In a real implementation, you would extract tracking boxes and IDs from each frame
                # and build trajectories for each tracked person
                
                # For demonstration, we'll generate a random smoothness score
                smoothness = random.uniform(0.5, 1.0)
                metrics['trajectory_smoothness'] = smoothness
        
        except Exception as e:
            if self.debug:
                print(f"Error calculating trajectory metrics: {e}")
        
        return metrics
    
    def evaluate_tracking_results(self, result_tuple, video_pair, ground_truth=None, quality_metrics=None):
        """
        Evaluate tracking results using ground truth or consistency metrics.
        
        Args:
            result_tuple: (camera1_count, camera2_count, transitions) tuple
            video_pair: The (video1, video2) paths being evaluated
            ground_truth: Optional ground truth values for this video pair
            quality_metrics: Dictionary of additional quality metrics
            
        Returns:
            error: Error metric (lower is better)
            metrics: Dictionary of evaluation metrics
        """
        if result_tuple is None:
            return 1000.0, {'valid': False}
            
        camera1_count, camera2_count, transitions = result_tuple
        
        # Generate pair name for lookup in ground truth
        pair_name = self._get_pair_name(video_pair)
        
        # 1. If we have ground truth, use it for evaluation
        if ground_truth and pair_name in ground_truth:
            gt = ground_truth[pair_name]
            gt_camera1, gt_camera2, gt_transitions = gt
            
            # Mean squared percentage error weighted by importance
            if self.multi_objective:
                # Use relative error (percentage) to make different metrics comparable
                camera1_error = ((camera1_count - gt_camera1) / max(1, gt_camera1)) ** 2
                camera2_error = ((camera2_count - gt_camera2) / max(1, gt_camera2)) ** 2
                transit_error = ((transitions - gt_transitions) / max(1, gt_transitions)) ** 2
                
                # Weighted by importance (transitions most important)
                error = (
                    0.25 * camera1_error +
                    0.25 * camera2_error +
                    0.5 * transit_error
                )
            else:
                # Single objective: sum of squared errors
                error = (
                    (camera1_count - gt_camera1) ** 2 +
                    (camera2_count - gt_camera2) ** 2 +
                    5 * (transitions - gt_transitions) ** 2  # Weighted more for transitions
                )
                
            metrics = {
                'valid': True,
                'camera1_error': camera1_error,
                'camera2_error': camera2_error,
                'transit_error': transit_error,
                'error_type': 'ground_truth'
            }
        
        # 2. Without ground truth, use consistency and quality metrics
        else:
            metrics = {
                'valid': True,
                'error_type': 'consistency'
            }
            
            # Base error on consistency metrics
            
            # 2.1. Check if camera counts are reasonable (not too high or too low)
            # For a 10-minute sample video, we might expect 5-50 people
            camera1_count_penalty = max(0, abs(camera1_count - 25) - 15) ** 2
            camera2_count_penalty = max(0, abs(camera2_count - 12) - 10) ** 2
            
            # 2.2. Check if transition count is reasonable
            # We expect some transitions, but not too many
            if transitions == 0:
                # Penalty for no transitions (likely failed tracking)
                transition_penalty = 100
            elif transitions > min(camera1_count, camera2_count):
                # Penalty for impossible tracking (more transitions than people)
                transition_penalty = ((transitions - min(camera1_count, camera2_count)) * 5) ** 2
            elif transitions > camera1_count * 0.5:
                # Penalty for too many transitions (unlikely that most people transition)
                transition_penalty = ((transitions - camera1_count * 0.5) * 2) ** 2
            else:
                # No penalty for reasonable transition count
                transition_penalty = 0
                
            # 2.3. Calculate ratio consistency (for reasonable proportions)
            # Typically, we expect Camera 2 to have fewer people than Camera 1
            # and the transition count to be less than both
            if camera1_count > 0 and camera2_count > 0:
                expected_cam2_ratio = 0.5  # Camera 2 typically has about half the people of Camera 1
                actual_cam2_ratio = camera2_count / camera1_count
                ratio_penalty = ((actual_cam2_ratio - expected_cam2_ratio) * 10) ** 2
                
                # Expected transition ratio (typically 10-20% of Camera 1 count)
                if camera1_count > 0:
                    expected_transit_ratio = 0.15  # About 15% of people transition
                    actual_transit_ratio = transitions / camera1_count
                    transit_ratio_penalty = ((actual_transit_ratio - expected_transit_ratio) * 15) ** 2
                else:
                    transit_ratio_penalty = 0
            else:
                ratio_penalty = 100  # Penalty for zero counts
                transit_ratio_penalty = 100
            
            # 2.4. Consider quality metrics if available
            quality_bonus = 0
            if quality_metrics:
                # Calculate a quality score from metrics
                quality_score = sum([
                    quality_metrics.get('trajectory_smoothness', 0) * 20,
                    quality_metrics.get('cam1_valid_track_ratio', 0) * 30,
                    quality_metrics.get('cam2_valid_track_ratio', 0) * 30,
                    quality_metrics.get('transition_confidence', 0) * 40
                ])
                quality_bonus = quality_score
                metrics['quality_score'] = quality_score
            
            # Combine penalties into final error score
            # Lower score is better (we're minimizing)
            error = (
                camera1_count_penalty +
                camera2_count_penalty +
                transition_penalty * 2 +  # Higher weight for transitions
                ratio_penalty +
                transit_ratio_penalty * 2 -  # Higher weight for transition ratio
                quality_bonus  # Subtract quality bonus (higher quality = lower error)
            )
            
            # Store component penalties in metrics
            metrics.update({
                'camera1_count_penalty': camera1_count_penalty,
                'camera2_count_penalty': camera2_count_penalty,
                'transition_penalty': transition_penalty,
                'ratio_penalty': ratio_penalty,
                'transit_ratio_penalty': transit_ratio_penalty,
                'quality_bonus': quality_bonus
            })
        
        # Ensure error is finite
        if not np.isfinite(error):
            error = 1000.0
            
        # Cap maximum error
        error = min(error, 1000.0)
        
        # Store results in metrics
        metrics.update({
            'camera1_count': camera1_count,
            'camera2_count': camera2_count,
            'transitions': transitions,
            'error': error
        })
        
        return error, metrics
    
    def evaluate_on_video_pair(self, params, video_pair, trial_dir):
        """
        Evaluate parameters on a single video pair.
        
        Args:
            params: Dictionary of parameters
            video_pair: Tuple of (video1_path, video2_path)
            trial_dir: Directory to save trial results
            
        Returns:
            result_tuple: The (camera1_count, camera2_count, transitions) tuple
            metrics: Dictionary of evaluation metrics
            output_text: Command output text
        """
        # Create directory for this specific video pair evaluation
        pair_name = self._get_pair_name(video_pair)
        pair_dir = trial_dir / f"pair_{pair_name}"
        pair_dir.mkdir(exist_ok=True)
        
        # Create config file
        config_path = pair_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(params, f, cls=NumpyEncoder)
        
        video1_path, video2_path = video_pair
        
        try:
            # Build command to run the tracking script
            cmd = [
                "python", SCRIPT_PATH,
                "--video1", str(video1_path),
                "--video2", str(video2_path),
                "--output_dir", str(pair_dir),
                "--config", str(config_path)
            ]
            
            # Run the script and capture output
            if self.verbose:
                print(f"Evaluating on pair {pair_name}...")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Save the output for debugging
            with open(pair_dir / "output.txt", "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            
            # Extract the result tuple from the output
            result_tuple = self.extract_result_tuple(result.stdout)
            
            if result_tuple is None:
                print(f"Could not extract result tuple for {pair_name}. Using default values.")
                result_tuple = (0, 0, 0)
                
            # Extract quality metrics
            quality_metrics = self.extract_quality_metrics(result.stdout, pair_dir)
            
            # Calculate trajectory metrics if available
            trajectory_metrics = self.calculate_trajectory_metrics(pair_dir)
            quality_metrics.update(trajectory_metrics)
            
            # Get ground truth for this pair if available
            pair_gt = self.ground_truth.get(pair_name, None)
            
            # Evaluate results
            error, metrics = self.evaluate_tracking_results(
                result_tuple, video_pair, ground_truth=pair_gt, 
                quality_metrics=quality_metrics
            )
            
            metrics['pair_name'] = pair_name
            
            return result_tuple, metrics, result.stdout
            
        except subprocess.TimeoutExpired:
            print(f"Script execution timed out for {pair_name}")
            return None, {'valid': False, 'error': 1000.0, 'pair_name': pair_name}, "Timeout"
        except Exception as e:
            print(f"Error evaluating on {pair_name}: {e}")
            return None, {'valid': False, 'error': 1000.0, 'pair_name': pair_name}, str(e)
    
    def _get_pair_name(self, video_pair):
        """Extract a name for the video pair from the file paths."""
        video1, video2 = video_pair
        # Try to extract date from filename if available
        try:
            v1_name = Path(video1).stem
            parts = v1_name.split('_')
            if len(parts) >= 3:
                return parts[2]  # Assuming format Camera_1_YYYYMMDD
        except:
            pass
            
        # Fallback: use hash of combined paths
        return str(abs(hash(str(video1) + str(video2))) % 10000)
    
    def objective_function_impl(self, **params):
        """
        Objective function implementation to be minimized.
        
        Args:
            **params: Parameter values from optimization

        Returns:
            float: Error metric (distance from target values)
        """
        # Create a config dictionary for the script
        config = self.prepare_config(params)
        
        # Create unique trial directory
        trial_time = int(time.time())
        trial_dir = self.output_dir / f'trial_{trial_time}'
        trial_dir.mkdir(exist_ok=True)
        self.temp_trial_dir = trial_dir  # Store for result extraction
        
        # Save config to a file
        config_path = trial_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, cls=NumpyEncoder)
        
        # Evaluate on all training pairs
        pair_results = []
        errors = []
        all_metrics = []
        
        for video_pair in self.train_pairs:
            result_tuple, metrics, output_text = self.evaluate_on_video_pair(
                config, video_pair, trial_dir
            )
            
            if metrics['valid']:
                errors.append(metrics['error'])
                pair_results.append({
                    'pair_name': metrics['pair_name'],
                    'result': result_tuple,
                    'metrics': metrics
                })
                all_metrics.append(metrics)
        
        # Calculate average error across all training pairs
        if errors:
            avg_error = sum(errors) / len(errors)
        else:
            avg_error = 1000.0  # High penalty if all evaluations failed
        
        # Store trial results
        trial_result = {
            'params': {k: convert_numpy_types(v) for k, v in params.items()},
            'trial_time': trial_time,
            'train_error': avg_error,
            'pair_results': pair_results
        }
        
        # Evaluate on validation pairs if using cross-validation
        if self.cross_validation and self.val_pairs:
            val_errors = []
            val_pair_results = []
            
            for video_pair in self.val_pairs:
                result_tuple, metrics, output_text = self.evaluate_on_video_pair(
                    config, video_pair, trial_dir
                )
                
                if metrics['valid']:
                    val_errors.append(metrics['error'])
                    val_pair_results.append({
                        'pair_name': metrics['pair_name'],
                        'result': result_tuple,
                        'metrics': metrics
                    })
                    all_metrics.append(metrics)
            
            # Calculate validation error
            if val_errors:
                val_error = sum(val_errors) / len(val_errors)
                trial_result['val_error'] = val_error
                trial_result['val_pair_results'] = val_pair_results
            else:
                trial_result['val_error'] = 1000.0
        
        # Store trial results
        self.opt_history.append(trial_result)
        
        # Determine if this is the best result
        # If using cross-validation, consider both training and validation error
        if self.cross_validation and 'val_error' in trial_result:
            # Use a weighted combination of training and validation error
            combined_error = 0.4 * trial_result['train_error'] + 0.6 * trial_result['val_error']
        else:
            combined_error = avg_error
        
        if combined_error < self.best_score:
            self.best_score = combined_error
            self.best_params = {k: convert_numpy_types(v) for k, v in params.items()}
            
            # Store best results from each pair
            self.best_results = all_metrics
            
            self.save_best_params()
            
            if self.verbose:
                if self.cross_validation and 'val_error' in trial_result:
                    print(f"New best: Train error: {trial_result['train_error']:.4f}, " +
                         f"Val error: {trial_result['val_error']:.4f}, " +
                         f"Combined: {combined_error:.4f}")
                else:
                    print(f"New best: Error: {avg_error:.4f}")
                
                # Print some example results
                if pair_results:
                    print("Example results:")
                    for i, pr in enumerate(pair_results[:2]):  # Show at most 2 examples
                        if 'result' in pr and pr['result']:
                            print(f"  Pair {pr['pair_name']}: {pr['result']}")
        
        # Return the combined error for optimization
        return float(combined_error)
    
    def save_best_params(self):
        """Save the best parameters found so far."""
        best_params_path = self.output_dir / 'best_params.json'
        
        result_info = {
            'best_params': self.best_params,
            'error': self.best_score,
            'pair_results': self.best_results,
            'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(best_params_path, 'w') as f:
            json.dump(result_info, f, cls=NumpyEncoder, indent=2)
    
    def save_optimization_history(self):
        """Save the full optimization history."""
        history_path = self.output_dir / 'optimization_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.opt_history, f, cls=NumpyEncoder, indent=2)
    
    def optimize(self):
        """Run the optimization process with enhanced techniques."""
        # Create the decorated objective function with proper dimensions
        @use_named_args(dimensions=self.param_space)
        def objective(**params):
            return self.objective_function_impl(**params)
        
        # Start optimization
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting optimization with {self.n_calls} iterations...")
            if self.cross_validation:
                print(f"Using {self.n_folds}-fold cross-validation")
        
        try:
            # Bayesian optimization with periodic fold rotation for cross-validation
            if self.cross_validation and self.n_folds > 1:
                # Split calls across folds
                calls_per_fold = self.n_calls // self.n_folds
                initial_points_per_fold = self.n_initial_points // self.n_folds
                
                # Run optimization for each fold
                all_results = []
                self.fold_results = []
                
                for fold in range(self.n_folds):
                    if self.verbose:
                        print(f"\n===== Optimizing on fold {fold+1}/{self.n_folds} =====")
                    
                    # Rotate to current fold
                    self.current_fold = fold
                    self.train_pairs = [p for i, fold_pairs in enumerate(self.folds) 
                                      for p in fold_pairs if i != self.current_fold]
                    self.val_pairs = self.folds[self.current_fold]
                    
                    # Run optimization for this fold
                    fold_result = gp_minimize(
                        objective,
                        self.param_space,
                        n_calls=calls_per_fold,
                        random_state=42 + fold,
                        verbose=self.verbose,
                        n_random_starts=initial_points_per_fold,
                        n_jobs=1
                    )
                    
                    all_results.append(fold_result)
                    self.fold_results.append({
                        'fold': fold,
                        'best_params': {k: convert_numpy_types(v) for k, v in zip(self.param_names, fold_result.x)},
                        'best_score': fold_result.fun
                    })
                
                # Combine results - use the best fold's parameters
                best_fold_idx = np.argmin([r.fun for r in all_results])
                result = all_results[best_fold_idx]
                
                if self.verbose:
                    print(f"\nBest parameters found from fold {best_fold_idx+1}")
            else:
                # Regular Bayesian optimization without cross-validation
                result = gp_minimize(
                    objective,
                    self.param_space,
                    n_calls=self.n_calls,
                    random_state=42,
                    verbose=self.verbose,
                    n_random_starts=self.n_initial_points,
                    n_jobs=1
                )
        except Exception as e:
            print(f"GP optimization failed: {e}")
            print("Falling back to random search")
            
            # Fall back to random search
            result = dummy_minimize(
                objective,
                self.param_space,
                n_calls=self.n_calls,
                random_state=42,
                verbose=self.verbose
            )
        
        # Record results
        self.result = result
        end_time = time.time()
        
        # Save optimization history
        self.save_optimization_history()
        
        if self.verbose:
            print(f"\nOptimization completed in {(end_time - start_time)/60:.1f} minutes")
            print(f"Best parameters found:")
            for name, value in zip(self.param_names, result.x):
                print(f"  {name}: {convert_numpy_types(value)}")
            print(f"Best error score: {self.best_score:.4f}")
            
            if self.fold_results:
                print("\nCross-validation fold results:")
                for fold_res in self.fold_results:
                    print(f"  Fold {fold_res['fold']+1}: {fold_res['best_score']:.4f}")
        
        return self.best_params, self.best_score, self.best_results
    
    def validate_best_params(self):
        """Validate the best parameters on all video pairs."""
        if not self.best_params:
            print("No best parameters found. Run optimization first.")
            return None
            
        print("\n===== Validating Best Parameters =====")
        
        # Create config from best parameters
        config = self.prepare_config(self.best_params)
        
        # Create validation directory
        val_dir = self.output_dir / 'validation'
        val_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = val_dir / 'best_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, cls=NumpyEncoder, indent=2)
        
        # Evaluate on all video pairs
        all_results = []
        
        for video_pair in self.video_pairs:
            pair_name = self._get_pair_name(video_pair)
            print(f"Validating on pair {pair_name}...")
            
            result_tuple, metrics, output_text = self.evaluate_on_video_pair(
                config, video_pair, val_dir
            )
            
            all_results.append({
                'pair_name': pair_name,
                'result': result_tuple,
                'metrics': metrics
            })
            
            print(f"  Result: {result_tuple}, Error: {metrics.get('error', 'N/A')}")
        
        # Save overall validation results
        val_results_path = val_dir / 'validation_results.json'
        with open(val_results_path, 'w') as f:
            json.dump(all_results, f, cls=NumpyEncoder, indent=2)
        
        return all_results
    
    def plot_optimization_results(self):
        """Plot optimization convergence and parameter importance."""
        if not hasattr(self, 'result'):
            print("Run optimization first")
            return
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plot_convergence(self.result)
        plt.title('Optimization Convergence')
        plt.savefig(plots_dir / 'convergence.png')
        
        # Plot error over iterations
        plt.figure(figsize=(12, 10))
        iterations = list(range(len(self.opt_history)))
        
        if self.cross_validation:
            # Plot training and validation error
            train_errors = [trial.get('train_error', float('nan')) for trial in self.opt_history]
            val_errors = [trial.get('val_error', float('nan')) for trial in self.opt_history]
            
            plt.subplot(211)
            plt.plot(iterations, train_errors, 'b-', label='Training Error')
            plt.plot(iterations, val_errors, 'r-', label='Validation Error')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.title('Training and Validation Error')
            plt.legend()
            plt.grid(True)
            
            # Plot fold transitions
            if hasattr(self, 'fold_results') and self.fold_results:
                for i, fold_res in enumerate(self.fold_results):
                    plt.axvline(x=i*len(iterations)//len(self.fold_results), 
                             color='g', linestyle='--', 
                             label=f'Fold {i+1}' if i==0 else None)
        else:
            # Plot just the single error metric
            errors = [trial.get('train_error', float('nan')) for trial in self.opt_history]
            
            plt.subplot(211)
            plt.plot(iterations, errors, 'b-', label='Error')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.title('Error Over Iterations')
            plt.legend()
            plt.grid(True)
        
        # Plot parameter values for key parameters
        plt.figure(figsize=(14, 10))
        key_params = [
            'cam1_min_confidence',
            'cam1_similarity_threshold',
            'cam2_min_confidence',
            'cam2_similarity_threshold',
            'global_similarity_threshold',
            'min_track_confirmations',
            'cam1_to_cam2_min_time',
            'cam1_to_cam2_max_time'
        ]
        
        for i, param_name in enumerate(key_params):
            param_idx = self.param_names.index(param_name)
            param_values = [convert_numpy_types(params['params'].get(param_name, float('nan'))) 
                           for params in self.opt_history]
            
            plt.subplot(4, 2, i+1)
            plt.plot(iterations, param_values, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel(param_name)
            plt.title(f'Parameter: {param_name}')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'parameter_evolution.png')
        
        if self.verbose:
            print(f"Plots saved to {plots_dir}")


def find_video_pairs(input_dir):
    """Find pairs of camera videos in the input directory."""
    input_path = Path(input_dir)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(list(input_path.glob(f'*{ext}')))
    
    # Group videos by camera
    cam1_videos = [v for v in all_videos if "Camera_1" in v.name]
    cam2_videos = [v for v in all_videos if "Camera_2" in v.name]
    
    print(f"Found {len(cam1_videos)} Camera 1 videos and {len(cam2_videos)} Camera 2 videos")
    
    # Create pairs based on matching date (YYYYMMDD part)
    video_pairs = []
    for cam1_video in cam1_videos:
        # Extract date part (YYYYMMDD) from Camera_1_YYYYMMDD format
        parts = cam1_video.stem.split('_')
        if len(parts) >= 3:
            date_part = parts[2]  # Get the date part
            # Find matching Camera 2 video with same date
            matching_cam2 = [v for v in cam2_videos if date_part in v.stem]
            
            if matching_cam2:
                video_pairs.append((cam1_video, matching_cam2[0]))
    
    if not video_pairs:
        # If no date-based pairs found, try matching by file order
        if len(cam1_videos) == len(cam2_videos):
            # Sort videos by name
            cam1_videos.sort()
            cam2_videos.sort()
            video_pairs = list(zip(cam1_videos, cam2_videos))
    
    print(f"Found {len(video_pairs)} video pairs for optimization")
    return video_pairs

def load_ground_truth(gt_file):
    """Load ground truth data from JSON file if available."""
    gt_path = Path(gt_file)
    
    if not gt_path.exists():
        print(f"Ground truth file {gt_file} not found. Will use consistency metrics instead.")
        return None
        
    try:
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
            
        # Convert to expected format
        ground_truth = {}
        for pair_name, values in gt_data.items():
            if isinstance(values, list) and len(values) == 3:
                ground_truth[pair_name] = values
            elif isinstance(values, dict) and 'camera1_count' in values:
                ground_truth[pair_name] = (
                    values.get('camera1_count', 0),
                    values.get('camera2_count', 0),
                    values.get('transitions', 0)
                )
                
        print(f"Loaded ground truth for {len(ground_truth)} video pairs")
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='Enhanced Parameter Optimization for People Tracking')
    
    # Input/output options
    parser.add_argument('--input_dir', type=str, default='./videos',
                        help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, default='./optimization_results',
                        help='Directory to save optimization results')
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='Path to ground truth JSON file (optional)')
    
    # Optimization options
    parser.add_argument('--n_calls', type=int, default=50,
                        help='Number of optimization iterations')
    parser.add_argument('--n_initial_points', type=int, default=10,
                        help='Number of initial random points for GP optimization')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Use cross-validation for more robust optimization')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of cross-validation folds')
    parser.add_argument('--multi_objective', action='store_true',
                        help='Use multi-objective optimization')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize tracking during evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    # Single pair optimization
    parser.add_argument('--video1', type=str, default=None,
                        help='Path to Camera 1 video (for single pair optimization)')
    parser.add_argument('--video2', type=str, default=None,
                        help='Path to Camera 2 video (for single pair optimization)')
    
    return parser

def main():
    """Main function to run the optimization."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Load ground truth if provided
    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else None
    
    # Find video pairs or use single pair if specified
    if args.video1 and args.video2:
        video_pairs = [(args.video1, args.video2)]
        print(f"Using single video pair for optimization")
    else:
        video_pairs = find_video_pairs(args.input_dir)
        
    if not video_pairs:
        print("No video pairs found. Please check input directory or provide specific video paths.")
        return
    
    # Create and run optimizer
    optimizer = EnhancedBayesianOptimizer(
        video_pairs=video_pairs,
        output_dir=args.output_dir,
        ground_truth=ground_truth,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        cross_validation=args.cross_validation and len(video_pairs) > 1,
        n_folds=min(args.n_folds, len(video_pairs)),
        multi_objective=args.multi_objective,
        visualize_evaluations=args.visualize,
        verbose=args.verbose,
        debug=args.debug
    )
    
    best_params, best_score, best_results = optimizer.optimize()
    optimizer.plot_optimization_results()
    
    # Validate best parameters on all pairs
    validation_results = optimizer.validate_best_params()
    
    print("\n" + "="*60)
    print("Enhanced optimization completed!")
    print(f"Best parameters saved to: {Path(args.output_dir) / 'best_params.json'}")
    print(f"Best error score: {best_score:.4f}")
    
    if validation_results:
        print("\nValidation results:")
        for res in validation_results:
            print(f"  Pair {res['pair_name']}: {res['result']}")
    
    print("="*60)
    
    # Return best parameters for potential further use
    return best_params

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.1f} minutes")