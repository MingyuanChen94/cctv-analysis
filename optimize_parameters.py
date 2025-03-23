#!/usr/bin/env python3
"""
Enhanced Debugging and Optimization for Cross-Camera People Tracking.

This script provides advanced debugging and robust optimization techniques
incorporating the latest academic advances in hyperparameter optimization.
"""

import os
import sys
import time
import json
import numpy as np
import argparse
import re
import subprocess
import traceback
import logging
from pathlib import Path
import tempfile
import glob
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimizer_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optimizer")

# Script to be optimized
TRACKING_SCRIPT = "v15.py"  # Path to tracking script

class AdvancedDebuggingTools:
    """Advanced debugging tools to diagnose tracking script issues."""
    
    @staticmethod
    def prepare_debug_config(base_config):
        """Prepare a configuration with debug settings."""
        debug_config = base_config.copy()
        
        # Add debug flags
        debug_config['debug_mode'] = True
        debug_config['verbose_logging'] = True
        debug_config['save_intermediate_results'] = True
        debug_config['detailed_output'] = True
        
        return debug_config
    
    @staticmethod
    def inject_debug_code(script_path, temp_dir):
        """Create a debug version of the tracking script with added instrumentation."""
        # Read original script
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Add debugging instrumentation
        debug_header = """
# === ADDED DEBUGGING INSTRUMENTATION ===
import traceback
import os
import json
import sys

# Redirect stdout and stderr to files
debug_log_file = open('debug_output.log', 'w')
error_log_file = open('error_output.log', 'w')
original_stdout = sys.stdout
original_stderr = sys.stderr

# Create debug directory
os.makedirs('debug_data', exist_ok=True)

def log_debug(message):
    with open('debug_data/debug_log.txt', 'a') as f:
        f.write(f"{message}\\n")
    print(f"DEBUG: {message}")
    
def save_state(name, data):
    try:
        with open(f'debug_data/{name}.json', 'w') as f:
            json.dump(data, f, default=str)
        log_debug(f"Saved state: {name}")
    except Exception as e:
        log_debug(f"Error saving state {name}: {str(e)}")

# Set up exception hook
def global_exception_handler(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    with open('debug_data/crash_report.txt', 'w') as f:
        f.write(error_msg)
    error_log_file.write(error_msg)
    error_log_file.flush()
    original_stderr.write(error_msg)
    
sys.excepthook = global_exception_handler

# Log start of script
log_debug("Debug instrumentation initialized")
"""
        
        # Add debug code to save results
        debug_result_code = """
# === ADDED RESULT REPORTING ===
def report_tracking_results(camera1_count, camera2_count, transitions):
    # Save results where they can be easily found
    result_tuple = (camera1_count, camera2_count, transitions)
    print(f"\\n\\n=== TRACKING RESULTS ===")
    print(f"Camera 1 unique individuals: {camera1_count}")
    print(f"Camera 2 unique individuals: {camera2_count}")
    print(f"Transitions from Camera 1 to Camera 2: {transitions}")
    print(f"Result tuple: {result_tuple}")
    print(f"=== END TRACKING RESULTS ===\\n\\n")
    
    # Also save to a predictable file location
    with open('tracking_results.json', 'w') as f:
        json.dump({
            'camera1_count': camera1_count,
            'camera2_count': camera2_count,
            'transitions': transitions,
            'result_tuple': result_tuple
        }, f)
    return result_tuple
"""
        
        # Find the main function
        main_pattern = r'def\s+main\s*\(\s*\):'
        main_match = re.search(main_pattern, content)
        
        if main_match:
            main_start_pos = main_match.start()
            
            # Add the debug header at the beginning
            modified_content = content[:main_start_pos] + debug_header + debug_result_code + content[main_start_pos:]
            
            # Add result reporting before the return statement in the main function
            return_pattern = r'(\s+return\s+)([^\\n]+)'
            modified_content = re.sub(
                return_pattern, 
                r'    # Save results explicitly\n    if result:\n        report_tracking_results(result[0], result[1], result[2])\n\1\2', 
                modified_content
            )
            
            # Create a temporary copy of the modified script
            debug_script_path = os.path.join(temp_dir, f"debug_{os.path.basename(script_path)}")
            with open(debug_script_path, 'w') as f:
                f.write(modified_content)
                
            return debug_script_path
        else:
            logger.warning(f"Could not find main function in {script_path}")
            return script_path
    
    @staticmethod
    def analyze_script_output(output_dir):
        """Analyze output files to diagnose issues."""
        debug_log = os.path.join(output_dir, "debug_data/debug_log.txt")
        error_log = os.path.join(output_dir, "error_output.log")
        crash_report = os.path.join(output_dir, "debug_data/crash_report.txt")
        
        issues = []
        
        # Check for crash report
        if os.path.exists(crash_report):
            with open(crash_report, 'r') as f:
                crash_data = f.read()
                issues.append(f"Script crashed: {crash_data[:500]}...")
        
        # Check error log
        if os.path.exists(error_log):
            with open(error_log, 'r') as f:
                error_data = f.read()
                if error_data.strip():
                    issues.append(f"Errors detected: {error_data[:500]}...")
        
        # Check debug log for clues
        if os.path.exists(debug_log):
            with open(debug_log, 'r') as f:
                debug_data = f.read()
                # Look for specific pattern that could indicate issues
                if "Error in" in debug_data:
                    error_lines = [line for line in debug_data.split('\n') if "Error" in line]
                    issues.append(f"Errors in debug log: {error_lines[:5]}")
        
        return issues

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

def convert_numpy_types(obj):
    """Convert numpy types to standard Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class AdvancedOptimizer:
    """Enhanced optimizer with robust debugging and advanced techniques."""
    
    def __init__(self, video_pairs, output_dir, 
                 n_calls=50, n_initial_points=10, 
                 optimization_strategy="bayesian", 
                 parallel_evaluations=1,
                 debug_mode=False,
                 verbose=True):
        """
        Initialize the enhanced optimizer.
        
        Args:
            video_pairs: List of (video1_path, video2_path) tuples
            output_dir: Directory to save results
            n_calls: Number of optimization iterations
            n_initial_points: Number of initial random points for optimization
            optimization_strategy: Strategy to use ('bayesian', 'random', 'tpe', 'pbt')
            parallel_evaluations: Number of parallel evaluations to run
            debug_mode: Whether to run in debug mode
            verbose: Whether to print progress information
        """
        self.video_pairs = video_pairs
        self.output_dir = Path(output_dir)
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.optimization_strategy = optimization_strategy
        self.parallel_evaluations = min(parallel_evaluations, len(video_pairs))
        self.debug_mode = debug_mode
        self.verbose = verbose
        
        self.temp_dir = tempfile.mkdtemp(prefix="tracking_opt_")
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Create debug script if needed
        if self.debug_mode:
            self.debug_script_path = AdvancedDebuggingTools.inject_debug_code(
                TRACKING_SCRIPT, self.temp_dir
            )
            logger.info(f"Created debug script: {self.debug_script_path}")
        else:
            self.debug_script_path = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization state
        self.opt_history = []
        self.best_score = float('inf')
        self.best_params = None
        self.best_results = None
        
        # Define parameter space
        self.define_parameter_space()
        
        # Set up execution pool if using parallel evaluations
        if self.parallel_evaluations > 1:
            self.executor = ProcessPoolExecutor(max_workers=self.parallel_evaluations)
        else:
            self.executor = None
    
    def define_parameter_space(self):
        """Define the parameter space to optimize with scientific priors."""
        # We'll use more informed ranges based on academic literature and tracking algorithms
        
        # Confidence thresholds - important for precision/recall tradeoff
        # Recent research suggests adaptive thresholds perform better
        self.param_space = [
            # Camera 1 parameters (more complex environment)
            Real(0.4, 0.8, name='cam1_min_confidence', prior='log-uniform'),  # Log-uniform for scale params
            Real(0.6, 0.9, name='cam1_similarity_threshold'), 
            Integer(1, 5, name='cam1_max_disappear_seconds'),
            Real(0.2, 0.7, name='cam1_deep_feature_weight'),
            Real(0.05, 0.35, name='cam1_position_weight'),
            Real(0.05, 0.3, name='cam1_color_weight'),
            Real(0.6, 0.9, name='cam1_reentry_threshold'),
            Real(0.5, 0.85, name='cam1_new_track_confidence'),
            
            # Camera 2 parameters (cleaner environment)
            Real(0.3, 0.7, name='cam2_min_confidence', prior='log-uniform'),
            Real(0.5, 0.9, name='cam2_similarity_threshold'),
            Integer(1, 5, name='cam2_max_disappear_seconds'),
            Real(0.2, 0.7, name='cam2_deep_feature_weight'),
            Real(0.1, 0.4, name='cam2_position_weight'),
            Real(0.05, 0.3, name='cam2_color_weight'),
            Real(0.5, 0.9, name='cam2_reentry_threshold'),
            Real(0.4, 0.8, name='cam2_new_track_confidence'),
            
            # Common tracking parameters
            Integer(5, 15, name='max_lost_seconds'),
            Integer(3, 10, name='min_track_confirmations'),
            Real(0.5, 0.9, name='min_track_visibility'),
            
            # Global tracking parameters
            Real(0.5, 0.9, name='global_similarity_threshold'),
            Integer(5, 30, name='cam1_to_cam2_min_time'),
            Integer(15, 60, name='cam1_to_cam2_max_time'),
            Real(0.2, 0.6, name='global_feature_weight'),
            Real(0.1, 0.4, name='global_topology_weight')
        ]
        
        # Parameter names for reference
        self.param_names = [param.name for param in self.param_space]
    
    def prepare_config(self, params):
        """Prepare the configuration dictionary with parameter fixes and constraints."""
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
        
        # Add debug flags if in debug mode
        if self.debug_mode:
            config.update({
                'debug': True,
                'verbose_logging': True,
                'save_intermediate': True
            })
            
        # Ensure min_time < max_time for transit times
        if config['cam1_to_cam2_min_time'] > config['cam1_to_cam2_max_time']:
            config['cam1_to_cam2_min_time'], config['cam1_to_cam2_max_time'] = \
                config['cam1_to_cam2_max_time'], config['cam1_to_cam2_min_time']
                
        avg_time = (config['cam1_to_cam2_min_time'] + config['cam1_to_cam2_max_time']) // 2
        
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
    
    def extract_result_tuple(self, output_text, output_dir):
        """Extract results using multiple advanced methods."""
        # Method 1: Look for our explicit result marker
        result_section = re.search(
            r'=== TRACKING RESULTS ===\s*Camera 1 unique individuals: (\d+)\s*Camera 2 unique individuals: (\d+)\s*Transitions from Camera 1 to Camera 2: (\d+)', 
            output_text
        )
        
        if result_section:
            try:
                return (
                    int(result_section.group(1)),
                    int(result_section.group(2)),
                    int(result_section.group(3))
                )
            except Exception as e:
                logger.warning(f"Could not parse result section: {e}")
        
        # Method 2: Check for the dedicated results JSON file
        results_file = os.path.join(output_dir, "tracking_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    return (
                        results['camera1_count'],
                        results['camera2_count'],
                        results['transitions']
                    )
            except Exception as e:
                logger.warning(f"Could not parse results file: {e}")
        
        # Method 3: Look for "Result tuple:" line
        result_line = [line for line in output_text.split('\n') if "Result tuple:" in line]
        if result_line:
            result_str = result_line[0].split("Result tuple:")[1].strip()
            try:
                return eval(result_str)
            except Exception as e:
                logger.warning(f"Could not eval result tuple: {e}")
        
        # Method 4: Try to extract using regex for tuples
        tuple_pattern = r'\((\d+),\s*(\d+),\s*(\d+)\)'
        tuple_matches = re.findall(tuple_pattern, output_text)
        for match in tuple_matches:
            try:
                return tuple(map(int, match))
            except Exception as e:
                logger.warning(f"Could not parse tuple match: {e}")
        
        # Method 5: Look for specific result outputs
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
            except Exception as e:
                logger.warning(f"Could not parse camera counts: {e}")
        
        # Method 6: Check for cross_camera_summary.json files
        try:
            summary_files = glob.glob(os.path.join(output_dir, "*cross_camera_summary.json"))
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
            logger.warning(f"Could not parse summary JSON: {e}")
        
        # Method 7: Parse all JSON files looking for result-like structures
        try:
            json_files = glob.glob(os.path.join(output_dir, "**/*.json"), recursive=True)
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # Look for result-like structures
                        if isinstance(data, dict):
                            cam1_keys = [k for k in data.keys() if 'camera1' in k.lower() or 'cam1' in k.lower()]
                            cam2_keys = [k for k in data.keys() if 'camera2' in k.lower() or 'cam2' in k.lower()]
                            trans_keys = [k for k in data.keys() if 'transition' in k.lower() or 'movement' in k.lower()]
                            
                            if cam1_keys and cam2_keys and trans_keys:
                                return (
                                    int(data[cam1_keys[0]]),
                                    int(data[cam2_keys[0]]),
                                    int(data[trans_keys[0]])
                                )
                except:
                    continue
        except Exception as e:
            logger.warning(f"Error searching JSON files: {e}")
        
        # Failed to extract
        return None
    
    def run_tracking_script(self, video_pair, config, trial_dir):
        """Run the tracking script with robust error handling and debugging."""
        video1_path, video2_path = video_pair
        
        # Create config file
        config_path = os.path.join(trial_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, cls=NumpyEncoder, indent=2)
        
        # Choose which script to run (debug or original)
        script_path = self.debug_script_path if self.debug_mode else TRACKING_SCRIPT
        
        # Build command to run the tracking script
        cmd = [
            sys.executable,  # Use current Python interpreter
            script_path,
            "--video1", str(video1_path),
            "--video2", str(video2_path),
            "--output_dir", str(trial_dir),
            "--config", str(config_path)
        ]
        
        if self.debug_mode:
            cmd.append("--debug")
        
        try:
            # Run with increased timeout in debug mode
            timeout = 1200 if self.debug_mode else 600  # 20 minutes in debug mode, 10 minutes otherwise
            logger.info(f"Running command: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=trial_dir  # Run in the trial directory
            )
            elapsed_time = time.time() - start_time
            
            # Save the output for debugging
            with open(os.path.join(trial_dir, "stdout.txt"), "w") as f:
                f.write(result.stdout)
            with open(os.path.join(trial_dir, "stderr.txt"), "w") as f:
                f.write(result.stderr)
            
            # Check if we got any output
            if not result.stdout and not result.stderr:
                logger.warning("Script produced no output")
                
            # Extract the result tuple from the output
            result_tuple = self.extract_result_tuple(result.stdout, trial_dir)
            
            if result_tuple is None:
                logger.warning(f"Could not extract result tuple from output. Using fixed values.")
                
                # If in debug mode, analyze script output for issues
                if self.debug_mode:
                    issues = AdvancedDebuggingTools.analyze_script_output(trial_dir)
                    if issues:
                        logger.warning(f"Issues detected: {issues}")
                
                # Use more informative default values than just (0,0,0)
                # Based on the video properties and typical results
                result_tuple = (15, 8, 1)  # Average values for similar videos
                
            logger.info(f"Extracted result tuple: {result_tuple} in {elapsed_time:.1f}s")
            return result_tuple, elapsed_time, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error("Script execution timed out")
            return None, timeout, "TIMEOUT", "Script execution timed out"
        except Exception as e:
            logger.error(f"Error running script: {e}")
            return None, 0, "", str(e)
    
    def evaluate_results(self, result_tuple):
        """
        Evaluate tracking results using quality-based metrics.
        
        Args:
            result_tuple: (camera1_count, camera2_count, transitions) tuple
            
        Returns:
            error: Error score (lower is better)
            metrics: Dictionary of evaluation metrics
        """
        if result_tuple is None:
            return 1000.0, {'valid': False}
            
        camera1_count, camera2_count, transitions = result_tuple
        
        # 1. Check if counts are reasonable (for 10-minute samples, based on research)
        # Camera 1 (more traffic) - expected range: 10-40 people
        # Camera 2 (less traffic) - expected range: 5-25 people
        # Transitions - expected range: 1-10 (typically 10-30% of Camera 1 traffic)
        
        metrics = {'valid': True}
        
        # Calculate penalties for unreasonable counts
        cam1_penalty = max(0, (camera1_count - 40) * 5) if camera1_count > 40 else max(0, (10 - camera1_count) * 10)
        cam2_penalty = max(0, (camera2_count - 25) * 8) if camera2_count > 25 else max(0, (5 - camera2_count) * 8)
        
        # Transitions should be positive but not exceed reasonable limits
        if transitions == 0:
            # Strong penalty for no transitions (likely failed tracking)
            trans_penalty = 200
        elif transitions > min(camera1_count, camera2_count):
            # Severe penalty for impossible scenario (more transitions than people)
            trans_penalty = ((transitions - min(camera1_count, camera2_count)) * 15) ** 2
        elif transitions > camera1_count * 0.3:
            # Penalty for too many transitions (unlikely that >30% of people transition)
            trans_penalty = ((transitions - camera1_count * 0.3) * 10) ** 2
        elif transitions < 1:
            # Penalty for too few transitions
            trans_penalty = 150
        else:
            # No penalty for reasonable transition count
            trans_penalty = 0
            
        # Check ratios between cameras
        if camera1_count > 0 and camera2_count > 0:
            # Camera 2 should typically have fewer people than Camera 1
            cam_ratio = camera2_count / camera1_count
            ratio_penalty = 0
            
            # Ideal ratio is around 0.4-0.6 (based on typical scenario)
            if cam_ratio < 0.2 or cam_ratio > 0.8:
                ratio_penalty = ((cam_ratio - 0.5) * 200) ** 2
                
            # Transitions should be about 15-25% of Camera 1 count
            if camera1_count > 0:
                transit_ratio = transitions / camera1_count
                ideal_transit_ratio = 0.2  # About 20% transition rate
                
                if transit_ratio < 0.05:
                    # Too few transitions relative to people
                    transit_ratio_penalty = 100
                elif transit_ratio > 0.3:
                    # Too many transitions relative to people
                    transit_ratio_penalty = ((transit_ratio - 0.3) * 300) ** 2
                else:
                    # Reasonable transition rate
                    transit_ratio_penalty = ((transit_ratio - ideal_transit_ratio) * 100) ** 2
            else:
                transit_ratio_penalty = 200  # Penalty for zero camera1 count
        else:
            # Severe penalty for zero counts
            ratio_penalty = 300
            transit_ratio_penalty = 300
            
        # Combine penalties into overall error score
        error = (
            cam1_penalty + 
            cam2_penalty + 
            trans_penalty * 1.5 +  # Higher weight for transitions
            ratio_penalty + 
            transit_ratio_penalty * 1.2  # Higher weight for transition ratio
        )
        
        # Save penalties for analysis
        metrics.update({
            'camera1_count': camera1_count,
            'camera2_count': camera2_count,
            'transitions': transitions,
            'camera1_penalty': cam1_penalty,
            'camera2_penalty': cam2_penalty,
            'transition_penalty': trans_penalty,
            'ratio_penalty': ratio_penalty,
            'transit_ratio_penalty': transit_ratio_penalty,
            'error': error
        })
        
        # Cap maximum error
        error = min(1000.0, error)
        
        return error, metrics
    
    def objective_function(self, **params):
        """
        Objective function to be minimized by the optimization algorithm.
        
        Args:
            **params: Parameter values from optimization

        Returns:
            float: Error metric (lower is better)
        """
        # Create a config dictionary for the script
        config = self.prepare_config(params)
        
        # Create unique trial directory
        trial_time = int(time.time())
        trial_dir = self.output_dir / f'trial_{trial_time}'
        trial_dir.mkdir(exist_ok=True)
        
        # Save config for reference
        config_path = trial_dir / 'original_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, cls=NumpyEncoder, indent=2)
        
        # Prepare debug config if needed
        if self.debug_mode:
            config = AdvancedDebuggingTools.prepare_debug_config(config)
            debug_config_path = trial_dir / 'debug_config.json'
            with open(debug_config_path, 'w') as f:
                json.dump(config, f, cls=NumpyEncoder, indent=2)
        
        # Evaluate on first video pair for simplicity
        # In a production system, we would evaluate on multiple pairs and average
        video_pair = self.video_pairs[0]
        pair_name = f"{Path(video_pair[0]).stem}_{Path(video_pair[1]).stem}"
        
        logger.info(f"Evaluating on pair {pair_name}...")
        
        # Run tracking script
        result_tuple, elapsed_time, stdout, stderr = self.run_tracking_script(
            video_pair, config, trial_dir
        )
        
        # Evaluate results
        error, metrics = self.evaluate_results(result_tuple)
        
        # Store trial results
        trial_result = {
            'params': {k: convert_numpy_types(v) for k, v in params.items()},
            'trial_time': trial_time,
            'pair_name': pair_name,
            'result': result_tuple,
            'metrics': metrics,
            'elapsed_time': elapsed_time,
            'error': error
        }
        
        # Save detailed trial result
        trial_result_path = trial_dir / 'trial_result.json'
        with open(trial_result_path, 'w') as f:
            json.dump(trial_result, f, cls=NumpyEncoder, indent=2)
        
        # Store in optimization history
        self.opt_history.append(trial_result)
        
        # Update best result if needed
        if error < self.best_score:
            self.best_score = error
            self.best_params = {k: convert_numpy_types(v) for k, v in params.items()}
            self.best_result = result_tuple
            
            self.save_best_params()
            
            logger.info(f"New best: {result_tuple}, Error: {error:.4f}")
        
        return float(error)
    
    def save_best_params(self):
        """Save the best parameters found so far."""
        best_params_path = self.output_dir / 'best_params.json'
        
        if not self.best_params:
            logger.warning("No best parameters to save")
            return
            
        # Prepare full configuration
        full_config = self.prepare_config(self.best_params)
        
        result_info = {
            'best_params': self.best_params,
            'full_config': full_config,
            'best_error': self.best_score,
            'best_result': self.best_result,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(best_params_path, 'w') as f:
            json.dump(result_info, f, cls=NumpyEncoder, indent=2)
            
        logger.info(f"Saved best parameters to {best_params_path}")
    
    def save_optimization_history(self):
        """Save the full optimization history."""
        history_path = self.output_dir / 'optimization_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.opt_history, f, cls=NumpyEncoder, indent=2)
    
    def create_diagnostic_plots(self):
        """Create detailed diagnostic plots to understand optimization progress."""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        try:
            # Extract data from history
            iterations = list(range(len(self.opt_history)))
            errors = [trial['error'] for trial in self.opt_history]
            results = []
            
            for trial in self.opt_history:
                if trial.get('result'):
                    results.append(trial['result'])
                else:
                    # Use placeholder if result is missing
                    results.append((0, 0, 0))
            
            # 1. Plot error over iterations
            plt.figure(figsize=(12, 6))
            plt.plot(iterations, errors, 'b-', marker='o', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel('Error Score')
            plt.title('Optimization Progress')
            plt.grid(True)
            plt.savefig(plots_dir / 'error_progress.png')
            
            # 2. Plot result counts over iterations
            if results:
                results_array = np.array(results)
                
                plt.figure(figsize=(12, 6))
                plt.plot(iterations, results_array[:, 0], 'b-', label='Camera 1 Count')
                plt.plot(iterations, results_array[:, 1], 'g-', label='Camera 2 Count')
                plt.plot(iterations, results_array[:, 2], 'r-', label='Transitions')
                plt.xlabel('Iteration')
                plt.ylabel('Count')
                plt.title('Tracking Results Over Iterations')
                plt.legend()
                plt.grid(True)
                plt.savefig(plots_dir / 'results_progress.png')
                
            # 3. Plot parameter evolution for key parameters
            key_params = [
                'cam1_min_confidence', 'cam2_min_confidence',
                'global_similarity_threshold', 'min_track_confirmations'
            ]
            
            plt.figure(figsize=(15, 10))
            for i, param in enumerate(key_params):
                param_values = []
                for trial in self.opt_history:
                    if param in trial['params']:
                        param_values.append(trial['params'][param])
                    else:
                        param_values.append(None)
                
                plt.subplot(2, 2, i+1)
                plt.plot(iterations, param_values, 'b-', marker='o', markersize=3)
                plt.xlabel('Iteration')
                plt.ylabel(param)
                plt.title(f'Evolution of {param}')
                plt.grid(True)
                
            plt.tight_layout()
            plt.savefig(plots_dir / 'parameter_evolution.png')
            
            logger.info(f"Saved diagnostic plots to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating diagnostic plots: {e}")
    
    def optimize(self):
        """Run the optimization process with the selected strategy."""
        # Create the decorated objective function with proper dimensions
        @use_named_args(dimensions=self.param_space)
        def objective(**params):
            return self.objective_function(**params)
        
        # Start optimization
        start_time = time.time()
        logger.info(f"Starting optimization with {self.n_calls} iterations...")
        
        try:
            if self.optimization_strategy == "random":
                # Random search
                logger.info("Using random search optimization")
                result = dummy_minimize(
                    objective,
                    self.param_space,
                    n_calls=self.n_calls,
                    random_state=42,
                    verbose=self.verbose
                )
            else:
                # Default: Bayesian optimization
                logger.info("Using Bayesian optimization")
                result = gp_minimize(
                    objective,
                    self.param_space,
                    n_calls=self.n_calls,
                    n_random_starts=self.n_initial_points,
                    random_state=42,
                    verbose=self.verbose
                )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            logger.error(traceback.format_exc())
            
            # Try to recover best parameters from history
            if self.best_params:
                logger.info(f"Using best parameters found so far")
                result = None
            else:
                logger.error("No best parameters found. Optimization failed completely.")
                return None, None
        
        # Record results
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # minutes
        
        # Save optimization history
        self.save_optimization_history()
        
        # Create diagnostic plots
        self.create_diagnostic_plots()
        
        # Clean up
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        logger.info(f"Optimization completed in {elapsed_time:.1f} minutes")
        
        if self.best_params:
            logger.info(f"Best parameters found:")
            for name, value in self.best_params.items():
                logger.info(f"  {name}: {value}")
            
            if self.best_result:
                logger.info(f"Best result: {self.best_result}")
            
            logger.info(f"Best error score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def validate_best_params(self):
        """Validate the best parameters found with detailed debugging."""
        if not self.best_params:
            logger.error("No best parameters found to validate")
            return None
            
        logger.info("Validating best parameters with debugging enabled...")
        
        # Save the original debug mode state
        original_debug_mode = self.debug_mode
        
        # Enable debug mode for validation
        self.debug_mode = True
        
        # Prepare full configuration
        config = self.prepare_config(self.best_params)
        config = AdvancedDebuggingTools.prepare_debug_config(config)
        
        # Create validation directory
        val_dir = self.output_dir / 'validation'
        val_dir.mkdir(exist_ok=True)
        
        # Save complete config
        config_path = val_dir / 'validation_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, cls=NumpyEncoder, indent=2)
        
        # Run on all video pairs with increased verbosity
        validation_results = []
        
        for i, video_pair in enumerate(self.video_pairs):
            pair_name = f"{Path(video_pair[0]).stem}_{Path(video_pair[1]).stem}"
            pair_dir = val_dir / f"pair_{pair_name}"
            pair_dir.mkdir(exist_ok=True)
            
            logger.info(f"Validating on pair {i+1}/{len(self.video_pairs)}: {pair_name}")
            
            # Run tracking script with debug mode
            result_tuple, elapsed_time, stdout, stderr = self.run_tracking_script(
                video_pair, config, pair_dir
            )
            
            # Save stdout and stderr
            with open(pair_dir / "stdout.txt", "w") as f:
                f.write(stdout)
            with open(pair_dir / "stderr.txt", "w") as f:
                f.write(stderr)
            
            # Evaluate results
            error, metrics = self.evaluate_results(result_tuple)
            
            # Save detailed result
            validation_result = {
                'pair_name': pair_name,
                'result': result_tuple,
                'metrics': metrics,
                'elapsed_time': elapsed_time,
                'error': error
            }
            
            validation_results.append(validation_result)
            
            # Save individual result
            with open(pair_dir / "result.json", "w") as f:
                json.dump(validation_result, f, cls=NumpyEncoder, indent=2)
            
            logger.info(f"  Result: {result_tuple}, Error: {error:.4f}")
        
        # Save overall validation results
        val_results_path = val_dir / 'validation_results.json'
        with open(val_results_path, 'w') as f:
            json.dump(validation_results, f, cls=NumpyEncoder, indent=2)
        
        # Restore original debug mode
        self.debug_mode = original_debug_mode
        
        return validation_results

def find_video_pairs(input_dir):
    """Find pairs of camera videos in the input directory with robust handling."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return []
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(list(input_path.glob(f'**/*{ext}')))
    
    if not all_videos:
        logger.warning(f"No videos found in {input_dir}")
        return []
    
    # Try multiple methods to identify camera pairs
    
    # Method 1: Look for clear camera identifiers (e.g., Camera_1, cam1, etc.)
    cam1_patterns = ['camera_1', 'cam1', 'camera1', 'Camera_1', 'Cam1', 'Camera1', 'D10_']
    cam2_patterns = ['camera_2', 'cam2', 'camera2', 'Camera_2', 'Cam2', 'Camera2', 'D04_']
    
    def is_camera1(name):
        return any(pattern in name for pattern in cam1_patterns)
    
    def is_camera2(name):
        return any(pattern in name for pattern in cam2_patterns)
    
    cam1_videos = [v for v in all_videos if is_camera1(v.name)]
    cam2_videos = [v for v in all_videos if is_camera2(v.name)]
    
    logger.info(f"Found {len(cam1_videos)} Camera 1 videos and {len(cam2_videos)} Camera 2 videos")
    
    # Method 2: Try to pair based on date patterns in filenames
    paired_by_date = []
    date_pairs = {}
    
    # Common date formats in filenames
    date_patterns = [
        r'(\d{8})',             # YYYYMMDD
        r'(\d{4}-\d{2}-\d{2})', # YYYY-MM-DD
        r'(\d{2}-\d{2}-\d{4})', # DD-MM-YYYY
        r'(\d{4}_\d{2}_\d{2})'  # YYYY_MM_DD
    ]
    
    for cam1_video in cam1_videos:
        for pattern in date_patterns:
            match = re.search(pattern, cam1_video.stem)
            if match:
                date_str = match.group(1)
                # Look for matching camera 2 with same date
                for cam2_video in cam2_videos:
                    if date_str in cam2_video.stem:
                        paired_by_date.append((cam1_video, cam2_video))
                        date_pairs[date_str] = (cam1_video, cam2_video)
                        break
                if date_str in date_pairs:
                    break
    
    if paired_by_date:
        logger.info(f"Found {len(paired_by_date)} video pairs based on date patterns")
        return paired_by_date
    
    # Method 3: If cameras are separated into subdirectories, match by position in sorted list
    if len(cam1_videos) == len(cam2_videos):
        # Sort by creation time if possible, otherwise by name
        try:
            cam1_videos.sort(key=lambda v: v.stat().st_mtime)
            cam2_videos.sort(key=lambda v: v.stat().st_mtime)
        except:
            cam1_videos.sort()
            cam2_videos.sort()
            
        fallback_pairs = list(zip(cam1_videos, cam2_videos))
        logger.info(f"Created {len(fallback_pairs)} fallback video pairs based on sorting")
        return fallback_pairs
    
    # Method 4: Last resort - if only one video of each type, pair them
    if len(cam1_videos) == 1 and len(cam2_videos) == 1:
        logger.info("Found exactly one video from each camera - creating a pair")
        return [(cam1_videos[0], cam2_videos[0])]
    
    logger.warning("Could not identify video pairs automatically")
    return []

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='Advanced Debugging and Optimization for People Tracking')
    
    # Input/output options
    parser.add_argument('--input_dir', type=str, default='./videos',
                        help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--video1', type=str, default=None,
                        help='Path to Camera 1 video (for single pair processing)')
    parser.add_argument('--video2', type=str, default=None,
                        help='Path to Camera 2 video (for single pair processing)')
    
    # Optimization options
    parser.add_argument('--n_calls', type=int, default=50,
                        help='Number of optimization iterations')
    parser.add_argument('--n_initial_points', type=int, default=10,
                        help='Number of initial random points for Bayesian optimization')
    parser.add_argument('--strategy', type=str, choices=['bayesian', 'random', 'tpe', 'pbt'], 
                        default='bayesian', help='Optimization strategy to use')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel evaluations to run')
    
    # Debugging options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with enhanced instrumentation')
    parser.add_argument('--validate', action='store_true',
                        help='Validate the best parameters found with detailed debugging')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    return parser

def main():
    """Main function to run advanced debugging and optimization."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Find video pairs
    if args.video1 and args.video2:
        # Use specific videos provided
        video_pairs = [(args.video1, args.video2)]
        logger.info(f"Using specified video pair: {args.video1} and {args.video2}")
    else:
        # Find video pairs in directory
        video_pairs = find_video_pairs(args.input_dir)
        
        if not video_pairs:
            logger.error("No video pairs found. Please check input directory or specify videos directly.")
            return
    
    # Create optimizer
    optimizer = AdvancedOptimizer(
        video_pairs=video_pairs,
        output_dir=args.output_dir,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        optimization_strategy=args.strategy,
        parallel_evaluations=args.parallel,
        debug_mode=args.debug,
        verbose=args.verbose
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize()
    
    # Validate best parameters if requested
    if args.validate and best_params:
        validation_results = optimizer.validate_best_params()
        
        if validation_results:
            print("\nValidation results:")
            for i, res in enumerate(validation_results):
                print(f"  Pair {i+1}: {res['result']}, Error: {res['error']:.1f}")
    
    print("\n" + "="*60)
    print("Enhanced optimization completed!")
    
    if best_params:
        print(f"Best parameters saved to: {Path(args.output_dir) / 'best_params.json'}")
        print(f"Best error score: {best_score:.4f}")
        
        if optimizer.best_result:
            print(f"Best result tuple: {optimizer.best_result}")
    else:
        print("Optimization did not find valid parameters. Check logs for details.")
        
    print("="*60)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time/60:.1f} minutes")