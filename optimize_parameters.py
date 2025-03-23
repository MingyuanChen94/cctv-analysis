#!/usr/bin/env python3
"""
Parameter optimization for cross-camera people tracking using Bayesian Optimization.
Finds optimal parameters to reach the target count values (25, 12, 2).
"""

import os
import time
import numpy as np
import json
from pathlib import Path
import subprocess
import argparse
import re
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Target values for optimization
TARGET_CAMERA1 = 25
TARGET_CAMERA2 = 12
TARGET_TRANSITIONS = 2

# Script to be optimized
SCRIPT_PATH = "v15.py"  # Adjust if necessary

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

class BayesianOptimizer:
    """Bayesian optimization for parameter tuning of people tracking script."""
    
    def __init__(self, video1_path, video2_path, output_dir, n_calls=50, verbose=True, debug=False):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            video1_path: Path to Camera 1 video
            video2_path: Path to Camera 2 video
            output_dir: Directory to save results
            n_calls: Number of optimization iterations
            verbose: Whether to print progress information
            debug: Whether to print detailed debugging information
        """
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.output_dir = output_dir
        self.n_calls = n_calls
        self.verbose = verbose
        self.debug = debug
        self.opt_history = []
        self.best_score = float('inf')
        self.best_params = None
        self.best_result = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the parameter space
        self.define_parameter_space()
    
    def define_parameter_space(self):
        """Define the space of parameters to optimize."""
        # We'll focus on a subset of key parameters to keep optimization manageable
        self.param_space = [
            # Camera 1 parameters (more complex environment)
            Real(0.3, 0.8, name='cam1_min_confidence'),
            Real(0.5, 0.9, name='cam1_similarity_threshold'),
            Integer(1, 5, name='cam1_max_disappear_seconds'),
            Real(0.1, 0.6, name='cam1_deep_feature_weight'),
            Real(0.05, 0.3, name='cam1_position_weight'),
            Real(0.05, 0.3, name='cam1_color_weight'),
            Real(0.5, 0.9, name='cam1_reentry_threshold'),
            Real(0.5, 0.9, name='cam1_new_track_confidence'),
            
            # Camera 2 parameters (cleaner environment)
            Real(0.2, 0.7, name='cam2_min_confidence'),
            Real(0.4, 0.9, name='cam2_similarity_threshold'),
            Integer(1, 5, name='cam2_max_disappear_seconds'),
            Real(0.1, 0.6, name='cam2_deep_feature_weight'),
            Real(0.05, 0.4, name='cam2_position_weight'),
            Real(0.05, 0.3, name='cam2_color_weight'),
            Real(0.5, 0.9, name='cam2_reentry_threshold'),
            Real(0.4, 0.8, name='cam2_new_track_confidence'),
            
            # Common tracking parameters
            Integer(3, 20, name='max_lost_seconds'),
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
        Extract result tuple from command output using regular expressions.
        Falls back to alternative extraction methods if the standard one fails.
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
        
        # Failed to extract
        return None
    
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
        
        # Save config to a temporary file
        config_path = Path(self.output_dir) / 'temp_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, cls=NumpyEncoder)
        
        # Create unique trial directory
        trial_dir = Path(self.output_dir) / f'trial_{int(time.time())}'
        os.makedirs(trial_dir, exist_ok=True)
        
        try:
            # Build command to run the tracking script
            cmd = [
                "python", SCRIPT_PATH,
                "--video1", self.video1_path,
                "--video2", self.video2_path,
                "--output_dir", str(trial_dir),
                "--config", str(config_path)
            ]
            
            # Run the script and capture output
            if self.verbose:
                print(f"Running trial with parameters: {params}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Save the output for debugging
            with open(trial_dir / "output.txt", "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            
            # Print output for debugging if requested
            if self.debug:
                print(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            
            # Extract the result tuple from the output
            result_tuple = self.extract_result_tuple(result.stdout)
            
            if not result_tuple:
                print(f"Could not extract result tuple from output. Using default values.")
                
                # Use defaults that are not identical to targets but in the ballpark
                # to help the optimization process explore
                result_tuple = (20, 10, 1)
                
                # Save a flag file to mark this trial as using defaults
                with open(trial_dir / "used_defaults.txt", "w") as f:
                    f.write("True")
            
            # Calculate error (distance from target)
            camera1_count, camera2_count, transitions = result_tuple
            
            # Weighted mean squared error
            mse = (
                5 * ((camera1_count - TARGET_CAMERA1) / TARGET_CAMERA1) ** 2 +
                5 * ((camera2_count - TARGET_CAMERA2) / TARGET_CAMERA2) ** 2 +
                10 * ((transitions - TARGET_TRANSITIONS) / max(1, TARGET_TRANSITIONS)) ** 2
            )
            
            # Ensure the error is finite
            if not np.isfinite(mse):
                mse = 1000.0  # Large but finite penalty
            
            # Store trial results with converted params
            trial_result = {
                'params': {k: convert_numpy_types(v) for k, v in params.items()},
                'result': result_tuple,
                'error': float(mse)
            }
            self.opt_history.append(trial_result)
            
            # Update best result if needed
            if mse < self.best_score:
                self.best_score = float(mse)
                self.best_params = {k: convert_numpy_types(v) for k, v in params.items()}
                self.best_result = result_tuple
                self.save_best_params()
                
                if self.verbose:
                    print(f"New best: {result_tuple}, Error: {mse:.4f}")
            
            return float(mse)
            
        except subprocess.TimeoutExpired:
            print("Script execution timed out after 300 seconds")
            return 1000.0  # Large but finite penalty
        except Exception as e:
            print(f"Error running script: {e}")
            return 1000.0  # Large but finite penalty
    
    def save_best_params(self):
        """Save the best parameters found so far."""
        best_params_path = Path(self.output_dir) / 'best_params.json'
        result_info = {
            'best_params': self.best_params,
            'best_result': self.best_result,
            'error': self.best_score,
            'target': (TARGET_CAMERA1, TARGET_CAMERA2, TARGET_TRANSITIONS)
        }
        with open(best_params_path, 'w') as f:
            json.dump(result_info, f, cls=NumpyEncoder, indent=2)
    
    def save_optimization_history(self):
        """Save the full optimization history."""
        history_path = Path(self.output_dir) / 'optimization_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.opt_history, f, cls=NumpyEncoder, indent=2)
    
    def optimize(self):
        """Run the optimization. If GP fails, fall back to random search."""
        # Create the decorated objective function with proper dimensions
        @use_named_args(dimensions=self.param_space)
        def objective(**params):
            return self.objective_function_impl(**params)
        
        # Start optimization
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting optimization with {self.n_calls} iterations...")
        
        try:
            # First try Bayesian optimization
            if self.verbose:
                print("Using Gaussian Process optimization")
                
            result = gp_minimize(
                objective,
                self.param_space,
                n_calls=self.n_calls,
                random_state=42,
                verbose=self.verbose,
                n_random_starts=10,
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
            print(f"Optimization completed in {end_time - start_time:.2f} seconds")
            print(f"Best parameters found:")
            for name, value in zip(self.param_names, result.x):
                print(f"  {name}: {convert_numpy_types(value)}")
            print(f"Best result: {self.best_result}")
            print(f"Error: {self.best_score:.4f}")
        
        return self.best_params, self.best_result
    
    def plot_optimization_results(self):
        """Plot optimization convergence and parameter importance."""
        if not hasattr(self, 'result'):
            print("Run optimization first")
            return
        
        # Create plots directory
        plots_dir = Path(self.output_dir) / 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plot_convergence(self.result)
        plt.title('Optimization Convergence')
        plt.savefig(plots_dir / 'convergence.png')
        
        # Plot parameter values over iterations
        plt.figure(figsize=(12, 10))
        plt.subplot(211)
        iterations = list(range(len(self.opt_history)))
        
        # Extract results for plotting
        if len(self.opt_history) > 0:
            results = np.array([trial['result'] for trial in self.opt_history])
            
            # Plot counts over iterations
            plt.plot(iterations, results[:, 0], 'b-', label='Camera 1')
            plt.plot(iterations, results[:, 1], 'g-', label='Camera 2')
            plt.plot(iterations, results[:, 2], 'r-', label='Transitions')
            
            # Add target lines
            plt.axhline(y=TARGET_CAMERA1, color='b', linestyle='--', label=f'Target Camera 1: {TARGET_CAMERA1}')
            plt.axhline(y=TARGET_CAMERA2, color='g', linestyle='--', label=f'Target Camera 2: {TARGET_CAMERA2}')
            plt.axhline(y=TARGET_TRANSITIONS, color='r', linestyle='--', label=f'Target Transitions: {TARGET_TRANSITIONS}')
            
            plt.xlabel('Iteration')
            plt.ylabel('Count')
            plt.title('Count Results Over Iterations')
            plt.legend()
            plt.grid(True)
            
            # Plot error over iterations
            plt.subplot(212)
            errors = [trial['error'] for trial in self.opt_history]
            plt.plot(iterations, errors, 'k-')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.title('Error Over Iterations')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'progress.png')
            
            if self.verbose:
                print(f"Plots saved to {plots_dir}")


def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='Bayesian Optimization for People Tracking Parameters')
    
    # Input/output options
    parser.add_argument('--video1', type=str, required=True,
                        help='Path to Camera 1 video')
    parser.add_argument('--video2', type=str, required=True,
                        help='Path to Camera 2 video')
    parser.add_argument('--output_dir', type=str, default='./optimization_results',
                        help='Directory to save optimization results')
    
    # Optimization options
    parser.add_argument('--n_calls', type=int, default=50,
                        help='Number of optimization iterations')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    return parser


def main():
    """Main function to run the optimization."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create and run optimizer
    optimizer = BayesianOptimizer(
        video1_path=args.video1,
        video2_path=args.video2,
        output_dir=args.output_dir,
        n_calls=args.n_calls,
        verbose=args.verbose,
        debug=args.debug
    )
    
    best_params, best_result = optimizer.optimize()
    optimizer.plot_optimization_results()
    
    print("\n" + "="*40)
    print("Optimization completed!")
    print(f"Best parameters saved to: {Path(args.output_dir) / 'best_params.json'}")
    print(f"Best result: {best_result}")
    print(f"Target result: ({TARGET_CAMERA1}, {TARGET_CAMERA2}, {TARGET_TRANSITIONS})")
    print("="*40)


if __name__ == "__main__":
    main()