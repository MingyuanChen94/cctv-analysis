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
from skopt import gp_minimize
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

class BayesianOptimizer:
    """Bayesian optimization for parameter tuning of people tracking script."""
    
    def __init__(self, video1_path, video2_path, output_dir, n_calls=50, verbose=True):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            video1_path: Path to Camera 1 video
            video2_path: Path to Camera 2 video
            output_dir: Directory to save results
            n_calls: Number of optimization iterations
            verbose: Whether to print progress information
        """
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.output_dir = output_dir
        self.n_calls = n_calls
        self.verbose = verbose
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
    
    def objective_function_impl(self, **params):
        """
        Objective function implementation to be minimized.
        
        Args:
            **params: Parameter values from optimization

        Returns:
            float: Error metric (distance from target values)
        """
        # Create a config dictionary from the parameters
        config = {name: value for name, value in params.items()}
        
        # Add some fixed parameters (these could be moved to optimization space if needed)
        config.update({
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
            
            # Camera topology information
            'camera_topology': {
                "1": {
                    "2": {
                        'direction': 'right_to_left',
                        'avg_transit_time': (config['cam1_to_cam2_min_time'] + config['cam1_to_cam2_max_time']) // 2,
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
        })
        
        # Save config to a temporary file
        config_path = Path(self.output_dir) / 'temp_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Run the tracking script with the current configuration
        trial_dir = Path(self.output_dir) / f'trial_{int(time.time())}'
        os.makedirs(trial_dir, exist_ok=True)
        
        try:
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
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extract the result tuple from the output
            output_lines = result.stdout.split('\n')
            result_line = [line for line in output_lines if "Result tuple:" in line]
            
            if not result_line:
                print("Warning: Could not find result tuple in output")
                return float('inf')  # Return a high value if no results found
            
            # Parse result tuple
            result_str = result_line[0].split("Result tuple:")[1].strip()
            result_tuple = eval(result_str)  # Convert string tuple to actual tuple
            
            # Calculate error (distance from target)
            camera1_count, camera2_count, transitions = result_tuple
            
            # Weighted mean squared error
            mse = (
                5 * ((camera1_count - TARGET_CAMERA1) / TARGET_CAMERA1) ** 2 +
                5 * ((camera2_count - TARGET_CAMERA2) / TARGET_CAMERA2) ** 2 +
                10 * ((transitions - TARGET_TRANSITIONS) / max(1, TARGET_TRANSITIONS)) ** 2
            )
            
            # Store trial results
            trial_result = {
                'params': params,
                'result': result_tuple,
                'error': mse
            }
            self.opt_history.append(trial_result)
            
            # Update best result if needed
            if mse < self.best_score:
                self.best_score = mse
                self.best_params = params
                self.best_result = result_tuple
                self.save_best_params()
                
                if self.verbose:
                    print(f"New best: {result_tuple}, Error: {mse:.4f}")
            
            return mse
            
        except Exception as e:
            print(f"Error running script: {e}")
            return float('inf')  # Return a high value in case of errors
    
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
            json.dump(result_info, f, indent=4)
    
    def save_optimization_history(self):
        """Save the full optimization history."""
        history_path = Path(self.output_dir) / 'optimization_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.opt_history, f, indent=4)
    
    def optimize(self):
        """Run the Bayesian optimization."""
        # Create the decorated objective function with proper dimensions
        @use_named_args(dimensions=self.param_space)
        def objective(**params):
            return self.objective_function_impl(**params)
        
        # Start optimization
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting Bayesian optimization with {self.n_calls} iterations...")
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            self.param_space,
            n_calls=self.n_calls,
            random_state=42,
            verbose=self.verbose,
            n_random_starts=10
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
                print(f"  {name}: {value}")
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
        verbose=args.verbose
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