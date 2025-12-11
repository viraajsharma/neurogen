"""
Logger Module for Neurogen v1.1

This module provides logging utilities for tracking training metrics
and writing them to CSV files.
"""

import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class TrainingLogger:
    """
    Logger for tracking and saving training metrics.
    
    Logs metrics to CSV file with timestamps and iteration numbers.
    
    Attributes:
        log_dir (str): Directory where log files are saved
        log_file (str): Path to the current log file
        metrics (List[str]): List of metric names to log
        log_data (List[Dict]): Accumulated log entries
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize the training logger.
        
        Args:
            log_dir: Directory to save log files (default: "logs")
            experiment_name: Name for this experiment (default: timestamp)
            metrics: List of metric names to log (default: standard metrics)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.csv"
        
        # Default metrics
        if metrics is None:
            metrics = [
                "iteration",
                "total_energy",
                "prediction_error",
                "weight_penalty",
                "avg_activation",
                "weight_change_norm"
            ]
        
        self.metrics = metrics
        self.log_data = []
        
        # Initialize CSV file with headers
        self._initialize_csv()
        
        print(f"✓ Logger initialized: {self.log_file}")
    
    def _initialize_csv(self):
        """Create CSV file and write header row."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics)
            writer.writeheader()
    
    def log(self, **kwargs):
        """
        Log metrics for current iteration.
        
        Args:
            **kwargs: Metric name-value pairs to log
        
        Example:
            >>> logger = TrainingLogger()
            >>> logger.log(iteration=0, total_energy=1.5, avg_activation=0.3)
        """
        # Create log entry with all metrics
        log_entry = {}
        for metric in self.metrics:
            log_entry[metric] = kwargs.get(metric, None)
        
        # Store in memory
        self.log_data.append(log_entry)
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics)
            writer.writerow(log_entry)
    
    def log_iteration(
        self,
        iteration: int,
        total_energy: float,
        prediction_error: Optional[float] = None,
        weight_penalty: Optional[float] = None,
        avg_activation: Optional[float] = None,
        weight_change_norm: Optional[float] = None,
        **extra_metrics
    ):
        """
        Log metrics for a training iteration (convenience method).
        
        Args:
            iteration: Current iteration number
            total_energy: Total energy value
            prediction_error: Prediction error component
            weight_penalty: Weight regularization penalty
            avg_activation: Average neuron activation
            weight_change_norm: Norm of weight changes
            **extra_metrics: Additional metrics to log
        """
        metrics = {
            'iteration': iteration,
            'total_energy': total_energy,
            'prediction_error': prediction_error,
            'weight_penalty': weight_penalty,
            'avg_activation': avg_activation,
            'weight_change_norm': weight_change_norm
        }
        
        # Add extra metrics
        metrics.update(extra_metrics)
        
        self.log(**metrics)
    
    def get_log_data(self) -> List[Dict[str, Any]]:
        """
        Get all logged data.
        
        Returns:
            List of log entries
        """
        return self.log_data
    
    def print_summary(self):
        """Print summary of logged metrics."""
        if not self.log_data:
            print("No data logged yet.")
            return
        
        print(f"\n{'='*60}")
        print(f"Training Summary: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total iterations: {len(self.log_data)}")
        
        # Print final metrics
        final_entry = self.log_data[-1]
        print(f"\nFinal metrics:")
        for metric, value in final_entry.items():
            if value is not None:
                print(f"  {metric}: {value:.6f}" if isinstance(value, float) else f"  {metric}: {value}")
        
        print(f"\nLog file: {self.log_file}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], iteration: int):
        """
        Save a checkpoint of the network state.
        
        Args:
            checkpoint_data: Dictionary containing network state
            iteration: Current iteration number
        """
        checkpoint_dir = self.log_dir / "checkpoints" / self.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"checkpoint_iter_{iteration}.pkl"
        
        # TODO: Implement checkpoint saving (e.g., using pickle or torch.save)
        # import pickle
        # with open(checkpoint_file, 'wb') as f:
        #     pickle.dump(checkpoint_data, f)
        
        print(f"✓ Checkpoint saved: {checkpoint_file}")
