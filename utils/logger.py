"""
Logging utilities for NEUROGEN.
"""

import sys
from datetime import datetime
from typing import Optional


class Logger:
    """Simple logger for training progress and results."""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        """
        Args:
            log_file: Optional file path to write logs
            verbose: Whether to print to console
        """
        self.log_file = log_file
        self.verbose = verbose
        self.file_handle = None
        
        if self.log_file:
            self.file_handle = open(self.log_file, 'w')
    
    def log(self, message: str, level: str = 'INFO'):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level ('INFO', 'WARNING', 'ERROR', 'SUCCESS')
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        if self.verbose:
            print(formatted_msg)
        
        if self.file_handle:
            self.file_handle.write(formatted_msg + '\n')
            self.file_handle.flush()
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, 'INFO')
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, 'WARNING')
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, 'ERROR')
    
    def success(self, message: str):
        """Log success message."""
        self.log(message, 'SUCCESS')
    
    def close(self):
        """Close log file if open."""
        if self.file_handle:
            self.file_handle.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def print_header(title: str):
    """Print a formatted header."""
    width = 60
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")
