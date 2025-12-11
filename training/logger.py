import csv
import os
import time

class CSVLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filepath = os.path.join(log_dir, f"run_{timestamp}.csv")
        
        # Initialize file with headers
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "total_energy", "avg_activation", "weight_change_norm"])
            
    def log(self, iteration, total_energy, avg_activation, weight_change_norm):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, total_energy, avg_activation, weight_change_norm])
            
    def close(self):
        pass # Nothing to explicitly close as we open/close on write
