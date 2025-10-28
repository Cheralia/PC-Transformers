import torch
import psutil

def get_optimal_data_sizes():
    """Determine optimal data sizes based on available memory"""
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return (20000, 5000) if mem_gb >= 8 else (2000, 400) if mem_gb >= 4 else (1200, 240)
    else:
        mem_gb = psutil.virtual_memory().total / (1024**3)
        return (1500, 300) if mem_gb >= 16 else (800, 160)
