import os
import random
import numpy as np
import torch
from PIL import Image
import time
from functools import wraps

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_mask(mask, path):
    """
    Save mask as PNG image
    Args:
        mask: numpy array of shape (H, W) with values 0-255
        path: output file path
    """
    ensure_dir(os.path.dirname(path))
    Image.fromarray(mask).save(path)

def timeit(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)
