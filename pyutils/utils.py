"""General utilities for training."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
