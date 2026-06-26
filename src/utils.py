import numpy as np
import random

CAT_PALETTE = {
    'Strong Male': '#2196F3',
    'Unisex': '#FF9800',
    'Strong Female': '#E91E63',
}


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)