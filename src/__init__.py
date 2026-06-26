from .data_loader import load_and_clean_data, prepare_splits
from .features import extract_features, FEATURE_NAMES
from .gmm_threshold import get_thresholds_and_categories
from .model import train_models
from .predict import predict_name_gender, print_prediction

__version__ = "1.0.0"
__all__ = [
    "load_and_clean_data", "prepare_splits",
    "extract_features", "FEATURE_NAMES",
    "get_thresholds_and_categories",
    "train_models",
    "predict_name_gender", "print_prediction",
]