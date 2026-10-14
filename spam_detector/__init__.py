"""Spam Detector package.

Loads the SMS spam dataset, trains a text-classification pipeline, and
exposes helpers to predict whether a message is spam or ham.
"""

from .data import load_dataset, split_features_labels
from .train import build_pipeline, train_and_evaluate

__all__ = [
    "load_dataset",
    "split_features_labels",
    "build_pipeline",
    "train_and_evaluate",
]
