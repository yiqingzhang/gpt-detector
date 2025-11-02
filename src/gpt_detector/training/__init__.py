"""
Training module for GPT Detector
"""

from .train import train, validation, load_datasets, set_dataloader

__all__ = ["train", "validation", "load_datasets", "set_dataloader"]

