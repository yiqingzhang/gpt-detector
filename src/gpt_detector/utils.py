"""
Utility functions for model training, evaluation, and configuration management.
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def parse_args():
    """Load and parse configuration arguments from a JSON file.
    
    This function creates an argument parser that accepts a config file path,
    loads the JSON configuration, and merges it with command-line arguments.
    
    Returns:
        Namespace object containing all configuration parameters.
    
    Example:
        >>> args = parse_args()
        >>> print(args.epochs)
        10
    """
    parser = argparse.ArgumentParser(
        description="GPT Detector - Configuration Parser"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/gpt_detector/config/hyperparameters.json",
        help="Path to the configuration JSON file",
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            data = json.load(f)

        for key in data:
            args.__dict__[key] = data[key]

    return args


def save_checkpoint(path: str, model: nn.Module) -> None:
    """Save model checkpoint to disk.
    
    Args:
        path: File path where the checkpoint will be saved.
        model: PyTorch model to save.
    
    Example:
        >>> model = ROBERTAClassifier()
        >>> save_checkpoint("model.pkl", model)
    """
    torch.save({"model_state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, model: nn.Module) -> nn.Module:
    """Load model checkpoint from disk.
    
    The model is loaded to CPU first to improve loading speed and memory efficiency,
    then can be moved to the desired device afterward.
    
    Args:
        path: File path to the saved checkpoint.
        model: PyTorch model instance to load the weights into.
    
    Returns:
        Model with loaded weights.
    
    Example:
        >>> model = ROBERTAClassifier()
        >>> model = load_checkpoint("model.pkl", model)
    """
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict["model_state_dict"])
    return model


def save_metrics(
    path: str,
    train_loss_list: List[float],
    valid_loss_list: Optional[List[float]] = None,
    best_valid_loss: Optional[float] = None,
) -> None:
    """Save training and validation metrics to disk.
    
    Args:
        path: File path where metrics will be saved.
        train_loss_list: List of training losses per epoch.
        valid_loss_list: List of validation losses per epoch. Optional.
        best_valid_loss: Best validation loss achieved. Optional.
    
    Example:
        >>> save_metrics("metrics.pkl", [0.5, 0.3, 0.2], [0.6, 0.4, 0.3], 0.3)
    """
    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "best_valid_loss": best_valid_loss,
    }
    torch.save(state_dict, path)


def load_metrics(path: str, device: torch.device) -> Tuple[List[float], List[float], float]:
    """Load training and validation metrics from disk.
    
    Args:
        path: File path to the saved metrics.
        device: PyTorch device to load the metrics to.
    
    Returns:
        A tuple containing:
            - train_loss_list: List of training losses
            - valid_loss_list: List of validation losses
            - best_valid_loss: Best validation loss value
    
    Example:
        >>> device = torch.device("cuda")
        >>> train_loss, valid_loss, best_loss = load_metrics("metrics.pkl", device)
    """
    state_dict = torch.load(path, map_location=device)
    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["best_valid_loss"],
    )
