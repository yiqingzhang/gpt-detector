"""
GPT Detector - AI-generated text detection using RoBERTa
"""

__version__ = "0.1.0"

from .model import ROBERTAClassifier
from .utils import load_checkpoint, save_checkpoint, parse_args

__all__ = ["ROBERTAClassifier", "load_checkpoint", "save_checkpoint", "parse_args"]

