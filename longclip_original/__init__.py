"""
Original LongCLIP implementation.
This package is used as a baseline for testing the transformers-compatible version.
"""

from .model import longclip
from .model.model_longclip import CLIP

__all__ = ["longclip", "CLIP"]
