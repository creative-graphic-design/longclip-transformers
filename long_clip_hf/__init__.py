"""
LongCLIP: Transformers-compatible implementation.

This package provides transformers-compatible versions of LongCLIP models.
"""

from .configuration_longclip import (
    LongCLIPConfig,
    LongCLIPTextConfig,
    LongCLIPVisionConfig,
)

__all__ = [
    "LongCLIPConfig",
    "LongCLIPTextConfig",
    "LongCLIPVisionConfig",
]
