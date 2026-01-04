"""
LongCLIP configuration classes.

These configuration classes extend the standard CLIP configuration to support
the extended context length and custom positional embeddings of LongCLIP.
"""

from typing import Dict, Any
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.configuration_utils import PretrainedConfig


class LongCLIPTextConfig(CLIPTextConfig):
    """
    Configuration class for LongCLIP text model.

    Extends CLIPTextConfig to support 248 token context length
    and custom positional embedding interpolation.

    Args:
        max_position_embeddings (int, optional): Maximum sequence length. Defaults to 248.
        use_position_interpolation (bool, optional): Whether to use position interpolation.
            Defaults to True.
        interpolation_keep_length (int, optional): Number of positions to keep from
            original embeddings before interpolation. Defaults to 20.
        **kwargs: Additional arguments passed to CLIPTextConfig.
    """

    model_type = "longclip_text_model"

    def __init__(
        self,
        max_position_embeddings: int = 248,
        use_position_interpolation: bool = True,
        interpolation_keep_length: int = 20,
        **kwargs,
    ):
        super().__init__(max_position_embeddings=max_position_embeddings, **kwargs)

        self.use_position_interpolation = use_position_interpolation
        self.interpolation_keep_length = interpolation_keep_length


class LongCLIPVisionConfig(CLIPVisionConfig):
    """
    Configuration class for LongCLIP vision model.

    This is identical to the standard CLIPVisionConfig as LongCLIP
    does not modify the vision encoder.

    Args:
        **kwargs: Arguments passed to CLIPVisionConfig.
    """

    model_type = "longclip_vision_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LongCLIPConfig(CLIPConfig):
    """
    Configuration class for LongCLIP model.

    Combines LongCLIPTextConfig and LongCLIPVisionConfig to create
    a complete LongCLIP model configuration.

    Args:
        text_config (Dict[str, Any] or LongCLIPTextConfig, optional):
            Configuration for the text model. If None, uses default LongCLIPTextConfig.
        vision_config (Dict[str, Any] or LongCLIPVisionConfig, optional):
            Configuration for the vision model. If None, uses default LongCLIPVisionConfig.
        projection_dim (int, optional): Dimensionality of text and vision projection layers.
            Defaults to 512.
        **kwargs: Additional arguments passed to CLIPConfig.

    Example:
        ```python
        >>> from long_clip_hf import LongCLIPConfig
        >>> # Initialize with default settings
        >>> config = LongCLIPConfig()
        >>>
        >>> # Initialize with custom text config
        >>> text_config = {"max_position_embeddings": 248, "hidden_size": 512}
        >>> config = LongCLIPConfig(text_config=text_config)
        >>>
        >>> # Save config
        >>> config.save_pretrained("./my-longclip-config")
        >>>
        >>> # Load config
        >>> config = LongCLIPConfig.from_pretrained("./my-longclip-config")
        ```
    """

    model_type = "longclip"
    is_composition = True

    def __init__(
        self,
        text_config: Dict[str, Any] | None = None,
        vision_config: Dict[str, Any] | None = None,
        projection_dim: int = 512,
        **kwargs,
    ):
        # Initialize text config
        if text_config is None:
            text_config = {}
            logger.info(
                "text_config is None. Initializing the LongCLIPTextConfig with default values."
            )

        if vision_config is None:
            vision_config = {}
            logger.info(
                "vision_config is None. Initializing the LongCLIPVisionConfig with default values."
            )

        # Create config objects if they're dictionaries
        if isinstance(text_config, dict):
            text_config = LongCLIPTextConfig(**text_config)

        if isinstance(vision_config, dict):
            vision_config = LongCLIPVisionConfig(**vision_config)

        # Call parent init with config dicts
        super().__init__(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            projection_dim=projection_dim,
            **kwargs,
        )

        # Store as config objects for easier access
        self.text_config = text_config
        self.vision_config = vision_config

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: LongCLIPTextConfig,
        vision_config: LongCLIPVisionConfig,
        **kwargs,
    ):
        """
        Instantiate a LongCLIPConfig from text and vision configs.

        Args:
            text_config (LongCLIPTextConfig): Text model configuration.
            vision_config (LongCLIPVisionConfig): Vision model configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            LongCLIPConfig: Configuration object.
        """
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary of all attributes.
        """
        output = super().to_dict()
        # Ensure text_config and vision_config are properly serialized
        if hasattr(self, "text_config") and isinstance(
            self.text_config, PretrainedConfig
        ):
            output["text_config"] = self.text_config.to_dict()
        if hasattr(self, "vision_config") and isinstance(
            self.vision_config, PretrainedConfig
        ):
            output["vision_config"] = self.vision_config.to_dict()
        return output


# For logging
import logging

logger = logging.getLogger(__name__)
