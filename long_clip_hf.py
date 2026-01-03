"""
LongCLIP: Unlocking the Long-Text Capability of CLIP

This module provides HuggingFace Transformers-compatible implementations of LongCLIP,
which extends CLIP's text encoder to support 248 tokens (vs 77 in original CLIP).

Repository: https://github.com/beichenzbc/Long-CLIP
Paper: https://arxiv.org/abs/2403.15378
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel
from transformers import CLIPImageProcessor, CLIPTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from transformers.processing_utils import ProcessorMixin

logger = logging.getLogger(__name__)


# ================== Configuration Classes ==================


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


# ================== Model Classes ==================


class LongCLIPTextEmbeddings(nn.Module):
    """
    Text embeddings for LongCLIP with custom positional embedding mechanism.

    This module implements the dual positional embedding approach used in LongCLIP:
    - The first 20 positions use the original CLIP positional embeddings (mask1)
    - The remaining positions (21-248) use interpolated embeddings (mask2)
    - position_embedding: Fixed base embeddings
    - position_embedding_res: Trainable residual embeddings

    Args:
        config (LongCLIPTextConfig): Configuration for text embeddings.
    """

    def __init__(self, config: LongCLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)

        # Dual positional embeddings (LongCLIP approach)
        # position_embedding: Base embeddings (typically loaded from checkpoint)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_embedding_res: Trainable residual embeddings
        self.position_embedding_res = nn.Parameter(
            torch.zeros(config.max_position_embeddings, embed_dim)
        )

        # Create masks for applying embeddings
        # mask1: Use original embeddings for first interpolation_keep_length positions
        # mask2: Use interpolated embeddings for remaining positions
        self.register_buffer(
            "mask1", self._create_mask(config, use_first=True), persistent=False
        )
        self.register_buffer(
            "mask2", self._create_mask(config, use_first=False), persistent=False
        )

        # Store position IDs for efficiency
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def _create_mask(self, config: LongCLIPTextConfig, use_first: bool) -> torch.Tensor:
        """
        Create mask for positional embeddings.

        Args:
            config: Configuration object.
            use_first: If True, mask first `interpolation_keep_length` positions.
                      If False, mask remaining positions.

        Returns:
            Mask tensor of shape [max_position_embeddings, 1].
        """
        mask = torch.zeros(config.max_position_embeddings, 1)
        if use_first:
            # mask1: First interpolation_keep_length positions
            mask[: config.interpolation_keep_length] = 1.0
        else:
            # mask2: Remaining positions
            mask[config.interpolation_keep_length :] = 1.0
        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for text embeddings.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_length].
            position_ids: Position IDs of shape [batch_size, seq_length].
            inputs_embeds: Pre-computed token embeddings.

        Returns:
            Embeddings of shape [batch_size, seq_length, hidden_size].
        """
        seq_length = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Get token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # Get positional embeddings
        position_embeddings = self.position_embedding(position_ids)

        # Add residual positional embeddings (for positions > interpolation_keep_length)
        # Expand position_embedding_res for batch dimension
        position_embeddings_res = self.position_embedding_res.unsqueeze(0).expand(
            position_ids.shape[0], -1, -1
        )[:, :seq_length, :]

        # Apply masks: mask1 for first 20, mask2 for rest
        # Broadcasting: [seq_length, 1] * [batch, seq_length, hidden_size]
        mask1 = self.mask1[:seq_length].transpose(0, 1)  # [1, seq_length]
        mask2 = self.mask2[:seq_length].transpose(0, 1)  # [1, seq_length]

        # Combine embeddings with masking
        embeddings = (
            inputs_embeds
            + position_embeddings * mask1.unsqueeze(-1)
            + position_embeddings_res * mask2.unsqueeze(-1)
        )

        return embeddings


class LongCLIPTextTransformer(CLIPTextTransformer):
    """
    Text transformer for LongCLIP.

    This extends CLIPTextTransformer to use LongCLIPTextEmbeddings
    with custom positional embedding mechanism.

    Args:
        config (LongCLIPTextConfig): Configuration for text transformer.
    """

    def __init__(self, config: LongCLIPTextConfig):
        super().__init__(config)
        # Replace embeddings with LongCLIP version
        self.embeddings = LongCLIPTextEmbeddings(config)


class LongCLIPTextModel(CLIPTextModel):
    """
    LongCLIP text model compatible with HuggingFace Transformers.

    This model extends CLIPTextModel to support 248 token context length
    with custom positional embedding interpolation.

    Args:
        config (LongCLIPTextConfig): Configuration for the text model.
    """

    config_class = LongCLIPTextConfig

    def __init__(self, config: LongCLIPTextConfig):
        super().__init__(config)
        # Replace text_model with LongCLIP version
        self.text_model = LongCLIPTextTransformer(config)
        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Get token embedding layer."""
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value: nn.Module):
        """Set token embedding layer."""
        self.text_model.embeddings.token_embedding = value


class LongCLIPVisionModel(CLIPVisionModel):
    """
    LongCLIP vision model.

    This is identical to CLIPVisionModel as LongCLIP does not modify
    the vision encoder. Provided for API consistency.

    Args:
        config (LongCLIPVisionConfig): Configuration for the vision model.
    """

    config_class = LongCLIPVisionConfig


class LongCLIPModel(CLIPModel):
    """
    LongCLIP model combining text and vision encoders.

    This model extends CLIPModel to use LongCLIPTextModel with 248 token
    context length while keeping the standard vision encoder.

    Args:
        config (LongCLIPConfig): Configuration for the complete model.
    """

    config_class = LongCLIPConfig

    def __init__(self, config: LongCLIPConfig):
        super().__init__(config)

        # Replace text model with LongCLIP version
        if not isinstance(config.text_config, LongCLIPTextConfig):
            text_config = LongCLIPTextConfig(**config.text_config)
        else:
            text_config = config.text_config

        self.text_model = LongCLIPTextModel(text_config)

        # Vision model stays the same (standard CLIP)
        if not isinstance(config.vision_config, LongCLIPVisionConfig):
            vision_config = LongCLIPVisionConfig(**config.vision_config)
        else:
            vision_config = config.vision_config

        self.vision_model = LongCLIPVisionModel(vision_config)

        # Initialize weights
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Get text features from the text encoder.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a ModelOutput object.

        Returns:
            Text features of shape [batch_size, projection_dim].
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = (
            text_outputs[1] if not return_dict else text_outputs.pooler_output
        )
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Get image features from the vision encoder.

        Args:
            pixel_values: Pixel values.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a ModelOutput object.

        Returns:
            Image features of shape [batch_size, projection_dim].
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = (
            vision_outputs[1] if not return_dict else vision_outputs.pooler_output
        )
        image_features = self.visual_projection(pooled_output)

        return image_features


# ================== Processor Class ==================


class LongCLIPProcessor(ProcessorMixin):
    """
    Processor for LongCLIP that combines image and text preprocessing.

    This processor wraps CLIPImageProcessor and CLIPTokenizer to provide
    a unified interface for preprocessing inputs for LongCLIP models.

    Args:
        image_processor (CLIPImageProcessor): Image processor for preprocessing images.
        tokenizer (CLIPTokenizer): Tokenizer for preprocessing text.

    Attributes:
        image_processor_class (str): Name of the image processor class.
        tokenizer_class (str): Name of the tokenizer class.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "CLIPTokenizer"

    def __init__(
        self,
        image_processor: Optional[CLIPImageProcessor] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[str, List[str], None] = None,
        images=None,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = True,
        max_length: Optional[int] = 248,
        truncation: Optional[bool] = True,
        **kwargs,
    ):
        """
        Preprocess text and images for LongCLIP model.

        Args:
            text (str, List[str], optional): Text or list of texts to process.
            images: Image or list of images to process. Can be PIL Image, numpy array, or tensor.
            return_tensors (str, optional): Type of tensors to return ('pt' for PyTorch).
            padding (bool or str, optional): Padding strategy. Defaults to True.
            max_length (int, optional): Maximum sequence length. Defaults to 248 for LongCLIP.
            truncation (bool, optional): Whether to truncate sequences. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            BatchEncoding: Dictionary containing processed inputs with keys:
                - input_ids: Tokenized text (if text provided)
                - attention_mask: Attention mask for text (if text provided)
                - pixel_values: Processed images (if images provided)
        """
        # Process text
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                **kwargs,
            )
        else:
            text_inputs = {}

        # Process images
        if images is not None:
            image_inputs = self.image_processor(
                images,
                return_tensors=return_tensors,
            )
        else:
            image_inputs = {}

        # Combine inputs
        return {**text_inputs, **image_inputs}

    def batch_decode(self, *args, **kwargs):
        """
        Decode token IDs back to text.

        This method is forwarded to the tokenizer's batch_decode method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Decode token IDs back to text.

        This method is forwarded to the tokenizer's decode method.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Get the names of model inputs.

        Returns:
            List[str]: List of input names.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# Register configuration for auto classes
from transformers import AutoConfig, AutoModel

AutoConfig.register("longclip", LongCLIPConfig)
AutoModel.register(LongCLIPConfig, LongCLIPModel)
