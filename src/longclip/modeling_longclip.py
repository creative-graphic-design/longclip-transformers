"""
LongCLIP model implementation compatible with HuggingFace Transformers.

This module provides transformers-compatible implementations of LongCLIP models.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer,
)

from .configuration_longclip import (
    LongCLIPConfig,
    LongCLIPTextConfig,
    LongCLIPVisionConfig,
)


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

    Example:
        ```python
        >>> from long_clip_hf import LongCLIPTextConfig, LongCLIPTextModel
        >>> from transformers import CLIPTokenizer
        >>>
        >>> # Initialize model
        >>> config = LongCLIPTextConfig()
        >>> model = LongCLIPTextModel(config)
        >>>
        >>> # Tokenize text
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> inputs = tokenizer(
        ...     ["a photo of a cat"],
        ...     return_tensors="pt",
        ...     padding="max_length",
        ...     max_length=248,
        ...     truncation=True,
        ... )
        >>>
        >>> # Get text features
        >>> outputs = model(**inputs)
        >>> text_features = outputs.pooler_output
        ```
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

    Example:
        ```python
        >>> from long_clip_hf import LongCLIPVisionConfig, LongCLIPVisionModel
        >>> from transformers import CLIPImageProcessor
        >>> from PIL import Image
        >>>
        >>> # Initialize model
        >>> config = LongCLIPVisionConfig()
        >>> model = LongCLIPVisionModel(config)
        >>>
        >>> # Process image
        >>> processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> image = Image.open("path/to/image.jpg")
        >>> inputs = processor(images=image, return_tensors="pt")
        >>>
        >>> # Get image features
        >>> outputs = model(**inputs)
        >>> image_features = outputs.pooler_output
        ```
    """

    config_class = LongCLIPVisionConfig


class LongCLIPModel(CLIPModel):
    """
    LongCLIP model combining text and vision encoders.

    This model extends CLIPModel to use LongCLIPTextModel with 248 token
    context length while keeping the standard vision encoder.

    Args:
        config (LongCLIPConfig): Configuration for the complete model.

    Example:
        ```python
        >>> from long_clip_hf import LongCLIPConfig, LongCLIPModel
        >>> from transformers import CLIPTokenizer, CLIPImageProcessor
        >>> from PIL import Image
        >>>
        >>> # Initialize model
        >>> config = LongCLIPConfig()
        >>> model = LongCLIPModel(config)
        >>>
        >>> # Prepare inputs
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>>
        >>> text = "a photo of a cat"
        >>> image = Image.open("path/to/image.jpg")
        >>>
        >>> text_inputs = tokenizer(
        ...     [text],
        ...     return_tensors="pt",
        ...     padding="max_length",
        ...     max_length=248,
        ...     truncation=True,
        ... )
        >>> image_inputs = processor(images=image, return_tensors="pt")
        >>>
        >>> # Get features
        >>> outputs = model(
        ...     input_ids=text_inputs["input_ids"],
        ...     pixel_values=image_inputs["pixel_values"],
        ... )
        >>>
        >>> # Compute similarity
        >>> logits_per_image = outputs.logits_per_image
        >>> probs = logits_per_image.softmax(dim=1)
        ```
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
