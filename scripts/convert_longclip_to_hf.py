#!/usr/bin/env python3
"""
Convert LongCLIP checkpoints to HuggingFace Transformers format.

This script converts the original LongCLIP checkpoint format (.pt files) to
the HuggingFace Transformers format for use with the LongCLIPModel class.

Usage:
    python scripts/convert_longclip_to_hf.py \
        --checkpoint_path checkpoints/longclip-B.pt \
        --output_path ./longclip-base-hf \
        --model_size B
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from long_clip_hf import (
    LongCLIPConfig,
    LongCLIPModel,
    LongCLIPTextConfig,
    LongCLIPVisionConfig,
)


def copy_linear(hf_linear: nn.Module, pt_weight: torch.Tensor, pt_bias: torch.Tensor):
    """Copy weights and biases to a linear layer."""
    hf_linear.weight.data = pt_weight
    hf_linear.bias.data = pt_bias


def copy_attn_layer(hf_attn_layer: nn.Module, state_dict: Dict, prefix: str):
    """
    Copy attention layer weights from original format to HF format.

    Original format uses in_proj_weight/bias (combined Q,K,V) which needs to be
    split into separate q_proj, k_proj, v_proj.
    """
    in_proj_weight = state_dict[f"{prefix}.in_proj_weight"]
    in_proj_bias = state_dict[f"{prefix}.in_proj_bias"]

    # Split combined QKV weights into separate Q, K, V
    q_proj, k_proj, v_proj = in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3, dim=0)

    # Copy Q, K, V projections
    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias

    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias

    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    # Copy output projection
    hf_attn_layer.out_proj.weight.data = state_dict[f"{prefix}.out_proj.weight"]
    hf_attn_layer.out_proj.bias.data = state_dict[f"{prefix}.out_proj.bias"]


def copy_mlp(hf_mlp: nn.Module, state_dict: Dict, prefix: str):
    """Copy MLP layer weights."""
    # fc1: c_fc in original
    hf_mlp.fc1.weight.data = state_dict[f"{prefix}.c_fc.weight"]
    hf_mlp.fc1.bias.data = state_dict[f"{prefix}.c_fc.bias"]

    # fc2: c_proj in original
    hf_mlp.fc2.weight.data = state_dict[f"{prefix}.c_proj.weight"]
    hf_mlp.fc2.bias.data = state_dict[f"{prefix}.c_proj.bias"]


def copy_transformer_layer(hf_layer: nn.Module, state_dict: Dict, prefix: str):
    """Copy a complete transformer layer (attention + MLP + layer norms)."""
    # Copy layer norms
    copy_linear(
        hf_layer.layer_norm1,
        state_dict[f"{prefix}.ln_1.weight"],
        state_dict[f"{prefix}.ln_1.bias"],
    )
    copy_linear(
        hf_layer.layer_norm2,
        state_dict[f"{prefix}.ln_2.weight"],
        state_dict[f"{prefix}.ln_2.bias"],
    )

    # Copy attention
    copy_attn_layer(hf_layer.self_attn, state_dict, f"{prefix}.attn")

    # Copy MLP
    copy_mlp(hf_layer.mlp, state_dict, f"{prefix}.mlp")


def copy_text_model(hf_model: LongCLIPModel, state_dict: Dict):
    """
    Copy text model weights with special handling for LongCLIP's dual positional embeddings.
    """
    text_model = hf_model.text_model.text_model

    # Copy token embeddings
    text_model.embeddings.token_embedding.weight.data = state_dict[
        "token_embedding.weight"
    ]

    # Copy dual positional embeddings (LongCLIP specific)
    text_model.embeddings.position_embedding.weight.data = state_dict[
        "positional_embedding"
    ]
    text_model.embeddings.position_embedding_res.data = state_dict[
        "positional_embedding_res"
    ]

    # Copy final layer norm
    copy_linear(
        text_model.final_layer_norm,
        state_dict["ln_final.weight"],
        state_dict["ln_final.bias"],
    )

    # Copy transformer layers
    num_layers = len(text_model.encoder.layers)
    for i in range(num_layers):
        copy_transformer_layer(
            text_model.encoder.layers[i], state_dict, f"transformer.resblocks.{i}"
        )

    # Copy text projection (note: transpose needed)
    hf_model.text_projection.weight.data = state_dict["text_projection"].T.contiguous()


def copy_vision_model(hf_model: LongCLIPModel, state_dict: Dict):
    """Copy vision model weights."""
    vision_model = hf_model.vision_model.vision_model

    # Copy embeddings
    vision_model.embeddings.patch_embedding.weight.data = state_dict[
        "visual.conv1.weight"
    ]
    vision_model.embeddings.class_embedding.data = state_dict["visual.class_embedding"]
    vision_model.embeddings.position_embedding.weight.data = state_dict[
        "visual.positional_embedding"
    ]

    # Copy pre/post layer norms
    copy_linear(
        vision_model.pre_layrnorm,
        state_dict["visual.ln_pre.weight"],
        state_dict["visual.ln_pre.bias"],
    )
    copy_linear(
        vision_model.post_layernorm,
        state_dict["visual.ln_post.weight"],
        state_dict["visual.ln_post.bias"],
    )

    # Copy transformer layers
    num_layers = len(vision_model.encoder.layers)
    for i in range(num_layers):
        copy_transformer_layer(
            vision_model.encoder.layers[i],
            state_dict,
            f"visual.transformer.resblocks.{i}",
        )

    # Copy visual projection (note: transpose needed)
    hf_model.visual_projection.weight.data = state_dict["visual.proj"].T.contiguous()


def determine_config_from_checkpoint(state_dict: Dict) -> LongCLIPConfig:
    """
    Automatically determine model configuration from checkpoint state dict.
    """
    # Text model dimensions
    text_hidden_size = state_dict["ln_final.weight"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    max_position_embeddings = state_dict["positional_embedding"].shape[0]

    # Count text transformer layers (count unique layer indices)
    text_layers = set()
    for k in state_dict.keys():
        if k.startswith("transformer.resblocks."):
            parts = k.split(".")
            if len(parts) >= 3:
                layer_num = int(parts[2])
                text_layers.add(layer_num)
    text_num_layers = len(text_layers)

    # Text attention dimensions
    text_in_proj_weight = state_dict["transformer.resblocks.0.attn.in_proj_weight"]
    text_num_heads = text_hidden_size // 64  # Standard head_dim=64
    text_intermediate_size = state_dict[
        "transformer.resblocks.0.mlp.c_fc.weight"
    ].shape[0]

    # Vision model dimensions
    vision_hidden_size = state_dict["visual.ln_post.weight"].shape[0]
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    vision_image_size = 224  # Standard CLIP image size

    # Count vision transformer layers (count unique layer indices)
    vision_layers = set()
    for k in state_dict.keys():
        if k.startswith("visual.transformer.resblocks."):
            parts = k.split(".")
            if len(parts) >= 4:
                layer_num = int(parts[3])
                vision_layers.add(layer_num)
    vision_num_layers = len(vision_layers)

    # Vision attention dimensions
    vision_num_heads = vision_hidden_size // 64
    vision_intermediate_size = state_dict[
        "visual.transformer.resblocks.0.mlp.c_fc.weight"
    ].shape[0]

    # Projection dimension
    projection_dim = state_dict["text_projection"].shape[0]

    # Create configs
    text_config = LongCLIPTextConfig(
        hidden_size=text_hidden_size,
        intermediate_size=text_intermediate_size,
        num_hidden_layers=text_num_layers,
        num_attention_heads=text_num_heads,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
    )

    vision_config = LongCLIPVisionConfig(
        hidden_size=vision_hidden_size,
        intermediate_size=vision_intermediate_size,
        num_hidden_layers=vision_num_layers,
        num_attention_heads=vision_num_heads,
        image_size=vision_image_size,
        patch_size=vision_patch_size,
    )

    config = LongCLIPConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=projection_dim,
    )

    return config


@torch.no_grad()
def convert_longclip_checkpoint(
    checkpoint_path: str,
    output_path: str,
    validate: bool = True,
):
    """
    Convert LongCLIP checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to original LongCLIP checkpoint (.pt file)
        output_path: Path to save HF model
        validate: Whether to validate conversion with test inputs
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    print("Determining model configuration from checkpoint")
    config = determine_config_from_checkpoint(state_dict)

    print("Creating LongCLIP model:")
    text_config_dict = (
        config.text_config
        if isinstance(config.text_config, dict)
        else config.text_config.to_dict()
    )
    vision_config_dict = (
        config.vision_config
        if isinstance(config.vision_config, dict)
        else config.vision_config.to_dict()
    )
    print(
        f"  Text: {text_config_dict['num_hidden_layers']} layers, "
        f"{text_config_dict['hidden_size']} hidden, "
        f"{text_config_dict['max_position_embeddings']} max positions"
    )
    print(
        f"  Vision: {vision_config_dict['num_hidden_layers']} layers, "
        f"{vision_config_dict['hidden_size']} hidden"
    )
    print(f"  Projection: {config.projection_dim}")

    # Create HF model
    hf_model = LongCLIPModel(config)
    hf_model.eval()

    print("Copying weights...")
    # Copy text model (includes dual positional embeddings)
    copy_text_model(hf_model, state_dict)

    # Copy vision model
    copy_vision_model(hf_model, state_dict)

    # Copy logit scale
    hf_model.logit_scale.data = state_dict["logit_scale"]

    if validate:
        print("Validating conversion...")
        # Create test inputs with 248 token context
        text_config_dict = (
            config.text_config
            if isinstance(config.text_config, dict)
            else config.text_config.to_dict()
        )
        input_ids = torch.tensor(
            [
                [text_config_dict["bos_token_id"]]
                + list(range(3, 248))
                + [text_config_dict["eos_token_id"]]
            ]
        )
        pixel_values = torch.randn(1, 3, 224, 224)

        # Run forward pass
        outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values)

        print(f"  Text features shape: {outputs.text_embeds.shape}")
        print(f"  Image features shape: {outputs.image_embeds.shape}")
        print(f"  Logits per image shape: {outputs.logits_per_image.shape}")
        print(f"  Logits per text shape: {outputs.logits_per_text.shape}")

        # Check for NaN or Inf
        assert not torch.isnan(outputs.logits_per_image).any(), "Found NaN in outputs"
        assert not torch.isinf(outputs.logits_per_image).any(), "Found Inf in outputs"
        print("  Validation passed!")

    # Save model
    print(f"Saving model to {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    print("Conversion complete!")
    print("\nTo load the model:")
    print("  from long_clip_hf import LongCLIPModel")
    print(f"  model = LongCLIPModel.from_pretrained('{output_path}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LongCLIP checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to original LongCLIP checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save converted HuggingFace model",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip validation step"
    )

    args = parser.parse_args()

    convert_longclip_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
