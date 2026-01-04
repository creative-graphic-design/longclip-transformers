#!/usr/bin/env python3
"""
Push LongCLIP model to Hugging Face Hub.

This script uploads a converted LongCLIP model to the Hugging Face Hub,
including the model, configuration, processor, and model card.

Usage:
    python scripts/push_to_hub.py \
        --model_path ./longclip-base-hf \
        --repo_id your-username/longclip-base \
        --token YOUR_HF_TOKEN \
        --private

    Or set HF_TOKEN environment variable:
    export HF_TOKEN=YOUR_HF_TOKEN
    python scripts/push_to_hub.py \
        --model_path ./longclip-base-hf \
        --repo_id your-username/longclip-base
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo
from longclip import LongCLIPModel, LongCLIPProcessor


def create_model_card(
    repo_id: str,
    model_name: str,
    model_size: str,
    base_model: str = "openai/clip-vit-base-patch32",
) -> str:
    """Create a model card (README.md) for the Hub."""

    size_info = {
        "B": {"params": "149M", "layers": 12, "hidden": 512},
        "L": {"params": "354M", "layers": 24, "hidden": 768},
    }

    info = size_info.get(
        model_size.upper(),
        {"params": "Unknown", "layers": "Unknown", "hidden": "Unknown"},
    )

    model_card = f"""---
language: en
license: apache-2.0
tags:
  - clip
  - vision
  - image-text
  - multimodal
  - zero-shot-classification
  - image-classification
  - longclip
library_name: transformers
pipeline_tag: zero-shot-image-classification
---

# {model_name}

## Model Description

LongCLIP is an extension of CLIP that supports longer text inputs (up to 248 tokens) compared to the original CLIP's 77 token limit. This enables more detailed and accurate image-text matching for complex queries.

This model is the {model_size.upper()} variant with:
- **Parameters**: {info["params"]}
- **Text Encoder Layers**: {info["layers"]}
- **Hidden Size**: {info["hidden"]}
- **Max Text Length**: 248 tokens
- **Image Resolution**: 224x224
- **Base Model**: {base_model}

## Usage

### Using Transformers

```python
from transformers import CLIPTokenizer
from longclip import LongCLIPModel, LongCLIPProcessor
from PIL import Image
import torch

# Load model and processor
model = LongCLIPModel.from_pretrained("{repo_id}")
processor = LongCLIPProcessor.from_pretrained("{repo_id}")

# Prepare inputs
image = Image.open("your_image.jpg")
texts = [
    "A photo of a cat sitting on a windowsill watching birds outside",
    "A dog playing in a park with children on a sunny day",
]

# Process inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# Get image-text similarity
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

print("Probabilities:", probs)
```

### Long Text Support

LongCLIP can handle texts up to 248 tokens:

```python
long_text = "A detailed description of a complex scene with many elements, " * 20
inputs = processor(text=long_text, images=image, return_tensors="pt")
outputs = model(**inputs)
```

## Model Architecture

LongCLIP extends CLIP with:
- **Dual Positional Embeddings**: Combines original and residual positional embeddings for better long-text understanding
- **Extended Context Length**: Supports up to 248 tokens (vs 77 in original CLIP)
- **Compatible API**: Drop-in replacement for CLIP with transformers library

## Training Data

This model was trained on the same data as the original CLIP model, with additional fine-tuning for long text understanding.

## Limitations and Bias

- Inherits biases from the original CLIP training data
- Performance may vary on domain-specific images
- Long text encoding may require more compute resources

## Citation

```bibtex
@article{{longclip2024,
  title={{LongCLIP: Unlocking the Long-Text Capability of CLIP}},
  author={{Zhang, Beichen and others}},
  journal={{arXiv preprint}},
  year={{2024}}
}}

@inproceedings{{radford2021learning,
  title={{Learning Transferable Visual Models From Natural Language Supervision}},
  author={{Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others}},
  booktitle={{International Conference on Machine Learning}},
  pages={{8748--8763}},
  year={{2021}},
  organization={{PMLR}}
}}
```

## License

Apache 2.0

## Acknowledgements

This implementation is based on the original LongCLIP paper and OpenAI's CLIP model.
"""

    return model_card


def push_to_hub(
    model_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    model_size: str = "B",
    commit_message: Optional[str] = None,
) -> str:
    """
    Push LongCLIP model to Hugging Face Hub.

    Args:
        model_path: Path to local model directory
        repo_id: Repository ID on Hub (e.g., "username/longclip-base")
        token: Hugging Face API token (or set HF_TOKEN env var)
        private: Whether to create a private repository
        model_size: Model size variant ("B" or "L")
        commit_message: Custom commit message

    Returns:
        URL of the created repository
    """
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "No token provided. Set HF_TOKEN environment variable or pass --token"
            )

    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    print(f"Loading model from {model_path}")

    # Load model and processor to verify they work
    try:
        model = LongCLIPModel.from_pretrained(model_path)
        processor = LongCLIPProcessor.from_pretrained(model_path)
        print("✓ Model and processor loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Create repository
    print(f"\nCreating repository: {repo_id}")
    api = HfApi()

    try:
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"✓ Repository created: {repo_url}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
        repo_url = f"https://huggingface.co/{repo_id}"

    # Create model card
    print("\nCreating model card...")
    model_name = repo_id.split("/")[-1]
    model_card = create_model_card(
        repo_id=repo_id,
        model_name=model_name,
        model_size=model_size,
    )

    readme_path = model_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"✓ Model card saved to {readme_path}")

    # Push model to hub
    print("\nPushing model to Hub...")
    commit_msg = commit_message or f"Upload {model_name}"

    try:
        model.push_to_hub(
            repo_id=repo_id,
            token=token,
            commit_message=commit_msg,
            private=private,
        )
        print("✓ Model pushed successfully")

        # Push processor separately to ensure it's included
        print("\nPushing processor to Hub...")
        processor.push_to_hub(
            repo_id=repo_id,
            token=token,
            commit_message=commit_msg,
            private=private,
        )
        print("✓ Processor pushed successfully")

    except Exception as e:
        print(f"Error during push: {e}")
        print("\nTrying alternative upload method...")

        # Upload entire directory as fallback
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_msg,
            repo_type="model",
        )
        print("✓ Files uploaded via folder upload")

    print(f"\n{'=' * 60}")
    print("✓ Model successfully pushed to Hub!")
    print(f"{'=' * 60}")
    print(f"\nRepository: {repo_url}")
    print("\nYou can now load the model with:")
    print("  from longclip import LongCLIPModel, LongCLIPProcessor")
    print(f"  model = LongCLIPModel.from_pretrained('{repo_id}')")
    print(f"  processor = LongCLIPProcessor.from_pretrained('{repo_id}')")

    return repo_url


def main():
    parser = argparse.ArgumentParser(
        description="Push LongCLIP model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to local model directory (e.g., ./longclip-base-hf)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID on Hub (e.g., username/longclip-base)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="B",
        choices=["B", "L"],
        help="Model size variant (B or L)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message",
    )

    args = parser.parse_args()

    push_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        model_size=args.model_size,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
