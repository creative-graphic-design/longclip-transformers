---
language: en
license: mit
tags:
  - clip
  - vision-language
  - image-text
  - zero-shot
  - retrieval
pipeline_tag: zero-shot-image-classification
---

# LongCLIP: Unlocking the Long-Text Capability of CLIP

[![Paper](https://img.shields.io/badge/arXiv-2403.15378-b31b1b)](https://arxiv.org/abs/2403.15378)
[![Conference](https://img.shields.io/badge/ECCV-2024-blue)](https://eccv2024.ecva.net/)
[![GitHub](https://img.shields.io/badge/GitHub-creative-graphic-design/longclip--transformers-black)](https://github.com/creative-graphic-design/longclip-transformers)

## Model Description

LongCLIP is an enhanced version of OpenAI's CLIP that extends the maximum input text length from **77 to 248 tokens**, enabling better understanding of detailed, long-form text descriptions. This model maintains CLIP's zero-shot capabilities while significantly improving performance on long-caption retrieval tasks.

### Key Features

- 🔥 **Extended Context Length**: 248 tokens (3.2× longer than original CLIP)
- 🔥 **Strong Performance**: +20% R@5 on long-caption retrieval, +6% on standard retrieval
- 🔥 **Plug-and-Play**: Drop-in replacement for CLIP in existing workflows
- 🔥 **Two Model Sizes**: Base (LongCLIP-B) and Large (LongCLIP-L)

### Model Variants

| Model          | Text Encoder    | Vision Encoder   | Params | Projection Dim |
| -------------- | --------------- | ---------------- | ------ | -------------- |
| **LongCLIP-B** | 12 layers, 512d | 12 layers, 768d  | ~150M  | 512            |
| **LongCLIP-L** | 12 layers, 768d | 24 layers, 1024d | ~430M  | 768            |

## Uses

### Direct Use

LongCLIP can be used for:

- **Zero-shot image classification** with detailed text descriptions
- **Image-text retrieval** with long, descriptive captions
- **Text-to-image generation** (e.g., Stable Diffusion XL integration)
- **Visual question answering** with complex queries

### Downstream Use

LongCLIP serves as a backbone for:

- Vision-language models requiring long text understanding
- Multimodal retrieval systems
- Content-based image search engines
- Automated image captioning evaluation

## How to Use

### Installation

```bash
pip install "transformers[torch,torch-vision]"
```

### Quick Start

```python
from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = AutoModel.from_pretrained(
    "creative-graphic-design/LongCLIP-B",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "creative-graphic-design/LongCLIP-B",
    trust_remote_code=True
)

# Prepare inputs
image = Image.open("your_image.jpg")
texts = [
    "A man is crossing the street with a red car parked nearby.",
    "A man is driving a car in an urban scene."
]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    max_length=248,
    padding="max_length"
)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1)

print("Probabilities:", probs)
```

### Advanced Usage: Feature Extraction

```python
# Extract features separately (unnormalized)
text_inputs = processor(text=texts, return_tensors="pt", max_length=248, padding="max_length")
image_inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    image_features = model.get_image_features(**image_inputs)

    # Compute similarity (like original CLIP)
    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1)
```

### Comparison with Original CLIP

```python
# Original CLIP: max 77 tokens
clip_text = "A cat"

# LongCLIP: up to 248 tokens
longclip_text = "A fluffy orange tabby cat with green eyes is sitting on a wooden table near a window, with sunlight streaming through the curtains in the background, creating a warm and cozy atmosphere in a modern living room."

# LongCLIP can handle both short and long texts effectively!
```

## Citation

If you use LongCLIP in your research, please cite:

```bibtex
@inproceedings{zhang2024longclip,
  title={Long-CLIP: Unlocking the Long-Text Capability of CLIP},
  author={Zhang, Beichen and Zhang, Pan and Dong, Xiaoyi and Zang, Yuhang and Wang, Jiaqi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## License

This model is released under the MIT License, consistent with the original CLIP model.

## Acknowledgments

- **OpenAI CLIP**: Foundation model and architecture
- **Original Authors**: Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Jiaqi Wang

## Model Card Contact

For questions and feedback, please open an issue on the [GitHub repository](https://github.com/creative-graphic-design/longclip-transformers).
