# Long-CLIP

This repository is the official implementation of Long-CLIP

**Long-CLIP: Unlocking the Long-Text Capability of CLIP**\
[Beichen Zhang](https://beichenzbc.github.io), [Pan Zhang](https://panzhang0212.github.io/), [Xiaoyi Dong](https://lightdxy.github.io/), [Yuhang Zang](https://yuhangzang.github.io/), [Jiaqi Wang](https://myownskyw7.github.io/)

## 💡 Highlights

- 🔥 **Long Input length** Increase the maximum input length of CLIP from **77** to **248**.
- 🔥 **Strong Performace** Improve the R@5 of long-caption text-image retrieval by **20%** and traditional text-image retrieval by **6%**.
- 🔥 **Plug-in and play** Can be directly applied in **any work** that requires long-text capability.
- ✨ **Transformers Compatible** Seamlessly integrated with HuggingFace Transformers ecosystem.
- 🚀 **Easy to Use** Load models directly from Hugging Face Hub with one line of code.

## 📜 News

🚀 [2026/1/4] Repository restructured with Transformers-compatible implementation! Now supports easy installation via pip and loading from Hugging Face Hub.

🚀 [2024/7/3] Our paper has been accepted by **_ECCV2024_**.

🚀 [2024/7/3] We release the code of using Long-CLIP in **_SDXL_**. For detailed information, you may refer to `SDXL/SDXL.md`.

🚀 [2024/5/21] We update the paper and checkpoints after fixing the bug in DDP and add results in Urban-1k. Special thanks to @MajorDavidZhang for finding and refining this bug in DDP! Now the fine-tuning only takes **_0.5_** hours on _8 GPUs_!

🚀 [2024/5/21] Urban-1k: a scaling-up version of Urban-200 dataset in the paper has been released at this [page](https://huggingface.co/datasets/BeichenZhang/Urban1k).

🚀 [2024/4/1] The training code is released!

🚀 [2024/3/25] The Inference code and models ([LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) and [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)) are released!

🚀 [2024/3/25] The [paper](https://arxiv.org/abs/2403.15378) is released!

## 👨‍💻 Todo

- [x] Training code for Long-CLIP based on OpenAI-CLIP
- [x] Evaluation code for Long-CLIP
- [x] evaluation code for zero-shot classification and text-image retrieval tasks.
- [x] Usage example of Long-CLIP
- [x] Checkpoints of Long-CLIP
- [x] Transformers-compatible implementation
- [x] Hugging Face Hub integration

## 📁 Repository Structure

```
Long-CLIP/
├── src/longclip/              # Transformers-compatible implementation (main package)
│   ├── configuration_longclip.py
│   ├── modeling_longclip.py
│   └── processing_longclip.py
├── longclip_original/         # Original CLIP-style implementation
│   ├── model/                 # Core model code
│   └── open_clip_long/        # OpenCLIP-based implementation
├── scripts/                   # Utility scripts
│   ├── convert_longclip_to_hf.py  # Convert .pt to Transformers format
│   └── push_to_hub.py         # Upload models to Hugging Face Hub
├── tests/                     # Test suite
├── train/                     # Training scripts
├── eval/                      # Evaluation scripts
├── SDXL/                      # SDXL integration
└── checkpoints/               # Model checkpoints (.pt files)
```

## 🛠️ Usage

### Installation

#### Option 1: Using Transformers (Recommended)

Install via pip with transformers support:

```bash
pip install git+https://github.com/creative-graphic-design/longclip-transformers
```

Or using uv:

```bash
uv pip install git+https://github.com/creative-graphic-design/longclip-transformers
```

#### Option 2: Development Installation

Clone the repository and install:

```bash
git clone https://github.com/creative-graphic-design/longclip-transformers
cd longclip-transformers
uv sync  # or: pip install -e .
```

To include the original implementation for comparison:

```bash
uv sync --group original
```

### How to Use

#### Using Transformers (Recommended)

Load pre-converted models from Hugging Face Hub:

```python
from longclip import LongCLIPModel, LongCLIPProcessor
from PIL import Image
import torch

# Load model and processor from Hub
model = LongCLIPModel.from_pretrained("BeichenZhang/LongCLIP-B")
processor = LongCLIPProcessor.from_pretrained("BeichenZhang/LongCLIP-B")

# Prepare inputs
image = Image.open("./img/demo.png")
texts = [
    "A man is crossing the street with a red car parked nearby.",
    "A man is driving a car in an urban scene."
]

# Process and get predictions
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

print("Label probs:", probs)
```

**Long text support (up to 248 tokens):**

```python
long_text = "A very detailed description of a complex scene with many objects, people, and activities happening simultaneously in an urban environment with buildings, cars, and natural elements." * 3

inputs = processor(text=long_text, images=image, return_tensors="pt")
outputs = model(**inputs)
```

#### Using Original Implementation

If you prefer the original CLIP-style API, download the checkpoints from [LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) or [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L) and place them under `./checkpoints`:

```python
from longclip_original.model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

text = longclip.tokenize([
    "A man is crossing the street with a red car parked nearby.",
    "A man is driving a car in an urban scene."
]).to(device)
image = preprocess(Image.open("./img/demo.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```

#### Converting Checkpoints to Transformers Format

If you have `.pt` checkpoints and want to convert them to Transformers format:

```bash
python scripts/convert_longclip_to_hf.py \
    --checkpoint_path checkpoints/longclip-B.pt \
    --output_path ./longclip-base-hf

# Then use with transformers
python -c "from longclip import LongCLIPModel; model = LongCLIPModel.from_pretrained('./longclip-base-hf')"
```

See `scripts/README.md` for more details on conversion and uploading to Hugging Face Hub.

#### Comparison: Transformers vs Original

| Feature          | Transformers (Recommended)        | Original Implementation           |
| ---------------- | --------------------------------- | --------------------------------- |
| **API Style**    | HuggingFace standard              | CLIP-style                        |
| **Loading**      | `from_pretrained()` from Hub      | Load from local `.pt` file        |
| **Processor**    | Unified `LongCLIPProcessor`       | Separate tokenizer & preprocessor |
| **Integration**  | Works with transformers ecosystem | Standalone                        |
| **Model Format** | SafeTensors/PyTorch               | PyTorch only                      |
| **Installation** | `pip install`                     | Requires manual setup             |
| **Use Case**     | Production, Easy deployment       | Research, Legacy compatibility    |

### Evaluation

#### Zero-shot classification

To run zero-shot classification on imagenet dataset, run the following command after preparing the data

```shell
cd eval/classification/imagenet
python imagenet.py
```

Similarly, run the following command for cifar datset

```shell
cd eval/classification/cifar
python cifar10.py               #cifar10
python cifar100.py              #cifar100
```

#### Retrieval

To run text-image retrieval on COCO2017 or Flickr30k, run the following command after preparing the data

```shell
cd eval/retrieval
python coco.py                  #COCO2017
python flickr30k.py             #Flickr30k
```

### Traning

Please refer to `train/train.md` for training details.

## ⭐ Demos

### Long-CLIP-SDXL

<p align="center"> <a>  
<img src="./img/demo_SDXL.png"  width="900" />
</a> </p>

### Long-caption text-image retrieval

<p align="center"> <a>  
<img src="./img/retrieval.png"  width="900" />
</a> </p>

### Plug-and-Play text to image generation

<p align="center"> <a>  
<img src="./img/generation.png"  width="900" />
</a> </p>

## Citation

If you find our work helpful for your research, please consider giving a citation:

```
@article{zhang2024longclip,
        title={Long-CLIP: Unlocking the Long-Text Capability of CLIP},
        author={Beichen Zhang and Pan Zhang and Xiaoyi Dong and Yuhang Zang and Jiaqi Wang},
        journal={arXiv preprint arXiv:2403.15378},
        year={2024}
}
```
