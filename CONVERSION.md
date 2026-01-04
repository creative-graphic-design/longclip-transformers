# LongCLIP Checkpoint Conversion Guide

このドキュメントでは、オリジナルの LongCLIP checkpoint (.pt 形式) から HuggingFace Transformers 形式への変換プロセスを説明します。

## 概要

LongCLIP のオリジナル実装は独自の checkpoint 形式を使用していますが、これを HuggingFace Transformers と互換性のある形式に変換することで、`AutoModel.from_pretrained()`などの標準 API で利用できるようになります。

## オリジナルの Checkpoint 構造

### State Dict Keys

オリジナルの LongCLIP checkpoint (`longclip-B.pt`, `longclip-L.pt`) には以下のキーが含まれています：

#### Text Encoder

```python
# Token embeddings
"token_embedding.weight"                              # [vocab_size, text_hidden_size]

# Dual positional embeddings (LongCLIP特有)
"positional_embedding"                                 # [248, text_hidden_size] - Base embeddings
"positional_embedding_res"                             # [248, text_hidden_size] - Residual embeddings

# Transformer layers (N layers)
"transformer.resblocks.{i}.ln_1.weight"               # Layer norm 1
"transformer.resblocks.{i}.ln_1.bias"
"transformer.resblocks.{i}.attn.in_proj_weight"       # Combined Q,K,V projection
"transformer.resblocks.{i}.attn.in_proj_bias"
"transformer.resblocks.{i}.attn.out_proj.weight"      # Output projection
"transformer.resblocks.{i}.attn.out_proj.bias"
"transformer.resblocks.{i}.ln_2.weight"               # Layer norm 2
"transformer.resblocks.{i}.ln_2.bias"
"transformer.resblocks.{i}.mlp.c_fc.weight"           # MLP first layer
"transformer.resblocks.{i}.mlp.c_fc.bias"
"transformer.resblocks.{i}.mlp.c_proj.weight"         # MLP second layer
"transformer.resblocks.{i}.mlp.c_proj.bias"

# Final layer norm
"ln_final.weight"
"ln_final.bias"

# Text projection
"text_projection"                                      # [text_hidden_size, projection_dim]
```

#### Vision Encoder

```python
# Patch embeddings
"visual.conv1.weight"                                  # [vision_hidden_size, 3, patch_size, patch_size]

# Position embeddings
"visual.class_embedding"                               # [vision_hidden_size]
"visual.positional_embedding"                          # [num_patches+1, vision_hidden_size]

# Pre-layer norm
"visual.ln_pre.weight"
"visual.ln_pre.bias"

# Transformer layers (N layers)
"visual.transformer.resblocks.{i}.ln_1.weight"
"visual.transformer.resblocks.{i}.ln_1.bias"
"visual.transformer.resblocks.{i}.attn.in_proj_weight"
"visual.transformer.resblocks.{i}.attn.in_proj_bias"
"visual.transformer.resblocks.{i}.attn.out_proj.weight"
"visual.transformer.resblocks.{i}.attn.out_proj.bias"
"visual.transformer.resblocks.{i}.ln_2.weight"
"visual.transformer.resblocks.{i}.ln_2.bias"
"visual.transformer.resblocks.{i}.mlp.c_fc.weight"
"visual.transformer.resblocks.{i}.mlp.c_fc.bias"
"visual.transformer.resblocks.{i}.mlp.c_proj.weight"
"visual.transformer.resblocks.{i}.mlp.c_proj.bias"

# Post-layer norm
"visual.ln_post.weight"
"visual.ln_post.bias"

# Visual projection
"visual.proj"                                          # [vision_hidden_size, projection_dim]
```

#### Shared

```python
# Logit scale (temperature parameter)
"logit_scale"                                          # scalar
```

## HuggingFace Transformers 形式の構造

### Model Architecture

```python
LongCLIPModel
├── text_model (LongCLIPTextModel)
│   └── text_model (LongCLIPTextTransformer)
│       ├── embeddings (LongCLIPTextEmbeddings)
│       │   ├── token_embedding
│       │   ├── position_embedding           # Base positional embeddings
│       │   └── position_embedding_res       # Residual positional embeddings
│       ├── encoder
│       │   └── layers[i]
│       │       ├── layer_norm1
│       │       ├── self_attn
│       │       │   ├── q_proj
│       │       │   ├── k_proj
│       │       │   ├── v_proj
│       │       │   └── out_proj
│       │       ├── layer_norm2
│       │       └── mlp
│       │           ├── fc1
│       │           └── fc2
│       └── final_layer_norm
├── text_projection
│
├── vision_model (LongCLIPVisionModel)
│   └── vision_model (CLIPVisionTransformer)
│       ├── embeddings
│       │   ├── patch_embedding
│       │   ├── class_embedding
│       │   └── position_embedding
│       ├── pre_layrnorm
│       ├── encoder
│       │   └── layers[i]
│       │       ├── layer_norm1
│       │       ├── self_attn (Q,K,V,out_proj)
│       │       ├── layer_norm2
│       │       └── mlp (fc1, fc2)
│       └── post_layernorm
├── visual_projection
│
└── logit_scale
```

## キーマッピング

### Text Encoder

| オリジナル                                  | HuggingFace                                                        | 備考                            |
| ------------------------------------------- | ------------------------------------------------------------------ | ------------------------------- |
| `token_embedding.weight`                    | `text_model.text_model.embeddings.token_embedding.weight`          | そのままコピー                  |
| `positional_embedding`                      | `text_model.text_model.embeddings.position_embedding.weight`       | Base embeddings                 |
| `positional_embedding_res`                  | `text_model.text_model.embeddings.position_embedding_res`          | Residual embeddings (Parameter) |
| `transformer.resblocks.{i}.ln_1.*`          | `text_model.text_model.encoder.layers[i].layer_norm1.*`            | Layer norm 1                    |
| `transformer.resblocks.{i}.attn.in_proj_*`  | `text_model.text_model.encoder.layers[i].self_attn.{q,k,v}_proj.*` | **3 つに分割が必要**            |
| `transformer.resblocks.{i}.attn.out_proj.*` | `text_model.text_model.encoder.layers[i].self_attn.out_proj.*`     | Output projection               |
| `transformer.resblocks.{i}.ln_2.*`          | `text_model.text_model.encoder.layers[i].layer_norm2.*`            | Layer norm 2                    |
| `transformer.resblocks.{i}.mlp.c_fc.*`      | `text_model.text_model.encoder.layers[i].mlp.fc1.*`                | MLP first layer                 |
| `transformer.resblocks.{i}.mlp.c_proj.*`    | `text_model.text_model.encoder.layers[i].mlp.fc2.*`                | MLP second layer                |
| `ln_final.*`                                | `text_model.text_model.final_layer_norm.*`                         | Final layer norm                |
| `text_projection`                           | `text_projection.weight`                                           | **転置が必要 (.T)**             |

### Vision Encoder

| オリジナル                                         | HuggingFace                                                            | 備考                 |
| -------------------------------------------------- | ---------------------------------------------------------------------- | -------------------- |
| `visual.conv1.weight`                              | `vision_model.vision_model.embeddings.patch_embedding.weight`          | Patch embedding      |
| `visual.class_embedding`                           | `vision_model.vision_model.embeddings.class_embedding`                 | CLS token            |
| `visual.positional_embedding`                      | `vision_model.vision_model.embeddings.position_embedding.weight`       | Position embeddings  |
| `visual.ln_pre.*`                                  | `vision_model.vision_model.pre_layrnorm.*`                             | Pre-layer norm       |
| `visual.transformer.resblocks.{i}.ln_1.*`          | `vision_model.vision_model.encoder.layers[i].layer_norm1.*`            | Layer norm 1         |
| `visual.transformer.resblocks.{i}.attn.in_proj_*`  | `vision_model.vision_model.encoder.layers[i].self_attn.{q,k,v}_proj.*` | **3 つに分割が必要** |
| `visual.transformer.resblocks.{i}.attn.out_proj.*` | `vision_model.vision_model.encoder.layers[i].self_attn.out_proj.*`     | Output projection    |
| `visual.transformer.resblocks.{i}.ln_2.*`          | `vision_model.vision_model.encoder.layers[i].layer_norm2.*`            | Layer norm 2         |
| `visual.transformer.resblocks.{i}.mlp.c_fc.*`      | `vision_model.vision_model.encoder.layers[i].mlp.fc1.*`                | MLP first layer      |
| `visual.transformer.resblocks.{i}.mlp.c_proj.*`    | `vision_model.vision_model.encoder.layers[i].mlp.fc2.*`                | MLP second layer     |
| `visual.ln_post.*`                                 | `vision_model.vision_model.post_layernorm.*`                           | Post-layer norm      |
| `visual.proj`                                      | `visual_projection.weight`                                             | **転置が必要 (.T)**  |

### Shared

| オリジナル    | HuggingFace   | 備考                  |
| ------------- | ------------- | --------------------- |
| `logit_scale` | `logit_scale` | Temperature parameter |

## 重要な変換処理

### 1. Dual Positional Embeddings (LongCLIP 特有)

LongCLIP の最も重要な特徴は、dual positional embeddings 機構です：

```python
# オリジナルcheckpointから
positional_embedding = state_dict["positional_embedding"]      # [248, hidden_size]
positional_embedding_res = state_dict["positional_embedding_res"]  # [248, hidden_size]

# HFモデルへ
hf_model.text_model.text_model.embeddings.position_embedding.weight.data = positional_embedding
hf_model.text_model.text_model.embeddings.position_embedding_res.data = positional_embedding_res
```

**注意事項:**

- `position_embedding`は通常の`nn.Embedding`
- `position_embedding_res`は`nn.Parameter`（直接テンソル）
- mask1/mask2 を使用して適用される（実装参照）

### 2. Attention Layer: in_proj_weight の分割

オリジナルは Q, K, V を 1 つのテンソルにまとめています：

```python
# オリジナル: [3*hidden_size, hidden_size]
in_proj_weight = state_dict[f"{prefix}.attn.in_proj_weight"]
in_proj_bias = state_dict[f"{prefix}.attn.in_proj_bias"]

# 3つに分割
q_proj, k_proj, v_proj = in_proj_weight.chunk(3, dim=0)
q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3, dim=0)

# HFモデルへコピー
hf_attn.q_proj.weight.data = q_proj
hf_attn.q_proj.bias.data = q_proj_bias
hf_attn.k_proj.weight.data = k_proj
hf_attn.k_proj.bias.data = k_proj_bias
hf_attn.v_proj.weight.data = v_proj
hf_attn.v_proj.bias.data = v_proj_bias
```

### 3. Projection Matrices の転置

Text projection と Visual projection は転置が必要です：

```python
# オリジナル: [hidden_size, projection_dim]
text_projection = state_dict["text_projection"]
visual_projection = state_dict["visual.proj"]

# HFモデル: [projection_dim, hidden_size] (Linear層のweight)
hf_model.text_projection.weight.data = text_projection.T.contiguous()
hf_model.visual_projection.weight.data = visual_projection.T.contiguous()
```

**理由:** オリジナルは `features @ projection` を使用するが、HF の`nn.Linear`は `x @ weight.T` を計算するため。

### 4. Layer Norm の扱い

Layer norm は weight と bias をそのままコピー：

```python
def copy_linear(hf_linear, pt_weight, pt_bias):
    hf_linear.weight.data = pt_weight
    hf_linear.bias.data = pt_bias
```

## アーキテクチャの自動検出

変換スクリプトは checkpoint からモデル構成を自動的に検出します：

### Text Encoder

```python
# Hidden size
text_hidden_size = state_dict["ln_final.weight"].shape[0]  # 512 (B) or 768 (L)

# Number of layers (一意のレイヤーインデックスをカウント)
text_layers = set()
for key in state_dict.keys():
    if key.startswith("transformer.resblocks."):
        layer_num = int(key.split(".")[2])
        text_layers.add(layer_num)
text_num_layers = len(text_layers)  # 12

# Attention heads
text_num_heads = text_hidden_size // 64  # head_dim=64 固定

# Intermediate size (MLP)
text_intermediate_size = state_dict["transformer.resblocks.0.mlp.c_fc.weight"].shape[0]

# Vocab size
vocab_size = state_dict["token_embedding.weight"].shape[0]  # 49408

# Max positions
max_position_embeddings = state_dict["positional_embedding"].shape[0]  # 248
```

### Vision Encoder

```python
# Hidden size
vision_hidden_size = state_dict["visual.ln_post.weight"].shape[0]  # 768 (B) or 1024 (L)

# Number of layers
vision_layers = set()
for key in state_dict.keys():
    if key.startswith("visual.transformer.resblocks."):
        layer_num = int(key.split(".")[3])
        vision_layers.add(layer_num)
vision_num_layers = len(vision_layers)  # 12 (B) or 24 (L)

# Patch size
vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # 16 (B) or 14 (L)

# Attention heads
vision_num_heads = vision_hidden_size // 64

# Intermediate size
vision_intermediate_size = state_dict["visual.transformer.resblocks.0.mlp.c_fc.weight"].shape[0]
```

### Projection Dimension

```python
projection_dim = state_dict["text_projection"].shape[0]  # 512 (B) or 768 (L)
```

## モデルサイズ別の構成

### LongCLIP-B (Base)

| Component           | Value |
| ------------------- | ----- |
| Text hidden size    | 512   |
| Text layers         | 12    |
| Text heads          | 8     |
| Text intermediate   | 2048  |
| Vision hidden size  | 768   |
| Vision layers       | 12    |
| Vision heads        | 12    |
| Vision patch size   | 16    |
| Vision intermediate | 3072  |
| Projection dim      | 512   |
| Max positions       | 248   |
| Vocab size          | 49408 |

### LongCLIP-L (Large)

| Component           | Value |
| ------------------- | ----- |
| Text hidden size    | 768   |
| Text layers         | 12    |
| Text heads          | 12    |
| Text intermediate   | 3072  |
| Vision hidden size  | 1024  |
| Vision layers       | 24    |
| Vision heads        | 16    |
| Vision patch size   | 14    |
| Vision intermediate | 4096  |
| Projection dim      | 768   |
| Max positions       | 248   |
| Vocab size          | 49408 |

## 変換手順

### 使用方法

```bash
# LongCLIP-B を変換
python scripts/convert_longclip_to_hf.py \
    --checkpoint_path checkpoints/longclip-B.pt \
    --output_path ./longclip-base-hf

# LongCLIP-L を変換
python scripts/convert_longclip_to_hf.py \
    --checkpoint_path checkpoints/longclip-L.pt \
    --output_path ./longclip-large-hf
```

### 変換プロセス

1. **Checkpoint 読み込み**

   ```python
   state_dict = torch.load(checkpoint_path, map_location="cpu")
   ```

2. **構成の自動検出**

   ```python
   config = determine_config_from_checkpoint(state_dict)
   ```

3. **HF モデルの作成**

   ```python
   hf_model = LongCLIPModel(config)
   ```

4. **Weight のコピー**

   - Text model weights (dual positional embeddings 含む)
   - Vision model weights
   - Logit scale

5. **検証**

   - 248 トークンのテスト入力で forward pass
   - NaN/Inf チェック
   - Output shape チェック

6. **保存**
   ```python
   hf_model.save_pretrained(output_path)
   config.save_pretrained(output_path)
   ```

## 出力ファイル

変換後のディレクトリには以下のファイルが含まれます：

```
longclip-base-hf/
├── config.json           # モデル構成
└── model.safetensors     # モデルweight (SafeTensors形式)

または

├── config.json
└── pytorch_model.bin     # モデルweight (PyTorch形式)
```

## 検証

変換の正確性を確認するため、integration tests を実行：

```bash
# Baseline fixtures作成（オリジナル実装）
uv run pytest tests/test_baseline.py -v

# Integration tests（変換後のモデルと比較）
uv run pytest tests/test_integration.py -v
```

### 許容誤差

- **rtol**: 1e-2 (1% relative tolerance)
- **atol**: 1e-2 (0.01 absolute tolerance)

Baseline が float16 で保存されているため、float32 との比較では若干大きめの許容誤差が必要。

### 検証項目

1. ✅ Text features match (single and batch)
2. ✅ Image features match (single and batch)
3. ✅ Similarity scores match
4. ✅ 248 token context works correctly
5. ✅ No NaN or Inf in outputs
6. ✅ Logits symmetry (logits_per_image == logits_per_text.T)

## トラブルシューティング

### よくあるエラー

#### 1. KeyError: Layer not found

**問題:** レイヤー数のカウントが間違っている

**原因:** 総キー数をカウントしている（各レイヤーに複数のキーがある）

**解決:** 一意のレイヤーインデックスをカウント

```python
# ❌ 間違い
num_layers = len([k for k in state_dict if "resblocks" in k]) // 4

# ✅ 正しい
layers = set()
for k in state_dict.keys():
    if k.startswith("transformer.resblocks."):
        layer_num = int(k.split(".")[2])
        layers.add(layer_num)
num_layers = len(layers)
```

#### 2. Shape mismatch in projection

**問題:** `RuntimeError: shape mismatch` for text_projection

**原因:** 転置を忘れている

**解決:**

```python
# ❌ 間違い
hf_model.text_projection.weight.data = state_dict["text_projection"]

# ✅ 正しい
hf_model.text_projection.weight.data = state_dict["text_projection"].T.contiguous()
```

#### 3. Integration test failures

**問題:** テストの許容誤差内に収まらない

**原因:** Baseline が float16 で保存されている

**解決:** 適切な許容誤差を設定

```python
# float16 baseline用
RTOL = 1e-2  # 1%
ATOL = 1e-2  # 0.01
```

## まとめ

LongCLIP の checkpoint 変換の主なポイント：

1. **Dual positional embeddings**: 最も重要な特徴、正確にコピーが必要
2. **Attention in_proj の分割**: Q, K, V を 3 つに分割
3. **Projection の転置**: text_projection と visual_projection
4. **アーキテクチャ自動検出**: レイヤー数の正確なカウント
5. **検証**: Integration tests で数値的な互換性を確認

変換スクリプト: `scripts/convert_longclip_to_hf.py`
