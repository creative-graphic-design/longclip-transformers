# Scripts

このディレクトリには、LongCLIP モデルの変換とアップロードに使用するスクリプトが含まれています。

## convert_longclip_to_hf.py

オリジナルの LongCLIP チェックポイント（.pt ファイル）を Hugging Face Transformers 形式に変換します。

### 使用方法

```bash
python scripts/convert_longclip_to_hf.py \
    --checkpoint_path checkpoints/longclip-B.pt \
    --output_path ./longclip-base-hf \
    --model_size B
```

### パラメータ

- `--checkpoint_path`: オリジナルの LongCLIP チェックポイントのパス（.pt ファイル）
- `--output_path`: 変換後のモデルを保存するパス
- `--no-validate`: 検証ステップをスキップする（オプション）

### 例

```bash
# LongCLIP-Bを変換
python scripts/convert_longclip_to_hf.py \
    --checkpoint_path checkpoints/longclip-B.pt \
    --output_path ./longclip-base-hf

# LongCLIP-Lを変換
python scripts/convert_longclip_to_hf.py \
    --checkpoint_path checkpoints/longclip-L.pt \
    --output_path ./longclip-large-hf
```

## push_to_hub.py

変換済みの LongCLIP モデルを Hugging Face Hub にアップロードします。

### 前提条件

1. Hugging Face アカウントを作成: https://huggingface.co/join
2. アクセストークンを取得: https://huggingface.co/settings/tokens
   - `write` 権限が必要です

### 使用方法

#### 方法 1: トークンを環境変数に設定

```bash
export HF_TOKEN=your_token_here

python scripts/push_to_hub.py \
    --model_path ./longclip-base-hf \
    --repo_id your-username/longclip-base \
    --model_size B
```

#### 方法 2: トークンをコマンドライン引数で指定

```bash
python scripts/push_to_hub.py \
    --model_path ./longclip-base-hf \
    --repo_id your-username/longclip-base \
    --token your_token_here \
    --model_size B
```

### パラメータ

- `--model_path`: ローカルのモデルディレクトリのパス（必須）
- `--repo_id`: Hub 上のリポジトリ ID（例: `username/longclip-base`）（必須）
- `--token`: Hugging Face API トークン（オプション、環境変数 `HF_TOKEN` でも設定可能）
- `--private`: プライベートリポジトリとして作成（オプション）
- `--model_size`: モデルサイズ（`B` または `L`、デフォルト: `B`）
- `--commit_message`: カスタムコミットメッセージ（オプション）

### 例

```bash
# パブリックリポジトリにアップロード
export HF_TOKEN=hf_xxxxxxxxxxxxx
python scripts/push_to_hub.py \
    --model_path ./longclip-base-hf \
    --repo_id myusername/longclip-base \
    --model_size B

# プライベートリポジトリにアップロード
python scripts/push_to_hub.py \
    --model_path ./longclip-large-hf \
    --repo_id myusername/longclip-large \
    --token hf_xxxxxxxxxxxxx \
    --model_size L \
    --private

# カスタムコミットメッセージを使用
python scripts/push_to_hub.py \
    --model_path ./longclip-base-hf \
    --repo_id myusername/longclip-base \
    --model_size B \
    --commit_message "Initial upload of LongCLIP-B model"
```

### アップロード後の使用方法

モデルがアップロードされると、以下のように使用できます：

```python
from longclip import LongCLIPModel, LongCLIPProcessor
from PIL import Image

# モデルとプロセッサーをロード
model = LongCLIPModel.from_pretrained("your-username/longclip-base")
processor = LongCLIPProcessor.from_pretrained("your-username/longclip-base")

# 使用例
image = Image.open("path/to/image.jpg")
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)
print("Probabilities:", probs)
```

## ワークフロー全体

オリジナルのチェックポイントから Hub へのアップロードまでの完全なワークフロー：

```bash
# 1. チェックポイントを変換
python scripts/convert_longclip_to_hf.py \
    --checkpoint_path checkpoints/longclip-B.pt \
    --output_path ./longclip-base-hf

# 2. 環境変数を設定
export HF_TOKEN=your_token_here

# 3. Hubにプッシュ
python scripts/push_to_hub.py \
    --model_path ./longclip-base-hf \
    --repo_id your-username/longclip-base \
    --model_size B
```

## トラブルシューティング

### 認証エラー

```
HTTPError: 401 Client Error: Unauthorized
```

→ トークンが無効または期限切れです。新しいトークンを取得してください。

### アップロードが遅い

大きなモデル（LongCLIP-L）のアップロードには時間がかかる場合があります。
高速なインターネット接続を使用することをお勧めします。

### リポジトリがすでに存在する

スクリプトは既存のリポジトリを上書きします（`exist_ok=True`）。
新しいバージョンをアップロードする場合は、カスタムコミットメッセージを使用してください。
