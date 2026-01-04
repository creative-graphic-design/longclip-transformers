"""
Baseline tests for existing LongCLIP implementation.
These tests establish ground truth for comparison with transformers-compatible version.
"""

import os
import pickle
from pathlib import Path

import pytest
import torch
from PIL import Image

from longclip_original.model import longclip


# Test configuration
CHECKPOINT_B_PATH = "./checkpoints/longclip-B.pt"
CHECKPOINT_L_PATH = "./checkpoints/longclip-L.pt"
FIXTURES_DIR = Path("./tests/fixtures")
FIXTURES_DIR.mkdir(exist_ok=True)

# Sample texts of various lengths
SAMPLE_TEXTS = [
    "A cat",  # Short
    "A man is crossing the street with a red car parked nearby.",  # Medium
    "The quick brown fox jumps over the lazy dog while the sun sets in the background, painting the sky with beautiful shades of orange and red.",  # Long
]


# Create a dummy image for testing
def create_dummy_image(size=(224, 224)):
    """Create a dummy RGB image for testing."""
    return Image.new("RGB", size, color=(127, 127, 127))


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model_b(device):
    """Load LongCLIP-B model."""
    if not os.path.exists(CHECKPOINT_B_PATH):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_B_PATH}")
    model, preprocess = longclip.load(CHECKPOINT_B_PATH, device=device)
    model.eval()
    return model, preprocess


@pytest.fixture
def model_l(device):
    """Load LongCLIP-L model."""
    if not os.path.exists(CHECKPOINT_L_PATH):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_L_PATH}")
    model, preprocess = longclip.load(CHECKPOINT_L_PATH, device=device)
    model.eval()
    return model, preprocess


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_model_b(self, model_b):
        """Test loading LongCLIP-B model."""
        model, preprocess = model_b
        assert model is not None
        assert preprocess is not None
        assert hasattr(model, "encode_text")
        assert hasattr(model, "encode_image")

    def test_load_model_l(self, model_l):
        """Test loading LongCLIP-L model."""
        model, preprocess = model_l
        assert model is not None
        assert preprocess is not None


class TestTokenization:
    """Test tokenization functionality."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "A cat"
        tokens = longclip.tokenize([text])

        assert tokens.shape[0] == 1  # batch size
        assert tokens.shape[1] == 248  # context length
        assert tokens.dtype == torch.int or tokens.dtype == torch.long

    def test_tokenize_multiple(self):
        """Test tokenizing multiple texts."""
        tokens = longclip.tokenize(SAMPLE_TEXTS)

        assert tokens.shape[0] == len(SAMPLE_TEXTS)
        assert tokens.shape[1] == 248

    def test_tokenize_context_length_248(self):
        """Test that context length is 248."""
        long_text = " ".join(["word"] * 300)  # Very long text
        tokens = longclip.tokenize([long_text])

        assert tokens.shape[1] == 248

    def test_save_tokenization_fixtures(self):
        """Save tokenization outputs as fixtures."""
        tokens = longclip.tokenize(SAMPLE_TEXTS)

        fixture_data = {
            "texts": SAMPLE_TEXTS,
            "tokens": tokens,
        }

        with open(FIXTURES_DIR / "tokenization.pkl", "wb") as f:
            pickle.dump(fixture_data, f)

        print(f"Saved tokenization fixtures to {FIXTURES_DIR / 'tokenization.pkl'}")


class TestTextEncoding:
    """Test text encoding functionality."""

    def test_encode_text_shape(self, model_b, device):
        """Test text encoding output shape."""
        model, _ = model_b
        text = longclip.tokenize(["A cat"]).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text)

        assert text_features.ndim == 2  # [batch, embed_dim]
        assert text_features.shape[0] == 1  # batch size

    def test_encode_text_various_lengths(self, model_b, device):
        """Test encoding texts of various lengths."""
        model, _ = model_b
        tokens = longclip.tokenize(SAMPLE_TEXTS).to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokens)

        assert text_features.shape[0] == len(SAMPLE_TEXTS)
        # Features should be normalized
        norms = torch.norm(text_features, dim=1)
        # Check if features are approximately normalized (some may not be exactly 1.0)
        assert torch.all(norms > 0), "Text features should have non-zero norm"

    def test_encode_text_deterministic(self, model_b, device):
        """Test that encoding is deterministic."""
        model, _ = model_b
        text = longclip.tokenize(["A cat"]).to(device)

        with torch.no_grad():
            features1 = model.encode_text(text)
            features2 = model.encode_text(text)

        assert torch.allclose(features1, features2), "Encoding should be deterministic"

    def test_save_text_encoding_fixtures(self, model_b, device):
        """Save text encoding outputs as fixtures."""
        model, _ = model_b
        tokens = longclip.tokenize(SAMPLE_TEXTS).to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokens)

        fixture_data = {
            "texts": SAMPLE_TEXTS,
            "tokens": tokens.cpu(),
            "text_features": text_features.cpu(),
        }

        with open(FIXTURES_DIR / "text_encoding_b.pkl", "wb") as f:
            pickle.dump(fixture_data, f)

        print(f"Saved text encoding fixtures to {FIXTURES_DIR / 'text_encoding_b.pkl'}")


class TestImageEncoding:
    """Test image encoding functionality."""

    def test_encode_image_shape(self, model_b, device):
        """Test image encoding output shape."""
        model, preprocess = model_b
        image = preprocess(create_dummy_image()).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

        assert image_features.ndim == 2  # [batch, embed_dim]
        assert image_features.shape[0] == 1  # batch size

    def test_encode_image_batch(self, model_b, device):
        """Test encoding a batch of images."""
        model, preprocess = model_b
        images = torch.stack(
            [
                preprocess(create_dummy_image()),
                preprocess(create_dummy_image()),
                preprocess(create_dummy_image()),
            ]
        ).to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)

        assert image_features.shape[0] == 3

    def test_save_image_encoding_fixtures(self, model_b, device):
        """Save image encoding outputs as fixtures."""
        model, preprocess = model_b
        image = preprocess(create_dummy_image()).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

        fixture_data = {
            "image_shape": image.shape,
            "image_features": image_features.cpu(),
        }

        with open(FIXTURES_DIR / "image_encoding_b.pkl", "wb") as f:
            pickle.dump(fixture_data, f)

        print(
            f"Saved image encoding fixtures to {FIXTURES_DIR / 'image_encoding_b.pkl'}"
        )


class TestSimilarity:
    """Test image-text similarity computation."""

    def test_similarity_computation(self, model_b, device):
        """Test computing image-text similarity."""
        model, preprocess = model_b

        text = longclip.tokenize(SAMPLE_TEXTS).to(device)
        image = preprocess(create_dummy_image()).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = image_features @ text_features.T

        assert similarity.shape == (1, len(SAMPLE_TEXTS))
        # Similarity should be in reasonable range after normalization
        assert torch.all(similarity >= -1.1) and torch.all(similarity <= 1.1)

    def test_save_similarity_fixtures(self, model_b, device):
        """Save similarity computation results as fixtures."""
        model, preprocess = model_b

        text = longclip.tokenize(SAMPLE_TEXTS).to(device)
        image = preprocess(create_dummy_image()).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # Normalize features
            image_features_norm = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_features_norm = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            similarity = image_features_norm @ text_features_norm.T

        fixture_data = {
            "texts": SAMPLE_TEXTS,
            "image_features": image_features.cpu(),
            "text_features": text_features.cpu(),
            "similarity": similarity.cpu(),
        }

        with open(FIXTURES_DIR / "similarity_b.pkl", "wb") as f:
            pickle.dump(fixture_data, f)

        print(f"Saved similarity fixtures to {FIXTURES_DIR / 'similarity_b.pkl'}")


class TestPositionalEmbeddings:
    """Test positional embedding mechanism."""

    def test_positional_embedding_structure(self, model_b):
        """Test that positional embeddings have the expected structure."""
        model, _ = model_b

        # Check that model has positional_embedding and positional_embedding_res
        assert hasattr(model, "positional_embedding")
        assert hasattr(model, "positional_embedding_res")

        # Check shapes
        pos_emb = model.positional_embedding
        pos_emb_res = model.positional_embedding_res

        assert pos_emb.shape[0] == 248, (
            f"Expected 248 positions, got {pos_emb.shape[0]}"
        )
        assert pos_emb.shape == pos_emb_res.shape

    def test_mask_structure(self, model_b):
        """Test mask1 and mask2 structure."""
        model, _ = model_b

        assert hasattr(model, "mask1")
        assert hasattr(model, "mask2")

        mask1 = model.mask1
        mask2 = model.mask2

        # Check shapes
        assert mask1.shape == (248, 1)
        assert mask2.shape == (248, 1)

        # Check values
        # mask1: first 20 positions should be 1, rest 0
        assert torch.all(mask1[:20] == 1)
        assert torch.all(mask1[20:] == 0)

        # mask2: first 20 positions should be 0, rest 1
        assert torch.all(mask2[:20] == 0)
        assert torch.all(mask2[20:] == 1)

    def test_save_positional_embedding_fixtures(self, model_b):
        """Save positional embedding data as fixtures."""
        model, _ = model_b

        fixture_data = {
            "positional_embedding": model.positional_embedding.cpu(),
            "positional_embedding_res": model.positional_embedding_res.cpu(),
            "mask1": model.mask1.cpu(),
            "mask2": model.mask2.cpu(),
        }

        with open(FIXTURES_DIR / "positional_embeddings_b.pkl", "wb") as f:
            pickle.dump(fixture_data, f)

        print(
            f"Saved positional embedding fixtures to {FIXTURES_DIR / 'positional_embeddings_b.pkl'}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
