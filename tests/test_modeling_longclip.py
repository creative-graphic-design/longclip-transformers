"""
Tests for LongCLIP model implementation.
"""

import pytest
import torch
from transformers import CLIPTokenizer
from PIL import Image

from longclip import (
    LongCLIPConfig,
    LongCLIPTextConfig,
    LongCLIPVisionConfig,
    LongCLIPModel,
    LongCLIPTextModel,
    LongCLIPVisionModel,
    LongCLIPTextEmbeddings,
)


def create_dummy_image(size=(224, 224)):
    """Create a dummy RGB image for testing."""
    return Image.new("RGB", size, color=(127, 127, 127))


class TestLongCLIPTextEmbeddings:
    """Test LongCLIPTextEmbeddings."""

    def test_initialization(self):
        """Test embeddings initialization."""
        config = LongCLIPTextConfig()
        embeddings = LongCLIPTextEmbeddings(config)

        assert embeddings.token_embedding.num_embeddings == config.vocab_size
        assert embeddings.token_embedding.embedding_dim == config.hidden_size
        assert embeddings.position_embedding.num_embeddings == 248
        assert embeddings.position_embedding_res.shape == (248, config.hidden_size)

    def test_mask_shapes(self):
        """Test mask shapes."""
        config = LongCLIPTextConfig()
        embeddings = LongCLIPTextEmbeddings(config)

        assert embeddings.mask1.shape == (248, 1)
        assert embeddings.mask2.shape == (248, 1)

        # Check mask values
        assert torch.all(embeddings.mask1[:20] == 1.0)
        assert torch.all(embeddings.mask1[20:] == 0.0)
        assert torch.all(embeddings.mask2[:20] == 0.0)
        assert torch.all(embeddings.mask2[20:] == 1.0)

    def test_forward_short_sequence(self):
        """Test forward pass with short sequence."""
        config = LongCLIPTextConfig()
        embeddings = LongCLIPTextEmbeddings(config)

        # Create dummy input
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

        # Forward pass
        output = embeddings(input_ids)

        assert output.shape == (batch_size, seq_length, config.hidden_size)

    def test_forward_long_sequence(self):
        """Test forward pass with long sequence (248 tokens)."""
        config = LongCLIPTextConfig()
        embeddings = LongCLIPTextEmbeddings(config)

        # Create dummy input
        batch_size, seq_length = 2, 248
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

        # Forward pass
        output = embeddings(input_ids)

        assert output.shape == (batch_size, seq_length, config.hidden_size)

    def test_mask_application(self):
        """Test that masks are correctly applied."""
        config = LongCLIPTextConfig(interpolation_keep_length=20)
        embeddings = LongCLIPTextEmbeddings(config)

        # Initialize embeddings to known values for testing
        with torch.no_grad():
            # Set position_embedding to 1.0
            embeddings.position_embedding.weight.fill_(1.0)
            # Set position_embedding_res to 2.0
            embeddings.position_embedding_res.fill_(2.0)
            # Set token embeddings to 0.0 for simplicity
            embeddings.token_embedding.weight.fill_(0.0)

        # Create input with known token IDs
        batch_size, seq_length = 1, 30
        input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)

        # Forward pass
        output = embeddings(input_ids)

        # Check that first 20 positions use position_embedding (value 1.0)
        # output[:, :20] should be approximately 1.0 (from position_embedding via mask1)
        assert torch.allclose(
            output[:, :20], torch.ones_like(output[:, :20]), atol=1e-5
        )

        # Check that positions 20-30 use position_embedding_res (value 2.0)
        # output[:, 20:] should be approximately 2.0 (from position_embedding_res via mask2)
        assert torch.allclose(
            output[:, 20:], torch.ones_like(output[:, 20:]) * 2.0, atol=1e-5
        )


class TestLongCLIPTextModel:
    """Test LongCLIPTextModel."""

    def test_initialization(self):
        """Test model initialization."""
        config = LongCLIPTextConfig()
        model = LongCLIPTextModel(config)

        assert isinstance(model.text_model.embeddings, LongCLIPTextEmbeddings)
        assert model.config.max_position_embeddings == 248

    def test_forward_pass(self):
        """Test forward pass."""
        config = LongCLIPTextConfig(
            vocab_size=49408,
            hidden_size=512,
            max_position_embeddings=248,
        )
        model = LongCLIPTextModel(config)
        model.eval()

        # Create dummy input
        batch_size, seq_length = 2, 50
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check output shapes
        assert outputs.last_hidden_state.shape == (
            batch_size,
            seq_length,
            config.hidden_size,
        )
        assert outputs.pooler_output.shape == (batch_size, config.hidden_size)

    def test_with_tokenizer(self):
        """Test with actual tokenizer."""
        config = LongCLIPTextConfig(
            vocab_size=49408,
            hidden_size=512,
            max_position_embeddings=248,
        )
        model = LongCLIPTextModel(config)
        model.eval()

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        texts = ["a photo of a cat", "a photo of a dog"]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.pooler_output.shape == (2, config.hidden_size)


class TestLongCLIPVisionModel:
    """Test LongCLIPVisionModel."""

    def test_initialization(self):
        """Test model initialization."""
        config = LongCLIPVisionConfig()
        model = LongCLIPVisionModel(config)

        assert model.config.image_size == config.image_size
        assert model.config.patch_size == config.patch_size

    def test_forward_pass(self):
        """Test forward pass."""
        config = LongCLIPVisionConfig()
        model = LongCLIPVisionModel(config)
        model.eval()

        # Create dummy input
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, config.image_size, config.image_size)

        # Forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        # Check output shape
        assert outputs.pooler_output.shape == (batch_size, config.hidden_size)


class TestLongCLIPModel:
    """Test complete LongCLIPModel."""

    def test_initialization(self):
        """Test model initialization."""
        config = LongCLIPConfig()
        model = LongCLIPModel(config)

        assert isinstance(model.text_model, LongCLIPTextModel)
        assert isinstance(model.vision_model, LongCLIPVisionModel)

    def test_get_text_features(self):
        """Test getting text features."""
        config = LongCLIPConfig(
            text_config={"vocab_size": 49408, "hidden_size": 512},
            projection_dim=512,
        )
        model = LongCLIPModel(config)
        model.eval()

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        text = ["a photo of a cat"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        assert text_features.shape == (1, config.projection_dim)

    def test_get_image_features(self):
        """Test getting image features."""
        config = LongCLIPConfig(projection_dim=512)
        model = LongCLIPModel(config)
        model.eval()

        # Create dummy image input
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=pixel_values)

        assert image_features.shape == (batch_size, config.projection_dim)

    def test_forward_pass(self):
        """Test complete forward pass."""
        config = LongCLIPConfig(
            text_config={"vocab_size": 49408, "hidden_size": 512},
            projection_dim=512,
        )
        model = LongCLIPModel(config)
        model.eval()

        # Prepare inputs
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        texts = ["a photo of a cat", "a photo of a dog"]
        text_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )

        pixel_values = torch.randn(2, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=pixel_values,
            )

        # Check outputs
        assert outputs.logits_per_image.shape == (2, 2)
        assert outputs.logits_per_text.shape == (2, 2)
        assert outputs.text_embeds.shape == (2, config.projection_dim)
        assert outputs.image_embeds.shape == (2, config.projection_dim)

    def test_similarity_computation(self):
        """Test image-text similarity computation."""
        config = LongCLIPConfig(
            text_config={"vocab_size": 49408, "hidden_size": 512},
            projection_dim=512,
        )
        model = LongCLIPModel(config)
        model.eval()

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        texts = ["a photo of a cat"]
        text_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )

        pixel_values = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outputs = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=pixel_values,
            )

        # Similarity should be in reasonable range
        similarity = outputs.logits_per_image[0, 0]
        # After softmax, should be between 0 and 1
        probs = outputs.logits_per_image.softmax(dim=1)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
