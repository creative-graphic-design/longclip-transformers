"""
Tests for LongCLIPProcessor.
"""

import pytest
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer

from long_clip_hf import LongCLIPProcessor


def create_dummy_image(size=(224, 224)):
    """Create a dummy RGB image for testing."""
    return Image.new("RGB", size, color=(127, 127, 127))


class TestLongCLIPProcessor:
    """Test LongCLIPProcessor."""

    @pytest.fixture
    def processor(self):
        """Create processor fixture."""
        image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        return LongCLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert hasattr(processor, "image_processor")
        assert hasattr(processor, "tokenizer")

    def test_text_only(self, processor):
        """Test processing text only."""
        text = "a photo of a cat"
        inputs = processor(
            text=text, return_tensors="pt", padding="max_length", max_length=248
        )

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape[1] == 248

    def test_image_only(self, processor):
        """Test processing image only."""
        image = create_dummy_image()
        inputs = processor(images=image, return_tensors="pt")

        assert "pixel_values" in inputs
        assert inputs["pixel_values"].shape[1] == 3  # RGB channels
        assert inputs["pixel_values"].shape[2] == 224  # Height
        assert inputs["pixel_values"].shape[3] == 224  # Width

    def test_text_and_image(self, processor):
        """Test processing both text and image."""
        text = "a photo of a cat"
        image = create_dummy_image()

        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
            max_length=248,
            padding="max_length",
            truncation=True,
        )

        # Check text outputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape[1] == 248

        # Check image outputs
        assert "pixel_values" in inputs
        assert inputs["pixel_values"].shape[1] == 3

    def test_batch_processing(self, processor):
        """Test processing batches."""
        texts = ["a photo of a cat", "a photo of a dog"]
        images = [create_dummy_image(), create_dummy_image()]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            max_length=248,
            padding=True,
            truncation=True,
        )

        # Check batch sizes
        assert inputs["input_ids"].shape[0] == 2
        assert inputs["pixel_values"].shape[0] == 2

    def test_long_text(self, processor):
        """Test processing long text (248 tokens)."""
        long_text = " ".join(["word"] * 300)  # Very long text

        inputs = processor(
            text=long_text,
            return_tensors="pt",
            max_length=248,
            truncation=True,
        )

        # Should be truncated to 248
        assert inputs["input_ids"].shape[1] == 248

    def test_decode(self, processor):
        """Test decoding tokens back to text."""
        text = "a photo of a cat"
        inputs = processor(text=text, return_tensors="pt", max_length=248)

        # Decode
        decoded = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)

        assert "photo" in decoded.lower()
        assert "cat" in decoded.lower()

    def test_batch_decode(self, processor):
        """Test batch decoding."""
        texts = ["a photo of a cat", "a photo of a dog"]
        inputs = processor(text=texts, return_tensors="pt", max_length=248)

        # Batch decode
        decoded = processor.batch_decode(inputs["input_ids"], skip_special_tokens=True)

        assert len(decoded) == 2
        assert "cat" in decoded[0].lower()
        assert "dog" in decoded[1].lower()

    def test_model_input_names(self, processor):
        """Test getting model input names."""
        input_names = processor.model_input_names

        assert isinstance(input_names, list)
        assert "input_ids" in input_names or "pixel_values" in input_names

    def test_padding_strategies(self, processor):
        """Test different padding strategies."""
        texts = ["cat", "a photo of a very long description"]

        # Test max_length padding
        inputs_max = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
        )
        assert inputs_max["input_ids"].shape[1] == 248

        # Test longest padding
        inputs_longest = processor(
            text=texts,
            return_tensors="pt",
            padding="longest",
            max_length=248,
        )
        # Should pad to longest sequence in batch, not necessarily 248
        assert inputs_longest["input_ids"].shape[1] <= 248

    def test_with_none_inputs(self, processor):
        """Test with None inputs (should handle gracefully)."""
        # Only text
        inputs_text = processor(
            text="test", images=None, return_tensors="pt", max_length=248
        )
        assert "input_ids" in inputs_text
        assert "pixel_values" not in inputs_text

        # Only image
        inputs_image = processor(
            text=None, images=create_dummy_image(), return_tensors="pt"
        )
        assert "pixel_values" in inputs_image
        assert "input_ids" not in inputs_image


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
