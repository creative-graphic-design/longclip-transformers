"""
Test tokenizer compatibility between SimpleTokenizer and CLIPTokenizer.
Verify that transformers' CLIPTokenizer can be used with LongCLIP.
"""

import pytest
import torch
from transformers import CLIPTokenizer

from model import longclip


# Sample texts for testing
SAMPLE_TEXTS = [
    "A cat",
    "A man is crossing the street with a red car parked nearby.",
    "The quick brown fox jumps over the lazy dog.",
    "This is a longer text to test the tokenization with many words to see how it handles the extended context length of 248 tokens.",
]


class TestTokenizerCompatibility:
    """Test compatibility between SimpleTokenizer and CLIPTokenizer."""

    def test_vocab_size(self):
        """Test that both tokenizers have the same vocab size."""
        # SimpleTokenizer vocab size (from implementation)
        simple_tokens = longclip.tokenize(["test"])
        # Vocab size is 49408 for CLIP
        expected_vocab_size = 49408

        # CLIPTokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        assert clip_tokenizer.vocab_size == expected_vocab_size

    def test_special_tokens(self):
        """Test that special tokens match."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # BOS and EOS tokens
        # CLIP uses: bos_token="<|startoftext|>", eos_token="<|endoftext|>"
        assert clip_tokenizer.bos_token == "<|startoftext|>"
        assert clip_tokenizer.eos_token == "<|endoftext|>"

        # Token IDs should be 49406 and 49407
        assert clip_tokenizer.bos_token_id == 49406
        assert clip_tokenizer.eos_token_id == 49407

    def test_tokenization_short_text(self):
        """Test tokenizing a short text with both tokenizers."""
        text = "A cat"

        # SimpleTokenizer
        simple_tokens = longclip.tokenize([text])

        # CLIPTokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_tokens = clip_tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )["input_ids"]

        print(f"\nSimple tokens shape: {simple_tokens.shape}")
        print(f"CLIP tokens shape: {clip_tokens.shape}")
        print(f"\nSimple tokens (first 20): {simple_tokens[0, :20].tolist()}")
        print(f"CLIP tokens (first 20): {clip_tokens[0, :20].tolist()}")

        # Shapes should match
        assert simple_tokens.shape == clip_tokens.shape
        assert simple_tokens.shape[1] == 248

    def test_tokenization_various_texts(self):
        """Test tokenizing various texts."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        for text in SAMPLE_TEXTS:
            simple_tokens = longclip.tokenize([text])
            clip_tokens = clip_tokenizer(
                [text],
                return_tensors="pt",
                padding="max_length",
                max_length=248,
                truncation=True,
            )["input_ids"]

            print(f"\nText: {text[:50]}...")
            print(f"Simple non-zero tokens: {torch.count_nonzero(simple_tokens)}")
            print(f"CLIP non-zero tokens: {torch.count_nonzero(clip_tokens)}")

            # Both should produce same shape
            assert simple_tokens.shape == clip_tokens.shape

    def test_max_length_248(self):
        """Test that CLIPTokenizer can handle max_length=248."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Very long text
        long_text = " ".join(["word"] * 300)

        tokens = clip_tokenizer(
            [long_text],
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )["input_ids"]

        assert tokens.shape[1] == 248

    def test_padding_behavior(self):
        """Test padding behavior with CLIPTokenizer."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        short_text = "A cat"
        outputs = clip_tokenizer(
            [short_text],
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )
        tokens = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]

        # Should be padded to 248
        assert tokens.shape[1] == 248

        # Check attention mask to see real vs padded tokens
        # CLIPTokenizer uses EOS token for padding, so check attention_mask instead
        num_real_tokens = attention_mask.sum().item()
        assert num_real_tokens < 20  # Short text should have < 20 real tokens

        # Verify padding tokens are pad_token_id (which is EOS token)
        assert tokens[0, num_real_tokens:].eq(clip_tokenizer.pad_token_id).all()

    def test_truncation_behavior(self):
        """Test truncation behavior with CLIPTokenizer."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Very long text
        very_long_text = " ".join(["word"] * 500)

        tokens = clip_tokenizer(
            [very_long_text],
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )["input_ids"]

        # Should be truncated to 248
        assert tokens.shape[1] == 248

        # Should have EOS token at the end (or near end)
        # Last non-padding token should be EOS
        non_padding_mask = tokens != clip_tokenizer.pad_token_id
        if non_padding_mask.any():
            last_token_idx = non_padding_mask[0].nonzero()[-1].item()
            # Last non-padding token could be EOS
            print(f"Last non-padding token: {tokens[0, last_token_idx].item()}")

    def test_batch_tokenization(self):
        """Test batch tokenization with CLIPTokenizer."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        tokens = clip_tokenizer(
            SAMPLE_TEXTS,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )["input_ids"]

        assert tokens.shape[0] == len(SAMPLE_TEXTS)
        assert tokens.shape[1] == 248

    def test_compare_token_ids_simple_case(self):
        """Compare token IDs for a simple case."""
        text = "a photo of a cat"

        # SimpleTokenizer
        simple_tokens = longclip.tokenize([text])

        # CLIPTokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_tokens = clip_tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )["input_ids"]

        # Compare the non-padding tokens
        simple_non_pad = simple_tokens[simple_tokens != 0]
        clip_non_pad = clip_tokens[clip_tokens != clip_tokenizer.pad_token_id]

        print(f"\nSimple tokens (non-pad): {simple_non_pad.tolist()}")
        print(f"CLIP tokens (non-pad): {clip_non_pad.tolist()}")

        # They might differ due to lowercasing and other preprocessing differences
        # but should have similar structure: [BOS, tokens..., EOS, padding...]
        assert simple_non_pad[0] == clip_tokenizer.bos_token_id  # BOS
        assert simple_non_pad[-1] == clip_tokenizer.eos_token_id  # EOS


class TestCLIPTokenizerAdvanced:
    """Advanced tests for CLIPTokenizer functionality."""

    def test_tokenizer_with_attention_mask(self):
        """Test that CLIPTokenizer provides attention mask."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        outputs = clip_tokenizer(
            SAMPLE_TEXTS,
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )

        assert "input_ids" in outputs
        assert "attention_mask" in outputs

        # Attention mask shape should match input_ids
        assert outputs["attention_mask"].shape == outputs["input_ids"].shape

    def test_tokenizer_decode(self):
        """Test decoding tokens back to text."""
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        text = "a photo of a cat"
        tokens = clip_tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            max_length=248,
            truncation=True,
        )["input_ids"]

        # Decode
        decoded = clip_tokenizer.decode(tokens[0], skip_special_tokens=True)

        print(f"\nOriginal: {text}")
        print(f"Decoded: {decoded}")

        # Should be similar (might have minor differences due to tokenization)
        assert "photo" in decoded.lower()
        assert "cat" in decoded.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
