"""
Tests for LongCLIP configuration classes.
"""

import pytest
from longclip import LongCLIPConfig, LongCLIPTextConfig, LongCLIPVisionConfig


class TestLongCLIPTextConfig:
    """Test LongCLIPTextConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LongCLIPTextConfig()

        assert config.max_position_embeddings == 248
        assert config.use_position_interpolation is True
        assert config.interpolation_keep_length == 20
        assert config.model_type == "longclip_text_model"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LongCLIPTextConfig(
            max_position_embeddings=300,
            hidden_size=768,
            use_position_interpolation=False,
            interpolation_keep_length=30,
        )

        assert config.max_position_embeddings == 300
        assert config.hidden_size == 768
        assert config.use_position_interpolation is False
        assert config.interpolation_keep_length == 30

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = LongCLIPTextConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "max_position_embeddings" in config_dict
        assert "use_position_interpolation" in config_dict
        assert "interpolation_keep_length" in config_dict

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        config_dict = {
            "max_position_embeddings": 248,
            "hidden_size": 512,
            "use_position_interpolation": True,
            "interpolation_keep_length": 20,
        }

        config = LongCLIPTextConfig(**config_dict)

        assert config.max_position_embeddings == 248
        assert config.hidden_size == 512


class TestLongCLIPVisionConfig:
    """Test LongCLIPVisionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LongCLIPVisionConfig()

        assert config.model_type == "longclip_vision_model"
        # Should have standard CLIP vision config attributes
        assert hasattr(config, "image_size")
        assert hasattr(config, "patch_size")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LongCLIPVisionConfig(
            image_size=336,
            patch_size=14,
            hidden_size=1024,
        )

        assert config.image_size == 336
        assert config.patch_size == 14
        assert config.hidden_size == 1024


class TestLongCLIPConfig:
    """Test LongCLIPConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LongCLIPConfig()

        assert config.model_type == "longclip"
        assert isinstance(config.text_config, LongCLIPTextConfig)
        assert isinstance(config.vision_config, LongCLIPVisionConfig)
        assert config.text_config.max_position_embeddings == 248

    def test_custom_config(self):
        """Test custom configuration with dictionaries."""
        text_config = {"max_position_embeddings": 248, "hidden_size": 512}
        vision_config = {"image_size": 224, "patch_size": 16}

        config = LongCLIPConfig(text_config=text_config, vision_config=vision_config)

        assert config.text_config.max_position_embeddings == 248
        assert config.text_config.hidden_size == 512
        assert config.vision_config.image_size == 224
        assert config.vision_config.patch_size == 16

    def test_from_text_vision_configs(self):
        """Test creating config from text and vision configs."""
        text_config = LongCLIPTextConfig(max_position_embeddings=248)
        vision_config = LongCLIPVisionConfig(image_size=224)

        config = LongCLIPConfig.from_text_vision_configs(
            text_config=text_config,
            vision_config=vision_config,
        )

        assert isinstance(config, LongCLIPConfig)
        assert config.text_config.max_position_embeddings == 248
        assert config.vision_config.image_size == 224

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = LongCLIPConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "text_config" in config_dict
        assert "vision_config" in config_dict
        assert isinstance(config_dict["text_config"], dict)
        assert isinstance(config_dict["vision_config"], dict)

    def test_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        config = LongCLIPConfig()

        # Save
        config.save_pretrained(tmp_path)

        # Load
        loaded_config = LongCLIPConfig.from_pretrained(tmp_path)

        assert loaded_config.text_config.max_position_embeddings == 248
        assert loaded_config.model_type == "longclip"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
