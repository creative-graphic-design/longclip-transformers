"""
LongCLIP processor for preprocessing images and text.

This module provides a processor that combines image and text preprocessing
for LongCLIP models.
"""

from typing import List, Optional, Union

from transformers import CLIPImageProcessor, CLIPTokenizer
from transformers.processing_utils import ProcessorMixin


class LongCLIPProcessor(ProcessorMixin):
    """
    Processor for LongCLIP that combines image and text preprocessing.

    This processor wraps CLIPImageProcessor and CLIPTokenizer to provide
    a unified interface for preprocessing inputs for LongCLIP models.

    Args:
        image_processor (CLIPImageProcessor): Image processor for preprocessing images.
        tokenizer (CLIPTokenizer): Tokenizer for preprocessing text.

    Attributes:
        image_processor_class (str): Name of the image processor class.
        tokenizer_class (str): Name of the tokenizer class.

    Example:
        ```python
        >>> from long_clip_hf import LongCLIPProcessor
        >>> from transformers import CLIPImageProcessor, CLIPTokenizer
        >>> from PIL import Image
        >>>
        >>> # Initialize processor
        >>> image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = LongCLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
        >>>
        >>> # Process inputs
        >>> image = Image.open("path/to/image.jpg")
        >>> text = "a photo of a cat"
        >>> inputs = processor(text=text, images=image, return_tensors="pt", padding=True, max_length=248)
        >>>
        >>> # inputs contains both 'input_ids', 'attention_mask' and 'pixel_values'
        ```
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "CLIPTokenizer"

    def __init__(
        self,
        image_processor: Optional[CLIPImageProcessor] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[str, List[str], None] = None,
        images=None,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = True,
        max_length: Optional[int] = 248,
        truncation: Optional[bool] = True,
        **kwargs,
    ):
        """
        Preprocess text and images for LongCLIP model.

        Args:
            text (str, List[str], optional): Text or list of texts to process.
            images: Image or list of images to process. Can be PIL Image, numpy array, or tensor.
            return_tensors (str, optional): Type of tensors to return ('pt' for PyTorch).
            padding (bool or str, optional): Padding strategy. Defaults to True.
            max_length (int, optional): Maximum sequence length. Defaults to 248 for LongCLIP.
            truncation (bool, optional): Whether to truncate sequences. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            BatchEncoding: Dictionary containing processed inputs with keys:
                - input_ids: Tokenized text (if text provided)
                - attention_mask: Attention mask for text (if text provided)
                - pixel_values: Processed images (if images provided)
        """
        # Process text
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                **kwargs,
            )
        else:
            text_inputs = {}

        # Process images
        if images is not None:
            image_inputs = self.image_processor(
                images,
                return_tensors=return_tensors,
            )
        else:
            image_inputs = {}

        # Combine inputs
        return {**text_inputs, **image_inputs}

    def batch_decode(self, *args, **kwargs):
        """
        Decode token IDs back to text.

        This method is forwarded to the tokenizer's batch_decode method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Decode token IDs back to text.

        This method is forwarded to the tokenizer's decode method.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Get the names of model inputs.

        Returns:
            List[str]: List of input names.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
