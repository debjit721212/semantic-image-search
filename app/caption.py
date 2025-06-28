# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# import torch
# from PIL import Image
# from config import DEVICE
# import logging

# logger = logging.getLogger(__name__)

# # Load model once globally
# _caption_model = VisionEncoderDecoderModel.from_pretrained(
#     "nlpconnect/vit-gpt2-image-captioning", use_safetensors=True, trust_remote_code=True
# ).to(DEVICE)

# _caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# _tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# def generate_caption(image: Image.Image) -> str:
#     try:
#         if not isinstance(image, Image.Image):
#             image = Image.open(image).convert("RGB")

#         pixel_values = _caption_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
#         output_ids = _caption_model.generate(pixel_values, max_length=16, num_beams=4)
#         caption = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         return caption
#     except Exception as e:
#         logger.error(f"[CAPTION] Error generating caption: {e}")
#         return "[ERROR] Caption failed"
