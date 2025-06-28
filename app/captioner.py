import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_blip_processor = None
_blip_model = None

def load_blip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    global _blip_processor, _blip_model
    if _blip_model is None or _blip_processor is None:
        print("[BLIP] Loading captioning model...")
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32,
            use_safetensors=True,
            device_map=None
        ).to(device)

    return _blip_model, _blip_processor

def generate_caption(image_or_path):
    model, processor = load_blip_model()

    if isinstance(image_or_path, (str, Path)):
        image = Image.open(str(image_or_path)).convert("RGB")
    elif isinstance(image_or_path, Image.Image):
        image = image_or_path
    else:
        raise ValueError(f"Unsupported type for image input: {type(image_or_path)}")

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=25)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption









# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load BLIP model once
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# def generate_caption(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = blip_processor(images=image, return_tensors="pt").to(device)
#     out = blip_model.generate(**inputs)
#     caption = blip_processor.decode(out[0], skip_special_tokens=True)
#     return caption