# test_openclip_lora.py

import open_clip
import torch
from PIL import Image
from openclip_lora_module import LoRALinear
import os

def apply_lora_to_model(model, r=8, alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(k in name for k in ["q_proj", "v_proj", "proj"]):
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], LoRALinear(module, r=r, alpha=alpha))
    return model

def load_lora_model(device, checkpoint_path, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = apply_lora_to_model(model)

    print(f"[INFO] Loading LoRA weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model, preprocess, tokenizer

def test_similarity(image_path, prompt, model, preprocess, tokenizer, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=1, keepdim=True)
        text_features /= text_features.norm(dim=1, keepdim=True)

        similarity = image_features @ text_features.T
        print(f"🧠 Cosine Similarity: {similarity.item():.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--text", required=True, help="Prompt/query sentence")
    parser.add_argument("--checkpoint", default="openclip_lora_output/clip_lora.pth", help="Path to LoRA checkpoint")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_lora_model(device, args.checkpoint)
    test_similarity(args.image, args.text, model, preprocess, tokenizer, device)