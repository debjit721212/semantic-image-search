# eval_clip_similarity.py

import os
import open_clip
import torch
from PIL import Image
from torchvision import transforms
from openclip_lora_module import LoRALinear
from tqdm import tqdm

def apply_lora_to_model(model, r=8, alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(k in name for k in ["q_proj", "v_proj", "proj"]):
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], LoRALinear(module, r=r, alpha=alpha))
    return model

def load_model(pretrained=True, checkpoint=None, device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', 'laion2b_s34b_b79k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    if pretrained is False:
        model = apply_lora_to_model(model)
        model.load_state_dict(torch.load(checkpoint))
        print("✅ LoRA weights applied")

    model.eval()
    return model, tokenizer, preprocess

def evaluate_batch(image_folder, prompts, model, tokenizer, preprocess, device):
    image_features = []
    image_paths = []

    print(f"🖼️ Encoding images from {image_folder}")
    for file in tqdm(os.listdir(image_folder)):
        if not file.lower().endswith(".jpg"):
            continue
        image_path = os.path.join(image_folder, file)
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)
        image_features.append(features)
        image_paths.append(file)

    image_features = torch.cat(image_features, dim=0)

    text_inputs = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compare each prompt to each image
    for i, prompt in enumerate(prompts):
        print(f"\n🔍 Prompt: {prompt}")
        sims = (image_features @ text_features[i].unsqueeze(1)).squeeze()
        topk = torch.topk(sims, k=3)
        for idx in topk.indices:
            print(f"  - {image_paths[idx]}  | Similarity: {sims[idx]:.4f}")

if __name__ == "__main__":
    prompts = [
        "man wearing a red shirt",
        "child climbing stairs",
        "worker operating pulley"
    ]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="Folder of test images")
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint. Leave empty for original CLIP")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load appropriate model
    if args.checkpoint:
        model, tokenizer, preprocess = load_model(pretrained=False, checkpoint=args.checkpoint, device=device)
    else:
        model, tokenizer, preprocess = load_model(pretrained=True, device=device)

    evaluate_batch(args.img_dir, prompts, model, tokenizer, preprocess, device)