# train_lora_openclip.py

import open_clip
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim  # ✅ this was missing
from tqdm import tqdm

from openclip_lora_module import LoRALinear
from flickr_csv_dataset_openclip import FlickrOpenCLIPDataset


# ✅ LoRA injection
def apply_lora_to_model(model, r=8, alpha=16):
    print("[INFO] Injecting LoRA into text and vision attention layers...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(key in name for key in ["q_proj", "v_proj", "proj"]):
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], LoRALinear(module, r=r, alpha=alpha))
            print(f"[LoRA Injected] - {name}")
    return model


# ✅ Main Trainer Function
def train_openclip(
    csv_path,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    batch_size=8,
    epochs=5,
    lr=1e-4,
    output_dir="./openclip_lora_output",
    r=8,
    alpha=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model + preprocessing transforms
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # Apply LoRA
    model = apply_lora_to_model(model, r=r, alpha=alpha)
    model.to(device)
    model.train()

    # Prepare DataLoader
    dataset = FlickrOpenCLIPDataset(csv_path, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    os.makedirs(output_dir, exist_ok=True)

    # 🚀 Training Loop
    for epoch in range(epochs):
        print(f"\n📚 Starting Epoch [{epoch + 1}/{epochs}]")
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training")

        for batch in pbar:
            images = batch["image"].to(device)
            texts = batch["text"]  # list[str]
            text_tokens = tokenizer(texts).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            # Normalize (safe style)
            image_features = image_features / image_features.norm(dim=1, keepdim=True).clone()
            text_features = text_features / text_features.norm(dim=1, keepdim=True).clone()

            # Cosine similarity
            similarity = image_features @ text_features.T
            labels = torch.arange(len(images), device=device)

            loss = F.cross_entropy(similarity, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch + 1}] Completed - ✅ Avg Loss: {avg_loss:.4f}")

    # Save LoRA model (state dict only — can be merged later to full model)
    save_path = os.path.join(output_dir, "clip_lora.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[💾] LoRA-tuned weights saved to: {save_path}")


# ✅ CLI Wrapper
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train OpenCLIP with LoRA on Flickr30k CSV data")

    parser.add_argument("--csv", required=True, help="Path to CSV (image, text)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./openclip_lora_output")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha scaling")

    args = parser.parse_args()

    train_openclip(
        csv_path=args.csv,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        r=args.r,
        alpha=args.alpha
    )