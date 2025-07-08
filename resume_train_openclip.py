# resume_train_openclip.py

import open_clip
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from openclip_lora_module import LoRALinear
from flickr_csv_dataset_openclip import FlickrOpenCLIPDataset


def apply_lora_to_model(model, r=8, alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(key in name for key in ["q_proj", "v_proj", "proj"]):
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], LoRALinear(module, r=r, alpha=alpha))
            print(f"[LoRA Injected] - {name}")
    return model


def resume_training(
    csv_path,
    checkpoint_path,
    start_epoch=5,
    total_epochs=10,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    batch_size=8,
    lr=1e-4,
    output_dir="./openclip_lora_output",
    r=8,
    alpha=16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load base model + tokenize + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # Inject LoRA and load saved weights
    model = apply_lora_to_model(model, r=r, alpha=alpha)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.to(device)
    model.train()

    # Dataset & Loader
    dataset = FlickrOpenCLIPDataset(csv_path, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer — only trainable (LoRA) params
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    os.makedirs(output_dir, exist_ok=True)

    # 🔁 Continue training
    for epoch in range(start_epoch, total_epochs):
        print(f"\n📚 Resuming at Epoch [{epoch + 1}/{total_epochs}]")
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training")

        for batch in pbar:
            images = batch["image"].to(device)
            texts = batch["text"]
            text_tokens = tokenizer(texts).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            # Normalize for contrastive loss
            image_features = image_features / image_features.norm(dim=1, keepdim=True).clone()
            text_features = text_features / text_features.norm(dim=1, keepdim=True).clone()

            logits = image_features @ text_features.T
            labels = torch.arange(len(images), device=device)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch + 1}] ✅ Completed — Avg Loss: {avg_loss:.4f}")

        # Save after every epoch
        save_path = os.path.join(output_dir, f"clip_lora_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[💾] Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume LoRA training on OpenCLIP")

    parser.add_argument("--csv", required=True, help="Path to flickr_cleaned.csv")
    parser.add_argument("--checkpoint_path", required=True, help="Path to previous .pth checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=5, help="Start epoch (last + 1)")
    parser.add_argument("--total_epochs", type=int, default=10, help="Total target epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./openclip_lora_output")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)

    args = parser.parse_args()

    resume_training(
        csv_path=args.csv,
        checkpoint_path=args.checkpoint_path,
        start_epoch=args.start_epoch,
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        r=args.r,
        alpha=args.alpha
    )