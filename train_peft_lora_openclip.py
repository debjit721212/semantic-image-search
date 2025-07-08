import open_clip
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from flickr_csv_dataset_openclip import FlickrOpenCLIPDataset

def train_peft_lora_openclip(
    csv_path,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    batch_size=8,
    epochs=5,
    lr=1e-4,
    output_dir="./openclip_lora_peft_adapter",
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

    # --- PEFT LoRA injection ---
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "proj", "out_proj", "c_proj"],  # adjust as needed
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model = get_peft_model(model, lora_config)
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
            texts = batch["text"]
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

    # --- Save in PEFT format ---
    model.save_pretrained(output_dir)
    print(f"[💾] LoRA adapter saved to: {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train OpenCLIP with PEFT LoRA and save as adapter directory")
    parser.add_argument("--csv", required=True, help="Path to CSV (image, text)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./openclip_lora_peft_adapter")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha scaling")

    args = parser.parse_args()

    train_peft_lora_openclip(
        csv_path=args.csv,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        r=args.r,
        alpha=args.alpha
    )