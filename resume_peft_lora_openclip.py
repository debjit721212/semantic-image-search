import open_clip
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from peft import PeftModel

from flickr_csv_dataset_openclip import FlickrOpenCLIPDataset

def unfreeze_lora_parameters(model):
    """
    Set requires_grad=True for all LoRA adapter weights after loading.
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

def resume_peft_lora_openclip(
    csv_path,
    adapter_dir,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    batch_size=8,
    epochs=5,
    lr=1e-4,
    output_dir="./openclip_lora_peft_adapter_custom"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load base model + preprocessing transforms
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # --- Load PEFT LoRA adapter ---
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.train()

    # --- Unfreeze LoRA adapter weights ---
    unfreeze_lora_parameters(model)

    # Check trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    if not trainable_params:
        print("[ERROR] No trainable parameters found! Check your adapter_dir and model compatibility.")
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        return

    # Prepare DataLoader
    dataset = FlickrOpenCLIPDataset(csv_path, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = optim.AdamW(trainable_params, lr=lr)

    os.makedirs(output_dir, exist_ok=True)

    # 🚀 Training Loop
    for epoch in range(epochs):
        print(f"\n📚 Fine-tuning Epoch [{epoch + 1}/{epochs}]")
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training")

        for batch in pbar:
            images = batch["image"].to(device)
            texts = batch["text"]
            text_tokens = tokenizer(texts).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=1, keepdim=True).clone()
            text_features = text_features / text_features.norm(dim=1, keepdim=True).clone()

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
    print(f"[💾] Fine-tuned LoRA adapter saved to: {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume PEFT LoRA training on OpenCLIP with custom data")
    parser.add_argument("--csv", required=True, help="Path to custom CSV (image, text)")
    parser.add_argument("--adapter_dir", required=True, help="Path to previous PEFT adapter directory (from Flickr30k training)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./openclip_lora_peft_adapter_custom")

    args = parser.parse_args()

    resume_peft_lora_openclip(
        csv_path=args.csv,
        adapter_dir=args.adapter_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir
    )