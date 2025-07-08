import open_clip
import torch
import os
from peft import LoraConfig, get_peft_model

def apply_peft_lora(model, r=8, alpha=16):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "proj", "out_proj", "c_proj"],  # adjust as needed
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    return get_peft_model(model, lora_config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LoRA .pth checkpoint to PEFT adapter directory")
    parser.add_argument("--pth", required=True, help="Path to .pth checkpoint (e.g., clip_lora_epoch60.pth)")
    parser.add_argument("--out_dir", required=True, help="Output directory for PEFT adapter")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    # Load base model
    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=args.device
    )

    # Inject PEFT LoRA
    model = apply_peft_lora(model, r=args.r, alpha=args.alpha)
    model = model.to(args.device)

    # Load .pth weights
    print(f"[INFO] Loading weights from {args.pth}")
    state_dict = torch.load(args.pth, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # Save as PEFT adapter
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    print(f"[INFO] Saved PEFT adapter to {args.out_dir}")