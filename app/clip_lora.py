# clip_lora.py
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model_lora(
    base_model="openai/clip-vit-base-patch32",
    lora_weights_path=None,
    enable_lora=False
):
    """
    Loads CLIP model with or without LoRA adapters.
    """
    print("[INFO] Loading CLIP model...", end=" ")
    model = CLIPModel.from_pretrained(base_model).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(base_model)
    print("✅")

    if enable_lora:
        print(f"[INFO] Injecting LoRA weights from: {lora_weights_path}")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )

        model = get_peft_model(model, lora_config)

        if lora_weights_path and os.path.exists(lora_weights_path):
            model.load_adapter(lora_weights_path, adapter_name="default")
            print("✅ LoRA weights loaded")
        else:
            print("[WARN] LoRA path not found. Running without fine-tuned weights.")

    return model.eval(), processor