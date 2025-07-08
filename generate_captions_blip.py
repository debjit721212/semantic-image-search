import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def load_blip_model(device="cuda"):
    print("[INFO] Loading BLIP image captioning model...")
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto",torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    return processor, model


# ✅ Use task-specific prompt/question
def caption_images(img_dir, out_csv_path, prompt="What is happening in this surveillance image?", device="cuda"):
    processor, model = load_blip_model(device)
    caption_data = []

    print(f"[INFO] Generating captions for images in: {img_dir}")
    files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for file_name in tqdm(files):
        img_path = os.path.join(img_dir, file_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[⚠️] Skipping {file_name}: {e}")
            continue

        # BLIP expects (image, text prompt) as inputs
        inputs = processor(image, prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)

        clean_caption = caption.strip().capitalize()
        caption_data.append({
            "image": img_path,
            "text": clean_caption
        })

    if not caption_data:
        print("❌ No valid images processed.")
        return

    df = pd.DataFrame(caption_data)
    df.to_csv(out_csv_path, index=False)
    print(f"\n✅ Finished generating captions for {len(df)} images.")
    print(f"[💾] Captions saved to: {out_csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate scene-level captions using BLIP")

    parser.add_argument("--img_dir", required=True, help="Folder containing input images")
    parser.add_argument("--out_csv", default="generated_custom_data.csv", help="Path to save results CSV")
    parser.add_argument("--prompt", default="What is happening in this surveillance image?", help="Prompt/question for BLIP")
    parser.add_argument("--device", default="cuda", help="Device to run BLIP model on (cuda or cpu)")

    args = parser.parse_args()

    caption_images(
        img_dir=args.img_dir,
        out_csv_path=args.out_csv,
        prompt=args.prompt,
        device=args.device
    )