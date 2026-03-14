import argparse
from pathlib import Path
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="path to image")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")

    device = torch.device(args.device)
    print("device:", device)
    print("image :", img_path.resolve())

    image = Image.open(img_path).convert("RGB")

    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)

    caption = processor.decode(out[0], skip_special_tokens=True)
    print("\ncaption:", caption)


if __name__ == "__main__":
    main()