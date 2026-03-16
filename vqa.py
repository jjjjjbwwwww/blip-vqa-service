

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from transformers import AutoProcessor, BlipForQuestionAnswering


def pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor(model_dir: str, device: str, local_only: bool = True):
    """
    model_dir: 本地模型目录（包含 config.json / model.safetensors 等）
    local_only=True -> 完全离线，不会联网下载
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path.resolve()}\n"
            f"请确认你已下载完成，并把 --local_dir 指向正确目录。"
        )

    # 让 transformers 离线：避免任何联网请求
   
    if local_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=local_only
    )
    model = BlipForQuestionAnswering.from_pretrained(
        str(model_path), local_files_only=local_only
    )
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def run_vqa(model, processor, image_path: str, question: str, device: str, max_new_tokens: int = 20) -> str:
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {img_p.resolve()}")

    image = Image.open(str(img_p)).convert("RGB")

    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return answer


def parse_args():
    p = argparse.ArgumentParser(description="BLIP VQA (offline-first)")
    p.add_argument("--image", type=str, required=True, help="Image path, e.g. images/sample.jpg")
    p.add_argument("--question", type=str, required=True, help='Question text, e.g. "What is in the image?"')


    p.add_argument(
        "--local_dir",
        type=str,
        default=str(Path(__file__).parent / "hf_tmp_blip_vqa"),
        help="Local model directory (downloaded snapshot). Default: ./hf_tmp_blip_vqa",
    )

    # 默认离线
    p.add_argument(
        "--online",
        type=int,
        default=0,
        help="Allow online download/check (0=offline/local only, 1=online). Default 0",
    )

    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--max_new_tokens", type=int, default=20, help="Generation length. Default 20")
    return p.parse_args()


def main():
    args = parse_args()

    device = pick_device(force_cpu=args.cpu)
    local_only = (args.online == 0)

    print(f"device : {device}")
    print(f"image  : {Path(args.image).resolve()}")
    print(f"model  : {Path(args.local_dir).resolve()}")
    print(f"offline: {local_only}")

    try:
        model, processor = load_model_and_processor(
            model_dir=args.local_dir,
            device=device,
            local_only=local_only
        )
    except Exception as e:
        print("\n❌ Failed to load model locally.")
        print("原因通常是：local_dir 指错 / 目录里缺少 model.safetensors 等文件。\n")
        raise

    answer = run_vqa(
        model=model,
        processor=processor,
        image_path=args.image,
        question=args.question,
        device=device,
        max_new_tokens=args.max_new_tokens
    )

    print(f"\nQ: {args.question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    main()
