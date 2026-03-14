import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering


DEFAULT_QUESTIONS = [
    "What is in the image?",
    "What room is this?",
    "What is the main object?",
    "Is there a bed?",
    "What color is the bed?",
    "Is there a window?",
    "Is this indoor or outdoor?",
    "How many beds are visible?",
    "What is on the bed?",
    "Describe the scene in one phrase.",
]


def pick_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor(model_dir: str, device: str, local_only: bool = True, use_fast: bool = True):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path.resolve()}")

    if local_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=local_only,
        use_fast=use_fast,  # 你现在看到的提示就是 fast processor 的变化提示
    )
    model = BlipForQuestionAnswering.from_pretrained(
        str(model_path),
        local_files_only=local_only,
    ).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def answer_one(model, processor, image: Image.Image, question: str, device: str, max_new_tokens: int = 20) -> str:
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out_ids[0], skip_special_tokens=True).strip()


def parse_args():
    p = argparse.ArgumentParser("Batch VQA (offline-first)")
    p.add_argument("--image", type=str, required=True, help="Image path")
    p.add_argument("--local_dir", type=str, default=str(Path(__file__).parent / "hf_tmp_blip_vqa"))
    p.add_argument("--out_json", type=str, default="results.json")
    p.add_argument("--out_md", type=str, default="results.md")
    p.add_argument("--online", type=int, default=0, help="0=offline, 1=online")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--use_fast", type=int, default=1, help="1=use fast processor, 0=use slow processor")
    p.add_argument("--questions", type=str, default="", help="Optional path to a txt file, one question per line")
    return p.parse_args()


def main():
    args = parse_args()

    device = pick_device(force_cpu=args.cpu)
    local_only = (args.online == 0)
    use_fast = (args.use_fast == 1)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path.resolve()}")

    print(f"device : {device}")
    print(f"image  : {img_path.resolve()}")
    print(f"model  : {Path(args.local_dir).resolve()}")
    print(f"offline: {local_only}")
    print(f"use_fast_processor: {use_fast}")

    model, processor = load_model_and_processor(args.local_dir, device, local_only=local_only, use_fast=use_fast)

    image = Image.open(str(img_path)).convert("RGB")

    # load questions
    if args.questions:
        qf = Path(args.questions)
        if not qf.exists():
            raise FileNotFoundError(f"questions file not found: {qf.resolve()}")
        questions = [line.strip() for line in qf.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    results = {
        "image": str(img_path.as_posix()),
        "model_dir": str(Path(args.local_dir).as_posix()),
        "device": device,
        "offline": local_only,
        "use_fast_processor": use_fast,
        "qa": [],
    }

    for i, q in enumerate(questions, 1):
        a = answer_one(model, processor, image, q, device=device, max_new_tokens=args.max_new_tokens)
        results["qa"].append({"q": q, "a": a})
        print(f"[{i:02d}] Q: {q}")
        print(f"     A: {a}")

    # save json
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ saved: {out_json.resolve()}")

    # save md (nice for README)
    out_md = Path(args.out_md)
    md_lines = []
    md_lines.append(f"# VQA Batch Results\n")
    md_lines.append(f"- Image: `{results['image']}`")
    md_lines.append(f"- Device: `{results['device']}`")
    md_lines.append(f"- Offline: `{results['offline']}`")
    md_lines.append("")
    md_lines.append("| # | Question | Answer |")
    md_lines.append("|---:|---|---|")
    for i, item in enumerate(results["qa"], 1):
        q = item["q"].replace("|", "\\|")
        a = item["a"].replace("|", "\\|")
        md_lines.append(f"| {i} | {q} | {a} |")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ saved: {out_md.resolve()}")


if __name__ == "__main__":
    main()
    