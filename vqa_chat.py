
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_prompt(history, new_question, style="qa"):
    """
    history: list of {"q":..., "a":...}
    style:
      - "qa":  Question/Answer 形式（推荐）
      - "chat": Chat 形式（有时更自然，但不一定更准）
    """
    if not history:
        return new_question.strip()

    if style == "chat":
      
        lines = []
        for turn in history:
            lines.append(f"User: {turn['q']}")
            lines.append(f"Assistant: {turn['a']}")
        lines.append(f"User: {new_question}")
        return "\n".join(lines)

    # 默认 qa
    prompt = ""
    for turn in history:
        prompt += f"Question: {turn['q']}\nAnswer: {turn['a']}\n\n"
    prompt += f"Question: {new_question}\nAnswer:"
    return prompt


@torch.inference_mode()
def answer_one(model, processor, image, prompt, device, max_new_tokens=20):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    ans = processor.decode(out[0], skip_special_tokens=True).strip()
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument("--model_dir", default="hf_tmp_blip_vqa", help="Local HF snapshot dir")
    ap.add_argument("--offline", action="store_true", help="Force offline local loading")
    ap.add_argument("--questions_json", default="", help='JSON list like ["q1","q2"]')
    ap.add_argument("--questions_text", default="", help="Lines separated by \\n")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--prompt_style", choices=["qa", "chat"], default="qa")
    ap.add_argument("--save", default="chat_results.json", help="Save dialogue json")
    args = ap.parse_args()

    device = pick_device()
    img_path = Path(args.image).resolve()
    model_dir = Path(args.model_dir).resolve()

    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    # 解析问题
    questions = []
    if args.questions_json.strip():
        questions = json.loads(args.questions_json)
    elif args.questions_text.strip():
        questions = [x.strip() for x in args.questions_text.splitlines() if x.strip()]
    else:
        # 默认演示
        questions = [
            "What is in the image?",
            "What room is this?",
            "Is there a bed?",
            "Is there a window?",
        ]

    # 加载模型（离线）
    local_files_only = True if args.offline else False
    processor = BlipProcessor.from_pretrained(str(model_dir), local_files_only=local_files_only)
    model = BlipForQuestionAnswering.from_pretrained(str(model_dir), local_files_only=local_files_only).to(device)
    model.eval()

    image = Image.open(img_path).convert("RGB")

    print(f"device : {device}")
    print(f"image  : {img_path}")
    print(f"model  : {model_dir}")
    print(f"offline: {local_files_only}")
    print(f"style  : {args.prompt_style}")
    print()

    dialogue = []
    for i, q in enumerate(questions, 1):
        prompt = build_prompt(dialogue, q, style=args.prompt_style)
        a = answer_one(model, processor, image, prompt, device, max_new_tokens=args.max_new_tokens)
        dialogue.append({"q": q, "a": a})
        print(f"[{i:02d}] Q: {q}")
        print(f"     A: {a}")
        print()

    out_path = Path(args.save).resolve()
    out_path.write_text(json.dumps({"device": device, "dialogue": dialogue}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ saved: {out_path}")


if __name__ == "__main__":
    main()
