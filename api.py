

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from fastapi.responses import HTMLResponse
from pathlib import Path
# ----------------------------
# config
# ----------------------------
DEFAULT_MODEL_DIR = "hf_tmp_blip_vqa"   # 本地离线模型目录
DEFAULT_MAX_NEW_TOKENS = 20

app = FastAPI(
    title="test6 - BLIP VQA API (Offline)",
    description="Single-image VQA + Batch VQA + Multi-turn Chat VQA (same image).",
    version="1.1.0",
)

@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = Path(__file__).parent / "ui.html"
    return html_path.read_text(encoding="utf-8")


# ----------------------------
# helpers
# ----------------------------
def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image_from_upload(file: UploadFile) -> Image.Image:
    img = Image.open(file.file).convert("RGB")
    return img


def parse_questions(questions_json: str, questions_text: str) -> List[str]:
    """
    优先解析 questions_json，否则解析 questions_text
    """
    if questions_json and questions_json.strip():
        try:
            data = json.loads(questions_json)
            if not isinstance(data, list):
                raise ValueError("questions_json must be a JSON array (list).")
            qs = [str(x).strip() for x in data if str(x).strip()]
            return qs
        except Exception as e:
            raise ValueError(f"Invalid questions_json: {e}")

    if questions_text and questions_text.strip():
        qs = [x.strip() for x in questions_text.splitlines() if x.strip()]
        return qs

    return []


def build_prompt(history: List[dict], new_question: str, style: str = "qa") -> str:
    """
    history: [{"q":..., "a":...}, ...]
    style:
      - qa  : Question/Answer 
      - chat: User/Assistant 对话格式（可选）
    """
    new_question = new_question.strip()
    if not history:
        return new_question

    if style == "chat":
        lines = []
        for t in history:
            lines.append(f"User: {t['q']}")
            lines.append(f"Assistant: {t['a']}")
        lines.append(f"User: {new_question}")
        return "\n".join(lines)

    # default qa
    prompt = ""
    for t in history:
        prompt += f"Question: {t['q']}\nAnswer: {t['a']}\n\n"
    prompt += f"Question: {new_question}\nAnswer:"
    return prompt


@torch.inference_mode()
def answer_one(model, processor, image: Image.Image, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    ans = processor.decode(out[0], skip_special_tokens=True).strip()
    return ans


# ----------------------------
# global model (loaded once)
# ----------------------------
DEVICE = pick_device()
MODEL_DIR = Path(DEFAULT_MODEL_DIR).resolve()

# 默认：离线加载
LOCAL_FILES_ONLY = True

processor = None
model = None


def ensure_model_loaded(model_dir: Optional[str] = None, offline: Optional[bool] = None):
    """
    - offline: True => local_files_only=True；False => 允许联网下载
    """
    global processor, model, MODEL_DIR, LOCAL_FILES_ONLY

    if model_dir:
        MODEL_DIR = Path(model_dir).resolve()
    if offline is not None:
        LOCAL_FILES_ONLY = bool(offline)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    if processor is None or model is None:
        p = BlipProcessor.from_pretrained(str(MODEL_DIR), local_files_only=LOCAL_FILES_ONLY)
        m = BlipForQuestionAnswering.from_pretrained(str(MODEL_DIR), local_files_only=LOCAL_FILES_ONLY).to(DEVICE)
        m.eval()
        processor, model = p, m


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_dir": str(MODEL_DIR),
        "offline": LOCAL_FILES_ONLY,
    }


# ----------------------------
# endpoints
# ----------------------------
@app.post("/vqa")
def vqa(
    image: UploadFile = File(...),
    question: str = Form(...),

    # 可选：是否离线
    model_dir: Optional[str] = Form(None),
    offline: Optional[bool] = Form(None),

    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
):
    """
    单张图片 + 单问题
    """
    try:
        ensure_model_loaded(model_dir=model_dir, offline=offline)
        img = load_image_from_upload(image)
        ans = answer_one(model, processor, img, question, DEVICE, max_new_tokens)
        return {
            "device": DEVICE,
            "model_dir": str(MODEL_DIR),
            "question": question,
            "answer": ans,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/vqa_batch")
def vqa_batch(
    image: UploadFile = File(...),
    questions_json: str = Form(""),
    questions_text: str = Form(""),

    model_dir: Optional[str] = Form(None),
    offline: Optional[bool] = Form(None),

    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
):
    """
    单张图片 + 多问题
    """
    try:
        ensure_model_loaded(model_dir=model_dir, offline=offline)
        qs = parse_questions(questions_json, questions_text)
        if not qs:
            raise ValueError("No questions provided. Use questions_json or questions_text.")

        img = load_image_from_upload(image)
        results = []
        for q in qs:
            a = answer_one(model, processor, img, q, DEVICE, max_new_tokens)
            results.append({"question": q, "answer": a})

        return {
            "device": DEVICE,
            "model_dir": str(MODEL_DIR),
            "num_questions": len(qs),
            "results": results,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/vqa_chat")
def vqa_chat(
    image: UploadFile = File(...),
    questions_json: str = Form(""),
    questions_text: str = Form(""),

    prompt_style: str = Form("qa"),  # qa / chat
    model_dir: Optional[str] = Form(None),
    offline: Optional[bool] = Form(None),

    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
):
    """
    单张图片 + 多轮对话
    """
    try:
        ensure_model_loaded(model_dir=model_dir, offline=offline)
        qs = parse_questions(questions_json, questions_text)
        if not qs:
            raise ValueError("No questions provided. Use questions_json or questions_text.")
        if prompt_style not in ["qa", "chat"]:
            raise ValueError("prompt_style must be 'qa' or 'chat'.")

        img = load_image_from_upload(image)

        dialogue = []
        for q in qs:
            prompt = build_prompt(dialogue, q, style=prompt_style)
            a = answer_one(model, processor, img, prompt, DEVICE, max_new_tokens)
            dialogue.append({"q": q, "a": a})

        return {
            "device": DEVICE,
            "model_dir": str(MODEL_DIR),
            "offline": LOCAL_FILES_ONLY,
            "prompt_style": prompt_style,
            "num_questions": len(qs),
            "dialogue": dialogue,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ----------------------------
# optional: CLI for local run
# ----------------------------
def main():
    """
    用法：
      python api.py --port 8006 --model_dir hf_tmp_blip_vqa --offline
    """
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--offline", action="store_true", help="Force local_files_only=True")
    args = ap.parse_args()

    # 覆盖全局设置
    global MODEL_DIR, LOCAL_FILES_ONLY
    MODEL_DIR = Path(args.model_dir).resolve()
    LOCAL_FILES_ONLY = bool(args.offline)

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
