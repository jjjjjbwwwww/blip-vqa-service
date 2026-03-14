
import io
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from transformers import AutoProcessor, BlipForQuestionAnswering

# =========================
# Config (offline-first)
# =========================
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "hf_tmp_blip_vqa"

# Hard offline: no network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# =========================
# Globals
# =========================
app = FastAPI(title="BLIP VQA API (offline-first)", version="1.1.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
model: Optional[BlipForQuestionAnswering] = None
processor: Optional[AutoProcessor] = None


def load_model_once(use_fast: bool = True) -> None:
    """Load processor/model only once at startup (or first request)."""
    global model, processor

    if model is not None and processor is not None:
        return

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR.resolve()}")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_fast=use_fast,  # will show 'fast processor' notice sometimes; safe to ignore
    )
    model = BlipForQuestionAnswering.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    ).to(device)
    model.eval()


@torch.no_grad()
def answer_one(pil: Image.Image, question: str, max_new_tokens: int = 20) -> str:
    assert processor is not None and model is not None
    inputs = processor(images=pil, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    ans = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return ans


def parse_questions(questions_json: Optional[str], questions_text: Optional[str]) -> List[str]:
    """
    Accept either:
    - questions_json: a JSON string (e.g. ["Q1","Q2"])
    - questions_text: multi-line text, each line is a question
    """
    if questions_json and questions_json.strip():
        try:
            obj = json.loads(questions_json)
            if isinstance(obj, list):
                qs = [str(x).strip() for x in obj if str(x).strip()]
                if qs:
                    return qs
            raise ValueError("questions_json must be a JSON list of strings, e.g. [\"Q1\",\"Q2\"]")
        except Exception as e:
            raise ValueError(f"Invalid questions_json: {e}")

    if questions_text and questions_text.strip():
        qs = [line.strip() for line in questions_text.splitlines() if line.strip()]
        if qs:
            return qs

    # Default fallback: one common question
    return ["What is in the image?"]


@app.on_event("startup")
def on_startup():
    # Preload model once
    load_model_once(use_fast=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "device": device,
        "model_dir": str(MODEL_DIR.resolve()),
        "offline": True,
    }


@app.post("/vqa")
async def vqa(
    image: UploadFile = File(...),
    question: str = Form(...),
    max_new_tokens: int = Form(20),
):
    """
    Single-question VQA.
    """
    try:
        load_model_once(use_fast=True)
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        ans = answer_one(pil, question, max_new_tokens=max_new_tokens)
        return {
            "device": device,
            "model_dir": str(MODEL_DIR.resolve()),
            "question": question,
            "answer": ans,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/vqa_batch")
async def vqa_batch(
    image: UploadFile = File(...),
    # Provide ONE of the following:
    questions_json: Optional[str] = Form(None),  # e.g. ["Q1","Q2"]
    questions_text: Optional[str] = Form(None),  # multi-line, one question per line
    max_new_tokens: int = Form(20),
):
    """
    Batch VQA: one image, multiple questions.
    - Use questions_json (JSON list) OR questions_text (one per line).
    """
    try:
        load_model_once(use_fast=True)
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        questions = parse_questions(questions_json, questions_text)
        answers = []
        for q in questions:
            a = answer_one(pil, q, max_new_tokens=max_new_tokens)
            answers.append({"question": q, "answer": a})

        return {
            "device": device,
            "model_dir": str(MODEL_DIR.resolve()),
            "num_questions": len(questions),
            "results": answers,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})