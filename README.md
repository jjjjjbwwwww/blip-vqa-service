

# Vision Tools Server (BLIP)

A multimodal vision-language service that provides image captioning and visual question answering (VQA) APIs based on the BLIP model.

This service is designed to be used as a **tool server** for AI agents.

---

# Features

- Image Caption Generation
- Visual Question Answering (VQA)
- Batch VQA inference
- REST API service using FastAPI
- Designed for integration with AI agents

---

# Architecture

User / Agent
      │
      ▼
FastAPI Service
      │
      ▼
BLIP Vision-Language Model
      │
      ▼
Caption / VQA Result

---

# Project Structure

vision-tools-blip
│
├── api.py            # FastAPI endpoints
├── app.py            # server startup
├── caption.py        # image captioning
├── vqa.py            # VQA inference
├── vqa_batch.py      # batch VQA
│
├── images            # example images
│
├── ui.html           # simple UI demo
│
├── requirements.txt
└── README.md
Installation
pip install -r requirements.txt
Start Server
python app.py

Server will start at:

http://localhost:8006
API Example
Caption
POST /caption

Example:

curl -X POST http://localhost:8006/caption \
  -F "image=@sample.jpg"
VQA
POST /vqa

Example:

curl -X POST http://localhost:8006/vqa \
  -F "image=@sample.jpg" \
  -F "question=What is in the image?"
Tech Stack

Python

FastAPI

HuggingFace Transformers

BLIP Vision Language Model

Use Case

This service is designed to be used as a multimodal tool for AI agents.

Example integration:

AI Agent
   │
   ▼
Vision Tool API
   │
   ▼
BLIP Model
