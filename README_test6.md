

# test6 — BLIP Multimodal VQA Chat (FastAPI + WebUI)

这是一个“多模态问答（VQA）+ 多轮对话记忆”的本地 Demo：
你上传一张图片，然后像聊天一样连续提问，系统会根据图片内容回答，并在左侧展示 Q1/Q2/Q3… 历史轮次。

## 功能 Features
- 单次问答：/vqa（One-shot）
- 多轮对话：/vqa_chat（Memory chat）
- 批量问答：/vqa_batch（一次问多个问题）
- WebUI：Apple Dark / iMessage 风格
  - 左侧历史轮次 Q1/Q2/Q3…（当前轮高亮，可点击跳转）
  - 系统角色提示（You are a multimodal assistant…）
  - 友好错误提示（未上传图片/问题为空/API异常）

## 项目结构
- app.py / api.py：FastAPI 服务入口
- vqa.py：命令行推理脚本（离线加载本地模型目录）
- ui.html：前端 Web UI（直接打开即可使用）
- hf_tmp_blip_vqa/：离线模型目录（snapshot_download 得到）
- images/：示例图片
- results.json / results.md：批量问答结果（可选）

## 架构图（文字版）
[Browser UI (ui.html)]
   ├─ 上传图片 + 输入问题（多轮）
   ├─ FormData: image + questions_json / question
   ↓
[FastAPI Server : http://127.0.0.1:8006]
   ├─ POST /vqa       -> 单次问答
   ├─ POST /vqa_chat  -> 多轮（questions_json = ["Q1","Q2",...])
   └─ POST /vqa_batch -> 批量问题
   ↓
[BLIP VQA Model (offline)]
   ├─ 从本地 model_dir 加载（避免重复下载）
   └─ 返回 answer / dialogue[]
   ↓
[UI]
   ├─ 显示聊天气泡
   └─ 左侧 History 列表同步显示 Q/A

## 快速开始 Quick Start

### 1) 启动后端（端口 8006）
```bash
# 进入项目目录
cd test6

# 启动 API（示例）
python app.py
# 或
python -m uvicorn app:app --host 127.0.0.1 --port 8006
2) 打开 WebUI
直接双击 ui.html 打开（或用 VSCode Live Server）。
在左侧确认 API Endpoint 为：
http://127.0.0.1:8006

3) API 示例（curl）
/vqa（单次问答）
curl -X POST "http://127.0.0.1:8006/vqa" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@images/sample.jpg;type=image/jpeg" \
  -F "question=What is in the image?" \
  -F "max_new_tokens=30"
/vqa_chat（多轮问答）
curl -X POST "http://127.0.0.1:8006/vqa_chat" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@images/sample.jpg;type=image/jpeg" \
  -F "questions_json=[\"What is in the image?\",\"Is there a bed?\",\"What color is the bed?\"]" \
  -F "prompt_style=qa" \
  -F "max_new_tokens=30"
UI 截图说明
主界面：顶部上传图片区域 + 聊天消息区 + 底部输入框

左侧栏：连接状态、模式选择、History（Q1/Q2/Q3…）

错误提示：以 iOS 风格 toast 显示（无需弹窗 alert）

images/docs_ui.png


---

### README.md（English）

```md
# test6 — BLIP Multimodal VQA Chat (FastAPI + WebUI)

A local multimodal demo: upload an image and ask questions in a chat-like flow.
The UI keeps Q1/Q2/Q3… history on the left (active turn highlighted, click-to-jump).

## Features
- One-shot VQA: `/vqa`
- Multi-turn memory chat: `/vqa_chat`
- Batch questions: `/vqa_batch`
- Apple Dark / iMessage-style WebUI
  - Left history Q1/Q2/Q3… (active highlight, jump to message)
  - System role tip (You are a multimodal assistant…)
  - Friendly errors (no image / empty question / API error)

## Structure
- `app.py` / `api.py`: FastAPI entry
- `vqa.py`: CLI inference (offline local model dir)
- `ui.html`: Web UI (open directly)
- `hf_tmp_blip_vqa/`: offline model snapshot directory
- `images/`: sample images

## Text Architecture Diagram
[Browser UI (ui.html)]
   ├─ upload image + ask questions (multi-turn)
   ├─ FormData: image + questions_json / question
   ↓
[FastAPI Server : http://127.0.0.1:8006]
   ├─ POST /vqa       -> one-shot
   ├─ POST /vqa_chat  -> memory chat (questions_json = ["Q1","Q2",...])
   └─ POST /vqa_batch -> batch
   ↓
[BLIP VQA Model (offline)]
   ├─ load from local model_dir (no repeated downloads)
   └─ return answer / dialogue[]
   ↓
[UI]
   ├─ chat bubbles
   └─ left History list synced with Q/A

## Quick Start

### 1) Start backend (port 8006)
```bash
cd test6
python app.py
# or
python -m uvicorn app:app --host 127.0.0.1 --port 8006
2) Open WebUI
Open ui.html in a browser (or VSCode Live Server).
Make sure API Endpoint is http://127.0.0.1:8006.

3) API Examples (curl)
/vqa (one-shot)
curl -X POST "http://127.0.0.1:8006/vqa" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@images/sample.jpg;type=image/jpeg" \
  -F "question=What is in the image?" \
  -F "max_new_tokens=30"
/vqa_chat (multi-turn)
curl -X POST "http://127.0.0.1:8006/vqa_chat" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@images/sample.jpg;type=image/jpeg" \
  -F "questions_json=[\"What is in the image?\",\"Is there a bed?\",\"What color is the bed?\"]" \
  -F "prompt_style=qa" \
  -F "max_new_tokens=30"
UI Screenshot Notes
Top: image upload stage

Middle: chat messages

Bottom: input dock

Left: status, settings, Q1/Q2/Q3 history

Errors are shown via iOS-style toast (no blocking alert)
