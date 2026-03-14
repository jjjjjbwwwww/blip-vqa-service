
@echo off
chcp 65001 >nul
setlocal

REM ====== 项目目录 ======
cd /d %~dp0

REM ====== 端口 ======
set PORT=8006

REM ====== 你的 python（建议用 torch 环境）=====
REM 如果你用的是 E:\anconda\envs_dirs\torch\python.exe，填这个：
set PY=E:\anconda\envs_dirs\torch\python.exe

REM ====== 本地离线模型目录 ======
set MODEL_DIR=hf_tmp_blip_vqa

echo [test6] Starting API on http://127.0.0.1:%PORT%
echo [test6] Using python: %PY%
echo [test6] Using model : %MODEL_DIR%
echo.

REM 启动 uvicorn（不 reload 更稳定）
%PY% -m uvicorn api:app --host 127.0.0.1 --port %PORT%

echo.
echo [test6] Server stopped.
pause