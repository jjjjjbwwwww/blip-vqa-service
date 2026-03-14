
@echo off
chcp 65001 >nul
setlocal

set PORT=8006

echo [test6] Trying to stop process on port %PORT% ...

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
  echo Found PID: %%a
  taskkill /PID %%a /F
)

echo Done.
pause