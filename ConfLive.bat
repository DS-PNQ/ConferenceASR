@echo off
rem ConfLive one-click: first run installs everything + downloads models, later runs just launch.
rem Needs: Windows, Python 3.11 (3.12 works but without DeepFilterNet denoise), Node.js 18+, internet on first run.
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

where node >nul 2>nul || (echo [!] Node.js not found. Install the LTS from https://nodejs.org then run this again. & pause & exit /b 1)

if not exist "%PY%" (
  echo [1/4] Creating Python venv...
  py -3.11 -m venv .venv 2>nul || (
    echo [!] Python 3.11 not found - using default Python. DeepFilterNet denoise needs 3.11:
    echo     https://www.python.org/downloads/release/python-3119/
    py -3 -m venv .venv 2>nul || python -m venv .venv || (echo [!] No Python found. & pause & exit /b 1)
  )
)

if not exist ".venv\.setup-done" (
  echo [2/4] Installing Python packages ^(several GB, one time^)...
  "%PY%" -m pip install --upgrade pip
  set "TORCH_IDX=https://download.pytorch.org/whl/cpu"
  where nvidia-smi >nul 2>nul && set "TORCH_IDX=https://download.pytorch.org/whl/cu126"
  call "%PY%" -m pip install torch torchaudio --index-url %%TORCH_IDX%% || goto :fail
  "%PY%" -m pip install -r requirements.txt || goto :fail
  "%PY%" -m pip install --no-deps rapidocr_onnxruntime || goto :fail
  "%PY%" -m pip install deepfilternet || echo [!] deepfilternet unavailable on this Python - denoise stays off.
  echo ok> ".venv\.setup-done"
)

rem Nemotron diarizer sidecar: transformers 5.19 in its own folder (backend keeps 4.57 for Hy-MT)
if not exist "..\diardeps\transformers" (
  "%PY%" -m pip install --no-deps --target ..\diardeps "transformers==5.19.0" "huggingface_hub>=1.31,<2" "tokenizers>=0.23.1,<0.24" || echo [!] diardeps install failed - diarizer falls back to NeMo Titanet.
)

if not exist "models\.downloaded" (
  echo [3/4] Downloading models ^(ASR, Hy-MT2, DeepFilterNet, Nemotron diarizer, Titanet^)...
  "%PY%" scripts\download_models.py || goto :fail
  echo ok> "models\.downloaded"
)

echo [4/4] Building and launching the app...
pushd electron
if not exist node_modules call npm install || goto :fail
if not exist ui\node_modules call npm --prefix ui install || goto :fail
call npm run build || goto :fail
call npx electron .
popd
exit /b 0

:fail
echo [!] Setup failed - see the error above. Fix it and run ConfLive.bat again (finished steps are skipped).
pause
exit /b 1
