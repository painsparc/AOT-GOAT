@echo off
REM One-step setup + test + run on Windows.
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if not exist .env copy .env.example .env >nul
python test_pipeline.py
echo Open http://127.0.0.1:8000
python app.py
