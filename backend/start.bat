@echo off
echo Starting Pneumonia Detection API...
echo.

REM Activate venv if it exists one level up
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

REM Install backend deps if needed
pip install -r requirements.txt --quiet

echo.
echo API running at http://localhost:8000
echo Docs at     http://localhost:8000/docs
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
