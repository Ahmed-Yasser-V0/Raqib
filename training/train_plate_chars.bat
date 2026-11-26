@echo off
cd /d "%~dp0\.."
echo Retraining the plate reader (new letters such as ك)...
".venv\Scripts\python.exe" training\train_plate_chars.py
pause
