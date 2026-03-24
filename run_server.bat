@echo off
echo Starting Optara Backend Server...
cd backend
..\.venv\Scripts\python.exe manage.py runserver
pause
