@echo off
echo Running migrations...
cd backend
..\.venv\Scripts\python.exe manage.py migrate
pause
