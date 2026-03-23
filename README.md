# Optara (Rewritten Architecture)

Optara now uses three clean layers:

1. **Backend (Django API only)**
   - Pure REST API (no web UI)
   - Handles users, embeddings, recognition events, and logs
   - Base URL: `http://127.0.0.1:8000/api/`

2. **Desktop App (CustomTkinter)**
   - Replaces Streamlit
   - Shows live camera feed
   - Start/stop camera controls
   - Periodically fetches recognition logs from backend

3. **AI Engine (service layer in backend)**
   - OpenCV capture + face processing
   - FaceNet embeddings (`facenet-pytorch`)
   - FAISS vector matching

## Common Error: `WinError 10061`

`[WinError 10061] No connection could be made because the target machine actively refused it` usually means:

- Django server is not running
- Wrong host/port
- Backend crashed during startup

## Correct Startup Order

### 1) Start backend

```bash
cd backend
python manage.py runserver
```

Expect:

```text
Starting development server at http://127.0.0.1:8000/
```

Check logs endpoint:

- `http://127.0.0.1:8000/api/logs/`

### 2) Start desktop app

```bash
python desktop_app/main.py
```

## API Endpoints

- `POST /api/register/`
- `POST /api/recognize/`
- `GET /api/logs/`

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project Layout

```text
backend/
  config/
  users/
  recognition/
  services/
desktop_app/
  main.py
realtime/
  webcam_recognition.py
storage/
```

## Notes

- FAISS index persists to `storage/faiss.index`.
- For production: add auth, encryption, consent flows, and role-based access controls.
