# Optara (Desktop + API + AI)

Optara uses three layers:

1. **Backend (Django API only)**
   - Stores users, encrypted embeddings, recognition logs
   - Runs at `http://127.0.0.1:8000/api/`
2. **Desktop App (CustomTkinter)**
   - Live camera feed
   - Face mapping workflow (capture -> prompt for name -> register)
3. **AI Engine (service layer)**
   - Face detection + embedding extraction + FAISS search

## Face Mapping Flow (requested behavior)

1. Start camera in desktop app.
2. Click **Map Face**.
3. App captures multiple face crops (12 samples by default).
4. After capture completes, app prompts for a **name**.
5. Name + face images are sent to `POST /api/register/` and saved.

## Encryption

Embeddings are encrypted at rest in the database using Fernet (from `cryptography`).

- Set a strong key in your environment:

```bash
export OPTARA_ENCRYPTION_KEY="replace-with-a-long-random-secret"
```

If not set, a development default is used (not safe for production).

## Common Error: `WinError 10061`

Connection refused usually means backend is not running.

Start backend first:

```bash
cd backend
python manage.py runserver
```

Then run desktop app:

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
.venv\Scripts\activate
pip install -r requirements.txt
```
