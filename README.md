# Optara Facial Recognition MVP

This repository contains an **advanced MVP scaffold** for a production-style facial recognition platform using:

- Django + Django REST Framework (API)
- FAISS (vector matching)
- PyTorch + facenet-pytorch (embeddings)
- OpenCV (real-time video loop)
- Streamlit (monitoring dashboard)

## Architecture

```text
[ Webcam / Upload ]
        ↓
[ Face Detection ]
        ↓
[ Embedding Model ]
        ↓
[ FAISS Index ]
        ↓
[ Matching Engine ]
        ↓
[ Django API + Streamlit Dashboard ]
```

## Project Layout

```text
backend/
  manage.py
  config/
  users/
  recognition/
services/
  embeddings.py
  faiss_index.py
  pipeline.py
realtime/
  webcam_recognition.py
dashboard/
  app.py
```

## Quick Start

1. Create virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run migrations:

   ```bash
   python backend/manage.py migrate
   ```

3. Start API server:

   ```bash
   python backend/manage.py runserver
   ```

4. Run Streamlit dashboard:

   ```bash
   streamlit run dashboard/app.py
   ```

5. Run webcam recognition demo:

   ```bash
   python realtime/webcam_recognition.py
   ```

## API Endpoints

- `POST /api/register/` — Register user and process uploaded face images.
- `POST /api/recognize/` — Recognize a face from an uploaded image.
- `GET /api/logs/` — Fetch recent recognition logs.

## Notes

- This MVP stores FAISS index locally in `storage/faiss.index`.
- In production, add authentication, encryption, access controls, and model hardening.
