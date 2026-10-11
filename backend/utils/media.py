"""Stockage local des médias générés (servis par FastAPI sur /media).

Nécessaire car OpenRouter exige l'en-tête Authorization pour télécharger les fichiers :
le navigateur (st.video / st.audio) ne peut pas les lire directement.
"""
import os
import uuid
from pathlib import Path

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")


def public_url(filename: str) -> str:
    return f"{PUBLIC_BASE_URL}/media/{filename}"


def save_media(data: bytes, ext: str, name: str | None = None) -> str:
    filename = f"{name or uuid.uuid4().hex}.{ext}"
    (MEDIA_DIR / filename).write_bytes(data)
    return public_url(filename)
