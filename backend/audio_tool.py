"""Audio via OpenRouter.

- voiceover : POST /api/v1/audio/speech (compatible OpenAI) -> octets MP3
- music     : non branché (le endpoint TTS ne fait que de la voix)

MOCK_MEDIA=1 -> renvoie un audio d'exemple sans appeler l'API.
"""
import os

import httpx

from media import save_media

BASE = "https://openrouter.ai/api/v1"
TTS_MODEL = os.getenv("TTS_MODEL", "")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
MOCK = os.getenv("MOCK_MEDIA", "0") == "1"
SAMPLE_AUDIO = "https://www.w3schools.com/html/horse.mp3"


def create_audio(kind: str, text: str, duration: int | None = None) -> dict:
    """Retourne {status: completed|unsupported|failed, audio_url, error?}."""
    if MOCK:
        return {"status": "completed", "audio_url": SAMPLE_AUDIO}
    if kind != "voiceover":
        return {"status": "unsupported", "audio_url": None}
    if not TTS_MODEL:
        return {"status": "failed", "audio_url": None, "error": "TTS_MODEL n'est pas défini dans .env"}

    try:
        r = httpx.post(
            f"{BASE}/audio/speech",
            headers={"Authorization": f"Bearer {os.getenv('API_KEY', '')}"},
            json={"model": TTS_MODEL, "input": text, "voice": TTS_VOICE, "response_format": "mp3"},
            timeout=120,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return {"status": "failed", "audio_url": None, "error": e.response.text[:300]}
    return {"status": "completed", "audio_url": save_media(r.content, "mp3")}
