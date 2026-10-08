"""Outil d'appel à l'API vidéo.

Si VIDEO_API_URL est vide -> mode mock (pratique pour tester).
Sinon, adapte `create_video` / `get_video_status` au contrat de ton fournisseur
(Runway, Luma, Replicate, Veo, Kling...). Seuls ces deux fonctions changent.
"""
import os
from functools import lru_cache
import httpx

from media import MEDIA_DIR, public_url, save_media

BASE = "https://openrouter.ai/api/v1"
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "")
MOCK = os.getenv("MOCK_MEDIA", "0") == "1"
SAMPLE_VIDEO = "https://www.w3schools.com/html/mov_bbb.mp4"
VIDEO_RESOLUTION = os.getenv("VIDEO_RESOLUTION", "480p")
VIDEO_ASPECT_RATIO=os.getenv("VIDEO_ASPECT_RATIO","16:9")
def _headers() -> dict:
    return {"Authorization": f"Bearer {os.getenv('API_KEY')}", "Content-Type": "application/json"}


@lru_cache(maxsize=1)
def _video_models() -> dict:
    r = httpx.get(f"{BASE}/videos/models", timeout=30)
    r.raise_for_status()
    return {m["id"]: m for m in r.json().get("data", [])}
 
 
def _closest_duration(wanted: int) -> int | None:
    """Chaque modèle n'accepte que certaines durées (ex. 4/6/8 s) -> on prend la plus proche."""
    try:
        durations = _video_models().get(VIDEO_MODEL, {}).get("supported_durations") or []
    except httpx.HTTPError:
        return None
    return min(durations, key=lambda d: abs(d - wanted)) if durations else None
 
 
def create_video(prompt: str, duration: int = 5) -> dict:
    """Soumet le job. Retourne {job_id, status, video_url, error?}."""
    if MOCK:
        return {"job_id": "mock-job", "status": "completed", "video_url": SAMPLE_VIDEO}
 
    payload = {"model": VIDEO_MODEL, "prompt": prompt, "generate_audio": True, "resolution": VIDEO_RESOLUTION, "aspect_ratio": VIDEO_ASPECT_RATIO}
    if (d := _closest_duration(duration)) is not None:
        payload["duration"] = d
    try:
        r = httpx.post(f"{BASE}/videos", headers=_headers(), json=payload, timeout=60)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return {"job_id": None, "status": "failed", "video_url": None, "error": e.response.text[:300]}
    data = r.json()
    return {"job_id": data["id"], "status": data.get("status", "pending"), "video_url": None}
 
 
def _download(job_id: str, url: str) -> str:
    path = MEDIA_DIR / f"{job_id}.mp4"
    if path.exists():
        return public_url(path.name)
    headers = {"Authorization": _headers()["Authorization"]} if url.startswith(BASE) else {}
    r = httpx.get(url, headers=headers, follow_redirects=True, timeout=300)
    r.raise_for_status()
    return save_media(r.content, "mp4", name=job_id)
 
 
def get_video_status(job_id: str) -> dict:
    """Interroge le job ; si terminé, télécharge le MP4 et renvoie son URL locale."""
    if MOCK:
        return {"job_id": job_id, "status": "completed", "video_url": SAMPLE_VIDEO}
 
    r = httpx.get(f"{BASE}/videos/{job_id}", headers=_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    status = data.get("status", "unknown")
    video_url = None
    if status == "completed":
        urls = data.get("unsigned_urls") or [f"{BASE}/videos/{job_id}/content?index=0"]
        video_url = _download(job_id, urls[0])
    return {"job_id": job_id, "status": status, "video_url": video_url, "error": data.get("error")}