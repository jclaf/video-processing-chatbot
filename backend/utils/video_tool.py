"""Outil d'appel à l'API vidéo.

Si VIDEO_API_URL est vide -> mode mock (pratique pour tester).
Sinon, adapte `create_video` / `get_video_status` au contrat de ton fournisseur
(Runway, Luma, Replicate, Veo, Kling...). Seuls ces deux fonctions changent.
"""
import os
import time
from functools import lru_cache
import httpx

from utils.media import MEDIA_DIR, public_url, save_media

BASE = "https://openrouter.ai/api/v1"
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "")
MOCK = os.getenv("MOCK_MEDIA", "0") == 1

SAMPLE_VIDEO = "https://www.w3schools.com/html/mov_bbb.mp4"
VIDEO_RESOLUTION = os.getenv("VIDEO_RESOLUTION", "480p")
VIDEO_ASPECT_RATIO=os.getenv("VIDEO_ASPECT_RATIO","16:9")
VIDEO_GENERATE_AUDIO = os.getenv("VIDEO_GENERATE_AUDIO", "")  # 0 ou 1

def _headers() -> dict:
    return {"Authorization": f"Bearer {os.getenv('API_KEY')}", "Content-Type": "application/json"}


@lru_cache(maxsize=1)
def _video_models() -> dict:
    r = httpx.get(f"{BASE}/videos/models", timeout=30)
    r.raise_for_status()
    return {m["id"]: m for m in r.json().get("data", [])}

def _model_info() -> dict:
    """Capacités du modèle choisi ({} si la liste est indisponible -> aucune validation locale)."""
    try:
        return _video_models().get(VIDEO_MODEL, {})
    except httpx.HTTPError:
        return {}
  
 
def _closest_duration(wanted: int, info: dict) -> int | None:
    """Chaque modèle n'accepte que certaines durées -> on prend la plus proche."""
    durations = info.get("supported_durations") or []
    return min(durations, key=lambda d: abs(d - wanted)) if durations else None
 
 
def _failed(msg: str) -> dict:
    return {"job_id": None, "status": "failed", "video_url": None, "error": msg}
 

def create_video(prompt: str, duration: int = 5, images: list[str] | None = None,
                 image_mode: str = "first_frame") -> dict:
    """Soumet le job. Retourne {job_id, status, video_url, error?}.
 
    images (optionnel) : URLs https ou data URLs base64.
    image_mode : "first_frame" (1re image = début ; 2e = fin seulement si le modèle le supporte)
                 ou "reference" (les images guident le style / le contenu).
    """
    if MOCK:
        return {"job_id": "mock-job", "status": "completed", "video_url": SAMPLE_VIDEO}
    
    info = _model_info()
    payload = {"model": VIDEO_MODEL, "prompt": prompt, "generate_audio": True}
    if (d := _closest_duration(duration, info)) is not None:
        payload["duration"] = d
        
    for key, value, field in (
        ("resolution", VIDEO_RESOLUTION, "supported_resolutions"),
        ("aspect_ratio", VIDEO_ASPECT_RATIO, "supported_aspect_ratios")):
        if value:
            allowed = info.get(field) or []
            if allowed and value not in allowed:
                return _failed(f"{key}='{value}' non supporté par {VIDEO_MODEL}. Valeurs valides : {', '.join(allowed)}")
            payload[key] = value
 
    if VIDEO_GENERATE_AUDIO in ("0", "1"):
        payload["generate_audio"] = VIDEO_GENERATE_AUDIO == "1"    
    if images:
        if image_mode == "reference":
            payload["input_references"] = [{"type": "image_url", "image_url": {"url": u}} for u in images]
        else:
            frames_ok = info.get("supported_frame_images") or ["first_frame"]
            if "first_frame" not in frames_ok:
                return _failed(f"{VIDEO_MODEL} n'accepte pas d'image de début (supporté : {', '.join(frames_ok)}).")
            frames = [{"type": "image_url", "image_url": {"url": images[0]}, "frame_type": "first_frame"}]
            if len(images) > 1 and "last_frame" in frames_ok:
                frames.append({"type": "image_url", "image_url": {"url": images[1]}, "frame_type": "last_frame"})
            payload["frame_images"] = frames   # images en trop ignorées si last_frame n'est pas supporté
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


# ---------- aide à la planification / production multi-scènes ----------
def effective_duration(d: int) -> int:
    """Durée réellement générée : la plus proche des durées supportées par le modèle."""
    return d if MOCK else (_closest_duration(d, _model_info()) or d)

def duration_bounds() -> tuple[int, int]:
    """(min, max) des durées de clip supportées par le modèle ; (5, 10) si inconnu."""
    durations = [] if MOCK else (_model_info().get("supported_durations") or [])
    return (min(durations), max(durations)) if durations else (5, 10)
 
 
def plan_cost(durations: list[int], reference: bool | list[bool] = False) -> tuple[float, float] | None:
    """Estimation (min, max) en $ d'après les tarifs de liste de /videos/models.
 
    reference : un booléen pour tout le plan, ou une liste (une valeur par scène : mode référence = plus cher).
    Si VIDEO_RESOLUTION est défini et connu -> estimation précise (min == max) ;
    sinon fourchette sur toutes les résolutions. None si les tarifs sont indisponibles.
    """
    if MOCK:
        return (0.0, 0.0)
    info = _model_info()
    skus = info.get("pricing_skus") or {}
 
    def prices(prefix: str) -> dict:
        return {k[len(prefix):]: float(v) for k, v in skus.items() if k.startswith(prefix)}
 
    standard, referenced = prices("duration_seconds_"), prices("reference_duration_seconds_")
    flags = reference if isinstance(reference, list) else [reference] * len(durations)
    res = VIDEO_RESOLUTION.lower()
    low = high = 0.0
    for d, is_ref in zip(durations, flags):
        table = referenced if is_ref else standard
        if not table:
            return None
        seconds = _closest_duration(d, info) or d
        chosen = [table[res]] if res in table else list(table.values())
        low += seconds * min(chosen)
        high += seconds * max(chosen)
    return (round(low, 2), round(high, 2))
 
 
def wait_for_video(job_id: str, timeout: float | None = None, interval: float | None = None) -> dict:
    """Interroge le job jusqu'à sa fin (completed / failed / ...) ou jusqu'au délai maximum."""
    timeout = timeout if timeout is not None else float(os.getenv("SCENE_TIMEOUT_S", "900"))
    interval = interval if interval is not None else float(os.getenv("VIDEO_POLL_INTERVAL_S", "10"))
    deadline = time.monotonic() + timeout
    while True:
        try:
            job = get_video_status(job_id)
        except httpx.HTTPError:
            job = {"job_id": job_id, "status": "in_progress", "video_url": None}   # erreur réseau passagère
        if job["status"] == "completed" or job["status"] in ("failed", "cancelled", "expired"):
            return job
        if time.monotonic() >= deadline:
            return {"job_id": job_id, "status": "timeout", "video_url": None, "error": "Délai d'attente dépassé"}
        time.sleep(interval)