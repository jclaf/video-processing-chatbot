import base64
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage

import utils.db
from utils.config import load_config, save_config
from agents.graph import graph
from utils.media import MEDIA_DIR, public_url, save_media
from agents.production import MAX_PARALLEL, MAX_PLAN_COST, MAX_SCENES, avatar_prefix, scene_mode, production_graph
from agents.editor import decide_edit
#from utils.config import load_config, save_config, get_conf

MAX_IMAGE_BYTES = int(float(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024)
IMAGE_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
MIME_BY_EXT = {ext: mime for mime, ext in IMAGE_EXT.items()}


# ---------- production multi-scènes en tâche de fond ----------
_active_runs: set[str] = set()
_runs_lock = threading.Lock()

db = utils.db

def _run_plan(conv_id: str, plan_id: str) -> None:
    try:
        plan = db.get_plan(plan_id)
        plan_images = [_file_to_data_url(f) if (MEDIA_DIR / f).exists() else None for f in (plan.get("plan_files") or [])]
        images, desc = [], plan["avatar_desc"]
        if plan["use_avatar"]:
            avatar = db.get_avatar(conv_id)
            images = [_file_to_data_url(f) for f in avatar["files"] if (MEDIA_DIR / f).exists()]
            desc = desc or avatar["desc"]
        out = production_graph.invoke(
            {"plan_id": plan_id, "scenes": plan["scenes"], "images": images,
             "plan_images": plan_images, "avatar_desc": desc,
             "title": plan["title"], "edit_style": plan["edit_style"]},
            config={"max_concurrency": MAX_PARALLEL},
        )
        db.save_messages(conv_id, out.get("final_messages", []))
        db.set_plan_status(plan_id, "done" if out.get("ok") else "failed")
    except Exception as e:
        db.save_messages(conv_id, [AIMessage(content=f"La production a échoué : {e}")])
        db.set_plan_status(plan_id, "failed")
    finally:
        with _runs_lock:
            _active_runs.discard(plan_id)


def _start_plan(conv_id: str, plan_id: str) -> None:
    with _runs_lock:
        if plan_id in _active_runs:
            return
        _active_runs.add(plan_id)
    threading.Thread(target=_run_plan, args=(conv_id, plan_id), daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reprise après redémarrage : jobs déjà soumis ré-interrogés, scènes terminées ignorées (rien n'est repayé)
    for p in db.list_plans_by_status("running"):
        _start_plan(p["conv_id"], p["id"])
    yield

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Video Chatbot API", lifespan=lifespan)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")  # vidéos, audio et images


def _plan_view(plan: dict) -> dict:
    rows = {r["idx"]: r for r in db.list_scenes(plan["id"])}
    scenes = []
    for s in plan["scenes"]:
        r = rows.get(s["index"]) or {}
        scenes.append({"index": s["index"], "duration": s["duration"],
                       "status": r.get("status") or "pending", "error": r.get("error")})
    return {
        "id": plan["id"], "status": plan["status"], "title": plan["title"],
        "est_low": plan["est_low"], "est_high": plan["est_high"], "use_avatar": plan["use_avatar"],
        "scenes": scenes, "total": len(scenes),
        "done": sum(1 for s in scenes if s["status"] == "completed"),
        "failed": sum(1 for s in scenes if s["status"] == "failed"),
    }


def _progress_text(conv_id: str) -> str | None:
    running = db.get_plan_by_status(conv_id, "running")
    if not running:
        return None
    v = _plan_view(running)
    text = f"{v['done']}/{v['total']} scènes terminées"
    return text + (f", {v['failed']} en échec" if v["failed"] else "")


def _serialize(m) -> dict:
    return {
        "role": "user" if m.type == "human" else "assistant",
        "content": m.content if isinstance(m.content, str) else str(m.content),
        "video_url": m.additional_kwargs.get("video_url"),
        "audio_url": m.additional_kwargs.get("audio_url"),
        "image_urls": m.additional_kwargs.get("image_urls") or [],
    }


def _clean_name(name: str | None) -> str:
    """Nom d'origine d'une image (indice pour le planificateur) : sans chemin ni caractères exotiques."""
    base = os.path.basename(name or "image")[:60]
    return "".join(ch for ch in base if ch.isprintable()) or "image"


def _check(conv_id: str) -> None:
    if not db.exists(conv_id):
        raise HTTPException(404, "Conversation introuvable")


def _file_to_data_url(filename: str) -> str:
    data = (MEDIA_DIR / filename).read_bytes()
    mime = MIME_BY_EXT.get(filename.rsplit(".", 1)[-1].lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


@app.get("/conversations")
def list_conversations(limit: int = 30):
    return db.list_conversations(limit)


@app.post("/conversations")
def new_conversation():
    conv_id = str(uuid.uuid4())
    db.create_conversation(conv_id)
    return {"id": conv_id, "title": db.DEFAULT_TITLE}


@app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str):
    _check(conv_id)
    return [_serialize(m) for m in db.load_messages(conv_id)]


@app.get("/conversations/{conv_id}/avatar")
def get_avatar(conv_id: str):
    _check(conv_id)
    av = db.get_avatar(conv_id)
    return {"image_urls": [public_url(f) for f in av["files"]], "description": av["desc"]}


@app.delete("/conversations/{conv_id}/avatar")
def delete_avatar(conv_id: str):
    _check(conv_id)
    db.clear_avatar(conv_id)
    return {"deleted": "avatar"}


# ---------- plan multi-scènes : consultation, validation, annulation ----------
@app.get("/conversations/{conv_id}/plan")
def get_plan(conv_id: str):
    _check(conv_id)
    plan = db.get_latest_plan(conv_id)
    return _plan_view(plan) if plan else {"status": None}


@app.post("/conversations/{conv_id}/plan/approve")
def approve_plan(conv_id: str):
    _check(conv_id)
    plan = db.get_plan_by_status(conv_id, "awaiting_approval")
    if not plan:
        raise HTTPException(409, "Aucun plan en attente de validation.")
    if db.get_plan_by_status(conv_id, "running"):
        raise HTTPException(409, "Une production est déjà en cours dans cette conversation.")
    if plan["est_high"] is not None and plan["est_high"] > MAX_PLAN_COST:
        raise HTTPException(400, (
            f"Coût estimé ({plan['est_high']:.2f} $) supérieur au plafond MAX_PLAN_COST_USD "
            f"({MAX_PLAN_COST:.0f} $). Réduis le plan ou augmente le plafond."
        ))
    db.set_plan_status(plan["id"], "running")
    _start_plan(conv_id, plan["id"])
    return _plan_view(db.get_plan(plan["id"]))


@app.post("/conversations/{conv_id}/plan/cancel")
def cancel_plan(conv_id: str):
    _check(conv_id)
    plan = db.get_plan_by_status(conv_id, "awaiting_approval")
    if not plan:
        raise HTTPException(409, "Aucun plan en attente de validation.")
    db.set_plan_status(plan["id"], "cancelled")
    return {"status": "cancelled"}


@app.post("/conversations/{conv_id}/chat")
def chat(
    conv_id: str,
    message: str = Form(...),
    image_mode: str = Form("first_frame"),   # "first_frame" (animer) | "reference" (style / contenu)
    images: list[UploadFile] = File(default=[]),
):
    _check(conv_id)
    message = message.strip()
    if not message:
        raise HTTPException(400, "Le message est vide : décris ta demande (même avec des images jointes).")
    if image_mode not in ("first_frame", "reference"):
        raise HTTPException(400, "image_mode invalide")

    # Images jointes : copie locale (affichage) + data URL base64 (envoyée à OpenRouter)
    data_urls, public_urls = [], []
    for f in images:
        mime = (f.content_type or "").lower()
        if mime not in IMAGE_EXT:
            raise HTTPException(400, f"Format non supporté ({mime or 'inconnu'}). Utilise PNG, JPEG ou WebP.")
        data = f.file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Image trop lourde (max {MAX_IMAGE_BYTES // 1024 // 1024} Mo).")
        public_urls.append(save_media(data, IMAGE_EXT[mime]))
        data_urls.append(f"data:{mime};base64,{base64.b64encode(data).decode()}")

    # Avatar déjà enregistré pour cette conversation
    avatar = db.get_avatar(conv_id)
    avatar_data = [_file_to_data_url(f) for f in avatar["files"] if (MEDIA_DIR / f).exists()]
    pending = db.get_plan_by_status(conv_id, "awaiting_approval")

    history = db.load_messages(conv_id)
    human = HumanMessage(content=message, additional_kwargs={"image_urls": public_urls})
    try:
        result = graph.invoke({
            "messages": history + [human],
            "last_job_id": db.get_last_job_id(conv_id),
            "images": data_urls,
            "image_mode": image_mode,
            "avatar_images": avatar_data,
            "avatar_saved_desc": avatar["desc"],
            "image_names": [_clean_name(f.filename) for f in images],
            "pending_plan": pending,
            "plan_progress": _progress_text(conv_id),
        })
    except Exception as e:   # erreur du LLM, de validation, d'API... : message lisible côté interface
        log.exception("Erreur dans le graphe de conversation")
        raise HTTPException(500, f"Le traitement a échoué ({type(e).__name__}) : {str(e)[:300]}")
    
    route = result.get("route")
    # Branche avatar : les images jointes deviennent l'avatar de la conversation
    if route == "generate_avatar_video":
        new_files = [u.rsplit("/", 1)[-1] for u in public_urls] or None
        db.set_avatar(conv_id, files=new_files, desc=result.get("avatar_description") or None)

    # Branche plan : le plan est enregistré en attente de validation (rien n'est généré ni facturé)
    plan = result.get("plan") if route == "plan_video" else None
    if plan:
        cost = result.get("plan_cost") or [None, None]
        plan_files = [u.rsplit("/", 1)[-1] for u in public_urls]   # toutes les images jointes, dans l'ordre
        db.create_plan(str(uuid.uuid4()), conv_id, plan["title"], plan["scenes"], cost[0], cost[1],
                       plan["use_avatar"], plan.get("avatar_description"), plan_files)
        # Seules les images désignées comme avatar deviennent l'avatar de la conversation
        nums = plan.get("avatar_images") or []
        if plan["use_avatar"] and (nums or plan.get("avatar_description")):
            avatar_files = [plan_files[n - 1] for n in nums if 1 <= n <= len(plan_files)] or None
            db.set_avatar(conv_id, files=avatar_files, desc=plan.get("avatar_description") or None)

    new_msgs = result["messages"][len(history):]   # message utilisateur + réponses (vidéo, audio...)
    db.save_messages(conv_id, new_msgs)
    db.touch(conv_id, message, result.get("last_job_id"))

    return {"messages": [_serialize(m) for m in new_msgs if m.type != "human"]}


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    db.delete_conversation(conv_id)
    return {"deleted": conv_id}

@app.get("/settings")
def get_settings():
    return load_config()

@app.post("/settings")
def update_settings(settings: dict = Body(...)):
    current = load_config()
    current.update(settings)
    save_config(current)
    return {"status": "success", "settings": current}