"""Graphe de production multi-scènes (map-reduce), lancé APRÈS validation du plan.

START --(Send x N)--> scene_worker --> aggregate_scenes --> END

- fan_out         : une tâche `scene_worker` par scène (exécutées en parallèle, MAX_PARALLEL_SCENES max)
- scene_worker    : choisit les images de la scène, soumet le job vidéo, l'enregistre en base, attend la fin
- aggregate_scenes: remet les scènes dans l'ordre et prépare les messages de la conversation

Images : une scène peut avoir SA propre image (`image` = numéro de l'image jointe au plan) :
  - image_mode "first_frame" : la scène commence par cette image (image-to-video)
  - image_mode "reference"   : l'image guide le lieu / le style / le contenu (s'ajoute à l'avatar éventuel)
L'avatar (personnage récurrent) est un usage parmi d'autres : il n'est envoyé qu'en mode référence.

Les identifiants de jobs sont enregistrés dès la soumission : si le serveur redémarre, la production reprend
(scènes terminées ignorées, jobs déjà soumis simplement ré-interrogés) sans repayer.
"""
import operator
import os
import utils.media
from typing import Annotated, Optional

from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

try:
    from langgraph.types import Send
except ImportError:  # anciennes versions de LangGraph
    from langgraph.constants import Send

import utils.db as db
import utils.ffmpeg_tools as ff
from utils.video_tool import create_video, effective_duration, wait_for_video

from utils.audio_tool import create_audio
from utils.config import get_conf
from agents.editor import decide_edit, sanitize
from utils.media import MEDIA_DIR, public_url


MAX_SCENES = int(get_conf("max_scenes"))
MAX_PARALLEL = int(get_conf("max_parallel_scenes"))
MAX_PLAN_COST = float(get_conf("max_plan_cost_usd"))
MAX_QA_RETRIES = int(os.getenv("MAX_QA_RETRIES", "1"))                 # relances d'un clip rejeté par la QA

QA_TOLERANCE = float(os.getenv("QA_DURATION_TOLERANCE_S", "1.5"))
QA_STRICT_DURATION = os.getenv("QA_STRICT_DURATION", "0") == "1" 
# durée hors tolérance = rejet (sinon avertissement)
ASSEMBLE = os.getenv("ASSEMBLE", "1") == "1"
ASSEMBLE_VOICEOVER = os.getenv("ASSEMBLE_VOICEOVER", "1") == "1"       # voix off TTS des scènes qui ont une narration


media = utils.media

def avatar_prefix(desc: Optional[str]) -> str:
    """Phrase placée devant le prompt pour garder le même personnage (image de référence)."""
    if desc:
        return f"Main character (keep the exact same face, hair, skin tone and outfit as the reference image): {desc}. "
    return "Main character: the person shown in the reference image. Keep the exact same face, hair, skin tone and outfit. "


def scene_mode(scene: dict, use_avatar: bool) -> str:
    """'first_frame' (l'image de la scène ouvre la scène), 'reference' (références) ou 'text' (texte seul)."""
    if scene.get("image") and (scene.get("image_mode") or "first_frame") == "first_frame":
        return "first_frame"
    if scene.get("image") or use_avatar:
        return "reference"
    return "text"


class ProdState(TypedDict, total=False):
    plan_id: str
    title: Optional[str]
    edit_style: Optional[str]
    scenes: list[dict]
    images: list[str]                      # avatar (data URLs), vide si le plan n'utilise pas d'avatar
    plan_images: list                      # images jointes au plan (data URLs, dans l'ordre ; None si fichier absent)

    
    avatar_desc: Optional[str]
    results: Annotated[list, operator.add]  # alimenté en parallèle par les workers, puis par la QA
    notes: Annotated[list, operator.add]    # avertissements (QA, voix off, assemblage) pour le résumé
    qa_round: int
    edit: dict
    
    final_messages: list
    final_video_url: Optional[str] 
    ok: int

def _payload(state: ProdState, scene: dict, retry: bool = False) -> dict:
    return {
        "plan_id": state["plan_id"],
        "scene": scene,
        "images": state.get("images") or [],
        "plan_images": state.get("plan_images") or [],
        "avatar_desc": state.get("avatar_desc"),
            "retry": retry,
        }


def fan_out(state: ProdState) -> list:
    return [
        Send("scene_worker", _payload(state, sc)) for sc in state["scenes"]
    ]


def _scene_request(sc: dict, avatar_imgs: list[str], plan_imgs: list, desc: Optional[str]):
    """Prompt + images + mode envoyés pour une scène."""
    n = sc.get("image")
    scene_img = plan_imgs[n - 1] if n and 1 <= n <= len(plan_imgs) else None   # None si numéro/fichier invalide
    mode = scene_mode({**sc, "image": n if scene_img else None}, bool(avatar_imgs))

    if mode == "first_frame":
        # La scène s'ouvre sur son image : pas de références (non combinables avec l'image de début).
        # Le personnage n'est alors maintenu que par sa description écrite.
        prefix = f"Main character: {desc}. " if desc and avatar_imgs else ""
        return prefix + sc["prompt"], [scene_img], "first_frame"
    if mode == "reference":
        refs = list(avatar_imgs) + ([scene_img] if scene_img else [])
        prefix = avatar_prefix(desc) if avatar_imgs else ""
        return prefix + sc["prompt"], refs, "reference"
    return sc["prompt"], None, "first_frame"


def scene_worker(state: dict) -> dict:
    plan_id, sc = state["plan_id"], state["scene"]
    idx = sc["index"]
    base = {"index": idx, "duration": sc["duration"]}

    def done(url: str) -> dict:
        db.upsert_scene(plan_id, idx, status="completed", video_url=url)
        return {"results": [{**base, "status": "completed", "video_url": url, "error": None}]}

    def failed(msg: str | None) -> dict:
        db.upsert_scene(plan_id, idx, status="failed", error=(msg or "erreur inconnue")[:300])
        return {"results": [{**base, "status": "failed", "video_url": None, "error": msg or "erreur inconnue"}]}

    try:
        if state.get("retry"):   # relance après rejet de la QA : on repart d'un nouveau job
            db.upsert_scene(plan_id, idx, status="pending", job_id=None, video_url=None, error=None)
        row = db.get_scene(plan_id, idx)
        if row and row["status"] == "completed" and row["video_url"]:
            return {"results": [{**base, "status": "completed", "video_url": row["video_url"], "error": None}]}

        job_id = row["job_id"] if row else None
        if not job_id:
            prompt, imgs, mode = _scene_request(
                sc, state.get("images") or [], state.get("plan_images") or [], state.get("avatar_desc")
            )
            job = create_video(prompt, sc["duration"], imgs, mode)
            if job["status"] == "failed":
                return failed(job.get("error"))
            job_id = job["job_id"]
            db.upsert_scene(plan_id, idx, status="submitted", job_id=job_id)   # job payé : on le garde
            if job["status"] == "completed" and job["video_url"]:
                return done(job["video_url"])

        final = wait_for_video(job_id)
        if final["status"] == "completed" and final["video_url"]:
            return done(final["video_url"])
        return failed(final.get("error") or final["status"])
    except Exception as e:  # une scène en échec ne doit pas faire tomber les autres
        return failed(str(e))

# ---------- QA ----------
def _latest(results: list) -> dict:
    """Dernier résultat connu de chaque scène (les relances et verdicts QA sont ajoutés à la suite)."""
    latest: dict = {}
    for r in results:
        latest[r["index"]] = r
    return latest
 
 
def _usable(r: dict) -> bool:
    return bool(r["status"] == "completed" and r.get("video_url") and (r.get("qa") is None or r["qa"]["ok"]))
 
 
def qa_scenes(state: ProdState) -> dict:
    """Contrôle ffprobe des clips terminés ; ajoute un verdict (`qa`) à chacun, `rejected` si inexploitable."""
    rnd = state.get("qa_round", 0) + 1
    if not ff.have_ffmpeg():
        return {"qa_round": rnd, "notes": ["QA et assemblage ignorés : ffmpeg / ffprobe introuvables (installe-les pour obtenir la vidéo finale)."]}
    new = []
    for idx, r in sorted(_latest(state.get("results", [])).items()):
        if r["status"] != "completed" or r.get("qa") is not None:
            continue
        verdict = ff.qa_check(ff.ensure_local(r["video_url"]), effective_duration(r["duration"]),
                              QA_TOLERANCE, QA_STRICT_DURATION)
        entry = {**r, "qa": verdict}
        if not verdict["ok"]:
            entry["status"] = "rejected"
            entry["error"] = "QA : " + ", ".join(verdict["issues"])
            db.upsert_scene(state["plan_id"], idx, status="failed", error=entry["error"][:300])
        new.append(entry)
    return {"results": new, "qa_round": rnd}
 
 
def after_qa(state: ProdState):
    """Relance (bornée) des scènes rejetées, sinon passage au montage."""
    rejected = [i for i, r in _latest(state.get("results", [])).items() if r["status"] == "rejected"]
    if rejected and state.get("qa_round", 0) <= MAX_QA_RETRIES:
        by_idx = {s["index"]: s for s in state["scenes"]}
        return [Send("scene_worker", _payload(state, by_idx[i], retry=True)) for i in rejected]
    return "plan_edit"
 
 
# ---------- montage + assemblage ----------
def _kept(state: ProdState) -> list[dict]:
    latest = _latest(state.get("results", []))
    return [latest[i] for i in sorted(latest) if _usable(latest[i])]
 
 
def plan_edit(state: ProdState) -> dict:
    by_idx = {s["index"]: s for s in state["scenes"]}
    scenes = [{**by_idx[r["index"]], "duration": r["duration"]} for r in _kept(state)]
    return {"edit": decide_edit(state.get("title"), state.get("edit_style"), scenes)}
 
 
def assemble_final(state: ProdState) -> dict:
    kept = _kept(state)
    if not ASSEMBLE or not kept:
        return {}
    if not ff.have_ffmpeg():
        if any("ffmpeg" in n for n in state.get("notes", [])):   # déjà signalé par la QA
            return {}
        return {"notes": ["Assemblage ignoré : ffmpeg / ffprobe introuvables (installe-les pour obtenir la vidéo finale)."]}
 
    by_idx = {s["index"]: s for s in state["scenes"]}
    notes: list[str] = []
    items = []
    for r in kept:
        path = ff.ensure_local(r["video_url"])
        if path is None:
            notes.append(f"Scène {r['index']} introuvable en local : absente de la vidéo finale.")
            continue
        narration = None
        text = by_idx[r["index"]].get("narration")
        if ASSEMBLE_VOICEOVER and text:   # voix off TTS seulement pour les scènes qui ont une narration
            res = create_audio("voiceover", text)
            narration = ff.ensure_local(res.get("audio_url")) if res["status"] == "completed" else None
            if narration is None:
                note = f"Voix off ignorée : {res.get('error') or 'synthèse indisponible'}"
                if note not in notes:
                    notes.append(note)
        items.append({"path": path, "narration": narration})
    if not items:
        return {"notes": notes}
 
    edit = sanitize(state.get("edit") or {}, len(items))
    out = MEDIA_DIR / f"final_{state['plan_id']}.mp4"
    try:
        warnings = ff.assemble(items, edit["transitions"], edit["transition_duration"], edit["narration_ducking"], out)
    except Exception as e:
        return {"notes": notes + [f"Assemblage échoué : {e}"]}
    return {"final_video_url": public_url(out.name), "notes": notes + warnings}
 
 
def aggregate_scenes(state: ProdState) -> dict:
    latest = _latest(state.get("results", []))
    total = len(state["scenes"])
    msgs: list[AIMessage] = []
    ok = 0
    for idx in sorted(latest):
        r = latest[idx]
        if _usable(r):
            ok += 1
            msgs.append(AIMessage(content=f"Scène {idx}/{total} ({r['duration']} s)",
                                  additional_kwargs={"video_url": r["video_url"]}))
        elif r["status"] == "rejected":
            msgs.append(AIMessage(content=f"⚠️ Scène {idx}/{total} : écartée ({r.get('error') or 'contrôle qualité'})"))
        else:
            msgs.append(AIMessage(content=f"⚠️ Scène {idx}/{total} : échec ({r.get('error') or 'erreur inconnue'})"))
    final = state.get("final_video_url")
    if final:
        msgs.append(AIMessage(content=f"🎬 Vidéo finale ({ok} scène(s) assemblée(s))", additional_kwargs={"video_url": final}))
    summary = f"{ok}/{total} scènes générées."
    notes = list(dict.fromkeys(state.get("notes", [])))   # sans doublons, dans l'ordre
    if notes:
        summary += "\n\n" + "\n".join(f"ℹ️ {n}" for n in notes)
    msgs.append(AIMessage(content=summary))
    return {"final_messages": msgs, "ok": ok}
 
 
_builder = StateGraph(ProdState)
_builder.add_node("scene_worker", scene_worker)
_builder.add_node("qa_scenes", qa_scenes)
_builder.add_node("plan_edit", plan_edit)
_builder.add_node("assemble_final", assemble_final)
_builder.add_node("aggregate_scenes", aggregate_scenes)
_builder.add_conditional_edges(START, fan_out, ["scene_worker"])
_builder.add_edge("scene_worker", "qa_scenes")
_builder.add_conditional_edges("qa_scenes", after_qa, ["scene_worker", "plan_edit"])
_builder.add_edge("plan_edit", "assemble_final")
_builder.add_edge("assemble_final", "aggregate_scenes")
_builder.add_edge("aggregate_scenes", END)
 
production_graph = _builder.compile()
 



# def aggregate_scenes(state: ProdState) -> dict:
#     results = sorted(state.get("results", []), key=lambda r: r["index"])
#     total = len(state["scenes"])
#     msgs: list[AIMessage] = []
#     ok = 0
#     for r in results:
#         if r["status"] == "completed" and r["video_url"]:
#             ok += 1
#             msgs.append(AIMessage(content=f"Scène {r['index']}/{total} ({r['duration']} s)",
#                                   additional_kwargs={"video_url": r["video_url"]}))
#         else:
#             msgs.append(AIMessage(content=f"⚠️ Scène {r['index']}/{total} : échec ({r.get('error') or 'erreur inconnue'})"))
#     summary = f"{ok}/{total} scènes générées."
#     if ok == total and total > 0:
#         summary += " L'assemblage final (FFmpeg) sera la prochaine étape."
#     msgs.append(AIMessage(content=summary))
#     return {"final_messages": msgs, "ok": ok}

# def qa_agent_node(state: ProdState) -> dict:
#     """Nœud QA Agent (ffprobe) : Vérifie l'intégrité de chaque fichier vidéo."""
#     if not state.get("ok"):
#         return {"final_messages": [AIMessage(content="⚠️ QA : Assemblage ignoré car des scènes ont échoué.")]}

#     results = sorted(state.get("results", []), key=lambda r: r["index"])
#     scenes_meta = state.get("scenes", [])
#     total = len(scenes_meta)
#     msgs = []
#     assembled_scenes = []
#     valid_count = 0

#     for r in results:
#         idx = r.get("index", 1)
#         video_url = r.get("video_url")
#         local_path = ensure_local(video_url)
#         expected_dur = scenes_meta[idx - 1].get("duration") if idx <= len(scenes_meta) else None

#         # Contrôle technique strict via ffprobe
#         qa = qa_check(local_path, expected_duration=expected_dur)
#         if qa["ok"]:
#             valid_count += 1
#             narration_url = scenes_meta[idx - 1].get("narration_url") if idx <= len(scenes_meta) else None
#             assembled_scenes.append({
#                 "path": local_path,
#                 "narration": ensure_local(narration_url) if narration_url else None
#             })
#             msgs.append(AIMessage(content=f"🔍 Scène {idx}/{total} validée par le QA technique."))
#         else:
#             issues = ", ".join(qa.get("issues", ["invalide"]))
#             msgs.append(AIMessage(content=f"❌ Scène {idx}/{total} rejetée par le QA : {issues}"))
#             valid_count = -1  # Marque un échec global QA
#             break

#     is_valid = (valid_count == total)
#     return {
#         "final_messages": msgs,
#         "ok": is_valid,
#         "assembled_scenes": assembled_scenes if is_valid else []
#     }


# def editor_agent_node(state: ProdState) -> dict:
#     """Nœud Editor Agent : Choix des transitions et du ducking audio."""
#     if not state.get("ok"):
#         return {}

#     title = state.get("title")
#     edit_style = state.get("edit_style")
#     scenes_meta = state.get("scenes", [])

#     decisions = decide_edit(title, edit_style, scenes_meta)
#     return {"edit_decisions": decisions}


# def ffmpeg_node(state: ProdState) -> dict:
#     """Nœud FFmpeg : Assemblage final et normalisation des clips."""
#     if not state.get("ok"):
#         return {"final_video_url": None}

#     output_filename = f"final_{state.get('plan_id', 'output')}.mp4"
#     output_path = media.MEDIA_DIR / output_filename
#     scenes = state.get("assembled_scenes", [])
#     decisions = state.get("edit_decisions", {"transitions": [], "transition_duration": 0.5, "narration_ducking": 0.25})

#     try:
#         assemble(
#             scenes=scenes,
#             transitions=decisions["transitions"],
#             td=decisions["transition_duration"],
#             duck=decisions["narration_ducking"],
#             dst=output_path
#         )
#         final_url = media.public_url(output_filename)
#         return {
#             "final_video_url": final_url,
#             "final_messages": [AIMessage(content="🎬 **Voici votre vidéo finale assemblée avec succès !**", additional_kwargs={"video_url": final_url})]
#         }
#     except Exception as e:
#         return {
#             "ok": False,
#             "final_messages": [AIMessage(content=f"❌ Échec de l'assemblage FFmpeg : {e}")]
#         }

# _builder = StateGraph(ProdState)
# _builder.add_node("scene_worker", scene_worker)
# _builder.add_node("aggregate_scenes", aggregate_scenes)
# _builder.add_node("qa_agent", qa_agent_node)
# _builder.add_node("editor_agent", editor_agent_node)
# _builder.add_node("ffmpeg_builder", ffmpeg_node)

# _builder.add_conditional_edges(START, fan_out, ["scene_worker"])
# _builder.add_edge("scene_worker", "aggregate_scenes")
# _builder.add_edge("aggregate_scenes", "qa_agent")
# _builder.add_edge("qa_agent", "editor_agent")
# _builder.add_edge("editor_agent", "ffmpeg_builder")
# _builder.add_edge("ffmpeg_builder", END)

# production_graph = _builder.compile()
