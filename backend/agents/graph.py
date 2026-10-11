"""Graphe LangGraph :

START -> orchestrator -> chat | generate_video | generate_audio | video_status
                                   |
                                   +-- (si with_audio) --> generate_audio

La mémoire est persistée par thread_id (= id de conversation) via SqliteSaver.
"""
import os
import traceback
import json
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict
from pydantic import BaseModel, model_validator
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

#from utils.media import MEDIA_DIR, public_url, save_media
from utils.video_tool import create_video, duration_bounds, get_video_status, plan_cost
from utils.audio_tool import create_audio
from utils.config import get_conf
from agents.production import MAX_PARALLEL, MAX_PLAN_COST, MAX_SCENES, avatar_prefix, scene_mode, production_graph

MAX_SCENES = int(get_conf("max_scenes"))
NATIVE_VOICE = "1" if get_conf("native_voice") else "0"

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
    temperature=0,
)

class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    route: str
    # vidéo
    video_prompt: Optional[str]
    duration: int
    last_job_id: Optional[str]
    # images jointes au dernier message (data URLs) et mode d'utilisation (first_frame | reference)
    images: list[str]
    image_mode: str
    image_names: list[str]              # noms d'origine des images jointes (même ordre)
    # avatar
    avatar_images: list[str]            # avatar enregistré pour la conversation (data URLs)
    avatar_saved_desc: Optional[str]    # description enregistrée
    avatar_description: Optional[str]   # description donnée dans ce message
    # plan multi-scènes
    pending_plan: Optional[dict]        # plan en attente de validation (entrée)
    plan_progress: Optional[str]        # production en cours (entrée), ex. "2/5 scènes terminées"
    plan: Optional[dict]                # plan proposé ce tour-ci (sortie)
    plan_cost: Optional[list]           # [min, max] en $ (sortie)
    # audio
    with_audio: bool
    audio_kind: str
    audio_text: Optional[str]
 

class _NullTolerant(BaseModel):
    """Les LLM renvoient souvent `null` pour les champs optionnels (même ceux qui ont une valeur par défaut) :
    on retire ces null pour que la valeur par défaut s'applique, au lieu d'une erreur de validation (HTTP 500)."""
 
    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        return {k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else data
 
class Route(_NullTolerant):
    route: Literal["chat", "generate_video", "generate_avatar_video", "plan_video", "generate_audio", "video_status"]
    video_prompt: Optional[str] = None        # prompt visuel détaillé (anglais)
    duration: int = 5                         # secondes
    avatar_description: Optional[str] = None  # description physique du personnage, si l'utilisateur la donne
    with_audio: bool = False                  # vidéo + audio demandés ensemble
    audio_kind: Literal["voiceover", "music"] = "voiceover"
    audio_text: Optional[str] = None          # texte de la voix off, ou prompt de la musique
    scene_count: int = 1
 
class Scene(_NullTolerant):
    prompt: str                         # prompt visuel en anglais
    duration: int                       # secondes
    narration: Optional[str] = None     # voix off de la scène (français), seulement si fournie / demandée
    image: Optional[int] = None         # numéro (1..N) de l'image jointe utilisée dans cette scène
    image_mode: Optional[Literal["first_frame", "reference"]] = "first_frame"   # début de scène animé | référence
 
 
class Plan(_NullTolerant):
    title: str
    scenes: list[Scene]
    use_avatar: bool = False
    avatar_images: list[int] = []       # numéros des images jointes qui montrent l'avatar (récurrent)
    avatar_description: Optional[str] = None
    edit_style: Optional[str] = None
 
router_llm = llm.with_structured_output(Route, method="function_calling")
planner_llm = llm.with_structured_output(Plan, method="function_calling")
 
ROUTER_PROMPT = SystemMessage(content=(
    "Tu es l'orchestrateur d'un assistant de création de vidéos et d'audio. Choisis la route :\n"
    "- generate_video : UN SEUL plan / clip (jusqu'à {max_clip} s) à partir d'un texte, ou animer une image "
    "jointe. Remplis video_prompt (prompt visuel détaillé, en anglais) et duration. Si l'utilisateur décrit "
    "PLUSIEURS plans ou scènes, ce n'est PAS cette route : choisis plan_video. Si l'utilisateur veut aussi une "
    "voix off ou une musique, mets with_audio=true et remplis audio_kind + audio_text.\n"
    "- generate_avatar_video : UN seul clip (jusqu'à {max_clip} s) mettant en scène un AVATAR / personnage qui "
    "doit rester identique (« avec cet avatar », « avec ce personnage », « avec mon avatar »). video_prompt "
    "décrit la SCÈNE (action, décor, caméra), sans redécrire le personnage. avatar_description = description "
    "physique du personnage UNIQUEMENT si l'utilisateur la donne. Mêmes champs audio que generate_video.\n"
    "- plan_video : vidéo en PLUSIEURS plans / scènes (dès 2 plans ou scènes décrits, MÊME si la durée totale "
    "tient dans {max_clip} s, ex. « 15 s avec 3 plans »), OU plus longue que {max_clip} s, avec ou sans avatar "
    "ou images. Chaque scène est générée séparément. Rien n'est généré à ce stade : un planificateur proposera "
    "un plan que l'utilisateur validera. Aussi pour toute demande de modification d'un plan en attente.\n"
    "- generate_audio : audio seul (voix off ou musique). audio_kind='voiceover' avec audio_text = "
    "le texte exact à dire, ou 'music' avec audio_text = description du style.\n"
    "- video_status : demande l'état / le résultat d'une vidéo ou d'une production en cours.\n"
    "- chat : tout le reste (idées, questions, précisions à demander avant de générer).\n"
    "Dans tous les cas, scene_count = nombre de plans / scènes distincts décrits par l'utilisateur "
    "(1 s'il n'y en a qu'un)."
))

#     "Tu es l'orchestrateur d'un assistant de création de vidéos et d'audio. Choisis la route :\n"
#     "- generate_video : UN clip court (quelques secondes) à partir d'un texte, ou animer une image jointe. "
#     "Remplis video_prompt (prompt visuel détaillé, en anglais) et duration. Si l'utilisateur veut aussi une "
#     "voix off ou une musique, mets with_audio=true et remplis audio_kind + audio_text.\n"
#     "- generate_avatar_video : UN clip court mettant en scène un AVATAR / personnage qui doit rester identique "
#     "(« avec cet avatar », « avec ce personnage », « avec mon avatar »). video_prompt décrit la SCÈNE "
#     "(action, décor, caméra), sans redécrire le personnage. avatar_description = description physique "
#     "du personnage UNIQUEMENT si l'utilisateur la donne. Mêmes champs audio que generate_video.\n"
#     "- plan_video : vidéo LONGUE ou en PLUSIEURS scènes / séquences (plus de ~15 secondes, ou plusieurs scènes "
#     "décrites), avec ou sans avatar. Rien n'est généré à ce stade : un planificateur proposera un plan que "
#     "l'utilisateur validera. Aussi pour toute demande de modification d'un plan en attente.\n"
#     "- generate_audio : audio seul (voix off ou musique). audio_kind='voiceover' avec audio_text = "
#     "le texte exact à dire, ou 'music' avec audio_text = description du style.\n"
#     "- video_status : demande l'état / le résultat d'une vidéo ou d'une production en cours.\n"
#     "- chat : tout le reste (idées, questions, précisions à demander avant de générer)."
 
IMAGE_HINT = (
    "\n\nL'utilisateur a joint {n} image(s) à son dernier message. S'il parle d'un avatar / d'un personnage "
    "à garder pour UN clip, choisis generate_avatar_video. Pour une vidéo longue / multi-scènes (avec avatar, "
    "ou avec des images à utiliser dans certaines scènes), choisis plan_video. S'il veut animer l'image (ou ne "
    "précise rien) pour UN clip, choisis generate_video. video_prompt décrit le mouvement / la scène, PAS le "
    "contenu des images."
)
 
AVATAR_HINT = (
    "\n\nUn avatar est déjà enregistré pour cette conversation. Si l'utilisateur demande une vidéo avec "
    "« mon avatar », « le même personnage », « lui » ou « elle », choisis generate_avatar_video "
    "(ou plan_video si la vidéo est longue / multi-scènes), même sans pièce jointe."
)
 
PLAN_HINT = (
    "\n\nUn plan multi-scènes attend la validation de l'utilisateur (il valide avec des boutons). "
    "S'il demande de le modifier (scène, durée, style, nombre de scènes), choisis plan_video."
)
 
PLANNER_PROMPT = (
    "Tu es le planificateur d'une production vidéo multi-scènes. Découpe la demande en scènes de {lo} à {hi} "
    "secondes chacune (maximum {max_scenes} scènes, durée totale proche de celle demandée ; sans durée donnée, "
    "vise 30 à 45 secondes).\n"
    "Pour chaque scène :\n"
    "- prompt : prompt visuel détaillé en ANGLAIS (sujet, action, décor, lumière, caméra). Aucun texte à l'écran, "
    "aucune parole dans le prompt. Termine TOUS les prompts par la même phrase de style (palette, ambiance, grain, "
    "mouvement de caméra) pour garder une cohérence visuelle entre les scènes.\n"
    "- duration : en secondes, entre {lo} et {hi}.\n"
    "- narration : voix off de la scène, en français, SEULEMENT si l'utilisateur fournit un texte ou demande une "
    "voix off ; découpe le texte fidèlement par scène (environ 2,5 mots par seconde de scène).\n"
    "use_avatar = true si l'utilisateur veut un avatar / personnage identique d'une scène à l'autre "
    "(« avec cet avatar », « mon avatar »). avatar_images = numéros des images jointes qui montrent cet avatar "
    "(liste vide si l'avatar est déjà enregistré ou si aucune image n'est jointe). avatar_description = "
    "description physique du personnage uniquement si l'utilisateur la donne.\n"
    "Images jointes (numérotées dans l'ordre d'envoi) : une image n'est PAS forcément un avatar. Si l'utilisateur "
    "dit à quoi elles servent, assigne-les aux scènes avec `image` (numéro) et `image_mode` : 'first_frame' = la "
    "scène COMMENCE par cette image (elle est animée), 'reference' = l'image guide le lieu, le style ou le "
    "contenu. Une même image peut servir à plusieurs scènes. N'assigne une image à une scène que si "
    "l'utilisateur le demande ou si c'est évident, sinon laisse `image` vide. Pour une scène animée à partir "
    "d'une image, le prompt décrit le MOUVEMENT et la caméra, pas le contenu de l'image."
    "edit_style = ambiance / rythme / transitions demandés pour le montage (ex. « transitions fluides », "
    "« rythme rapide »), en quelques mots, UNIQUEMENT si l'utilisateur en parle."
)
 
CHAT_PROMPT = SystemMessage(content=(
    "Tu es un assistant créatif qui aide à concevoir des vidéos et leur audio (voix off, musique). "
    "Réponds en français, pose des questions courtes si le brief est flou (sujet, style, durée, voix)."
))
 
 
def _last_human(state: State) -> str:
    return next(m.content for m in reversed(state["messages"]) if m.type == "human")
 
 
def _with_native_voice(state: State, prompt: str) -> str:
    if NATIVE_VOICE and state.get("with_audio") and state.get("audio_kind") == "voiceover" and state.get("audio_text"):
        prompt += f' A calm, energetic narrator speaks in French: "{state["audio_text"]}"'
    return prompt
 
 
def orchestrator(state: State) -> dict:
    system = ROUTER_PROMPT.content.format(max_clip=duration_bounds()[1])
    if state.get("images"):
        system += IMAGE_HINT.format(n=len(state["images"]))
    if state.get("avatar_images"):
        system += AVATAR_HINT
    if state.get("pending_plan"):
        system += PLAN_HINT
    r = router_llm.invoke([SystemMessage(content=system)] + state["messages"][-10:])
    
    route = r.route
    if route in ("generate_video", "generate_avatar_video") and (r.duration > duration_bounds()[1] or r.scene_count >= 2):
        route = "plan_video"
    
    return {
        "route": route,
        "video_prompt": r.video_prompt,
        "duration": r.duration,
        "avatar_description": r.avatar_description,
        "with_audio": r.with_audio,
        "audio_kind": r.audio_kind,
        "audio_text": r.audio_text,
    }
 
 
def chat_node(state: State) -> dict:
    return {"messages": [llm.invoke([CHAT_PROMPT] + state["messages"][-20:])]}
 
 
def _video_result(state: State, job: dict, ready_text: str) -> dict:
    if job["status"] == "failed":
        return {"messages": [AIMessage(content=f"La génération vidéo a échoué : {job.get('error', 'erreur inconnue')}")]}
    done = job["status"] == "completed" and job["video_url"]
    text = ready_text if done else (
        "Génération lancée, ça prend de 30 s à quelques minutes. Demande-moi « où en est ma vidéo ? » pour la récupérer."
    )
    msg = AIMessage(content=text, additional_kwargs={"video_url": job["video_url"], "job_id": job["job_id"]})
    return {"messages": [msg], "last_job_id": job["job_id"]}
 
 
def video_node(state: State) -> dict:
    prompt = _with_native_voice(state, state.get("video_prompt") or _last_human(state))
    job = create_video(prompt, state.get("duration", 5), state.get("images"), state.get("image_mode", "first_frame"))
    return _video_result(state, job, "Voici ta vidéo !")
 
 
def avatar_video_node(state: State) -> dict:
    """Clip avec avatar : l'image de l'avatar est envoyée en RÉFÉRENCE (même personnage d'un clip à l'autre)."""
    images = state.get("images") or state.get("avatar_images") or []
    if not images:
        return {"messages": [AIMessage(content=(
            "Je n'ai pas encore d'avatar pour cette conversation. Joins une image de ton avatar (📎) "
            "en précisant « avec cet avatar », et je le garderai pour les prochaines vidéos."
        ))]}
 
    desc = state.get("avatar_description") or state.get("avatar_saved_desc")
    scene = _with_native_voice(state, state.get("video_prompt") or _last_human(state))
    job = create_video(avatar_prefix(desc) + scene, state.get("duration", 5), images, "reference")
    return _video_result(state, job, "Voici ta vidéo avec ton avatar !")
 
 
def _cost_line(cost: Optional[tuple], use_avatar: bool) -> str:
    if cost is None:
        return "💰 Coût : non estimable (tarifs indisponibles pour ce modèle)."
    low, high = cost
    if high == 0:
        return "💰 Mode mock : aucun coût réel."
    mode = ", dont scènes en mode référence" if use_avatar else ""
    if low == high:
        return f"💰 Coût estimé : ≈ {low:.2f} $ (tarif de liste{mode})."
    return (f"💰 Coût estimé : entre {low:.2f} et {high:.2f} $ (résolution par défaut inconnue : "
            f"fixe VIDEO_RESOLUTION pour une estimation précise{mode}).")
 
 
def planner_node(state: State) -> dict:
    """Découpe la demande en scènes (avec leurs images) et estime le coût. Rien n'est généré : le plan attend la validation."""
    lo, hi = duration_bounds()
    n_images = len(state.get("images") or [])
    names = state.get("image_names") or []
    system = PLANNER_PROMPT.format(lo=lo, hi=hi, max_scenes=MAX_SCENES)
    if n_images:
        listing = ", ".join(f"{i}. {names[i - 1] if i <= len(names) else 'image'}" for i in range(1, n_images + 1))
        system += f"\n\nImages jointes à ce message : {listing}."
    pending = state.get("pending_plan")
    if pending:
        system += (
            "\n\nPlan actuel, en attente de validation (modifie-le selon la demande de l'utilisateur, ou "
            "remplace-le si la demande porte sur un autre sujet) :\n"
            + json.dumps(pending["scenes"], ensure_ascii=False)
        )
    #plan = planner_llm.invoke([SystemMessage(content=system)] + state["messages"][-10:])

    # Votre code qui lance le graphe ou la génération
    plan = planner_llm.invoke([SystemMessage(content=system)] + state["messages"][-10:])
   
    # --- SÉCURITÉ : Vérifier si le LLM a bien renvoyé un plan valide ---
    if plan is None or not hasattr(plan, "scenes") or not plan.scenes:
        return {
            "messages": [AIMessage(content=(
                "Je n'ai pas réussi à générer ou structurer un plan valide pour cette demande. Peux-tu reformuler ou préciser ta requête ?"
            ))]
        }
    # -----------------------------------------------------------------
    
    def valid(n):   # numéro d'image valide (1..N) ou None
        return n if isinstance(n, int) and 1 <= n <= n_images else None
 
    scenes = [
        {
            "index": i,
            "prompt": s.prompt.strip(),
            "duration": max(lo, min(hi, s.duration)),
            "narration": (s.narration or "").strip() or None,
            "image": valid(s.image),
            "image_mode": s.image_mode if valid(s.image) else None,
        }
        for i, s in enumerate(plan.scenes[:MAX_SCENES], start=1)
    ]
    if not scenes:
        return {"messages": [AIMessage(content=(
            "Je n'ai pas réussi à découper cette demande en scènes. Peux-tu préciser le sujet et la durée voulue ?"
        ))]}
 
    # Avatar : images désignées comme telles ; « avec cet avatar » + images sans affectation aux scènes -> toutes
    avatar_nums = sorted({n for n in plan.avatar_images if valid(n)})
    if plan.use_avatar and not avatar_nums and n_images and not any(sc["image"] for sc in scenes):
        avatar_nums = list(range(1, n_images + 1))
    saved_avatar = bool(state.get("avatar_images"))
    use_avatar = bool(avatar_nums) or (plan.use_avatar and saved_avatar)
    desc = plan.avatar_description or state.get("avatar_description")
 
    cost = plan_cost(
        [sc["duration"] for sc in scenes],
        [scene_mode(sc, use_avatar) == "reference" for sc in scenes],   # le mode référence coûte plus cher
    )
    total = sum(sc["duration"] for sc in scenes)
 
    blocks = [f"**{plan.title}** : {len(scenes)} scène(s), {total} s au total."]
    if use_avatar:
        blocks.append("👤 Avatar : " + (
            ", ".join(f"image {n}" for n in avatar_nums) if avatar_nums else "avatar enregistré") + " (gardé en référence)")
    for sc in scenes:
        block = f"**Scène {sc['index']} ({sc['duration']} s)** : {sc['prompt']}"
        if sc["image"]:
            block += f"\n\n🖼 image {sc['image']} (" + ("début de scène" if sc["image_mode"] == "first_frame" else "référence") + ")"
        if sc["narration"]:
            block += f"\n\n🎙 {sc['narration']}"
        blocks.append(block)
    if len(plan.scenes) > MAX_SCENES:
        blocks.append(f"ℹ️ Plan limité à {MAX_SCENES} scènes (MAX_SCENES) : le reste a été retiré.")
    if plan.use_avatar and not use_avatar:
        blocks.append("ℹ️ Aucun avatar disponible : joins une image en précisant « avec cet avatar » pour garder le même personnage.")
    if use_avatar and any(sc["image"] and sc["image_mode"] == "first_frame" for sc in scenes):
        blocks.append("ℹ️ Les scènes qui commencent par une image n'utilisent pas l'avatar en référence (description écrite seulement).")
    blocks.append(_cost_line(cost, use_avatar))
    if cost and cost[1] > MAX_PLAN_COST:
        blocks.append(
            f"⚠️ Au-dessus du plafond de {MAX_PLAN_COST:.0f} $ (MAX_PLAN_COST_USD) : réduis le plan ou "
            "augmente le plafond pour pouvoir le lancer."
        )
    blocks.append("Valide avec les boutons **Lancer** / **Annuler**, ou dis-moi ce que tu veux changer.")
 
    return {
        "messages": [AIMessage(content="\n\n".join(blocks))],
        "plan": {"title": plan.title, "scenes": scenes, "use_avatar": use_avatar,
                 "avatar_description": desc, "avatar_images": avatar_nums, 
                 "edit_style": (plan.edit_style or "").strip() or None},
        "plan_cost": list(cost) if cost else None,
    }
 
 
def audio_node(state: State) -> dict:
    kind = state.get("audio_kind", "voiceover")
    text = state.get("audio_text") or state.get("video_prompt") or _last_human(state)
    res = create_audio(kind, text, state.get("duration"))
    label = "ta voix off" if kind == "voiceover" else "ta musique"
    if res["status"] == "completed" and res["audio_url"]:
        msg = AIMessage(content=f"Voici {label} !", additional_kwargs={"audio_url": res["audio_url"]})
    elif res["status"] == "unsupported":
        msg = AIMessage(content="La musique n'est pas encore branchée dans ce projet. Je peux faire une voix off.")
    else:
        msg = AIMessage(content=f"La génération audio a échoué : {res.get('error', 'erreur inconnue')}")
    return {"messages": [msg]}
 
 
def status_node(state: State) -> dict:
    if state.get("plan_progress"):
        return {"messages": [AIMessage(content=f"Production en cours : {state['plan_progress']}.")]}
    job_id = state.get("last_job_id")
    if not job_id:
        return {"messages": [AIMessage(content="Aucune vidéo en cours dans cette conversation.")]}
    job = get_video_status(job_id)
    if job["status"] == "completed" and job["video_url"]:
        msg = AIMessage(content="Ta vidéo est prête !", additional_kwargs={"video_url": job["video_url"]})
    elif job["status"] in ("failed", "cancelled", "expired"):
        msg = AIMessage(content=f"La génération a échoué ({job['status']}) : {job.get('error') or 'erreur inconnue'}")
    else:
        msg = AIMessage(content="Toujours en cours de génération, réessaie dans quelques instants.")
    return {"messages": [msg]}
 
 
builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("chat", chat_node)
builder.add_node("generate_video", video_node)
builder.add_node("generate_avatar_video", avatar_video_node)
builder.add_node("plan_video", planner_node)
builder.add_node("generate_audio", audio_node)
builder.add_node("video_status", status_node)
 
builder.add_edge(START, "orchestrator")
builder.add_conditional_edges(
    "orchestrator",
    lambda s: s["route"],
    {
        "chat": "chat",
        "generate_video": "generate_video",
        "generate_avatar_video": "generate_avatar_video",
        "plan_video": "plan_video",
        "generate_audio": "generate_audio",
        "video_status": "video_status",
    },
)
 
 
def _after_video(s: State) -> str:
    # Voix off séparée (TTS) enchaînée après la vidéo, sauf si le modèle la dit lui-même (NATIVE_VOICE)
    return "generate_audio" if s.get("with_audio") and not NATIVE_VOICE else "end"
 
 
for n in ("generate_video", "generate_avatar_video"):
    builder.add_conditional_edges(n, _after_video, {"generate_audio": "generate_audio", "end": END})
for n in ("chat", "plan_video", "generate_audio", "video_status"):
    builder.add_edge(n, END)
 
graph = builder.compile()