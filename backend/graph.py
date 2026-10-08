"""Graphe LangGraph :

START -> orchestrator -> chat | generate_video | generate_audio | video_status
                                   |
                                   +-- (si with_audio) --> generate_audio

La mémoire est persistée par thread_id (= id de conversation) via SqliteSaver.
"""
import os
import sqlite3
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from video_tool import create_video, get_video_status
from audio_tool import create_audio

DB_PATH = os.getenv("DB_PATH", "chatbot.db")
NATIVE_VOICE = os.getenv("NATIVE_VOICE", "0")

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY"),
    temperature=0,
)

# Connexion partagée (checkpoints LangGraph + table des conversations)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    route: str
    # vidéo
    video_prompt: Optional[str]
    duration: int
    last_job_id: Optional[str]
    # audio
    with_audio: bool
    audio_kind: str
    audio_text: Optional[str]


class Route(BaseModel):
    route: Literal["chat", "generate_video", "generate_audio", "video_status"]
    video_prompt: Optional[str] = None   # prompt visuel détaillé (anglais)
    duration: int = 5                    # secondes
    with_audio: bool = False             # vidéo + audio demandés ensemble
    audio_kind: Literal["voiceover", "music"] = "voiceover"
    audio_text: Optional[str] = None     # texte de la voix off, ou prompt de la musique


router_llm = llm.with_structured_output(Route, method="function_calling")

ROUTER_PROMPT = SystemMessage(content=(
    "Tu es l'orchestrateur d'un assistant de création de vidéos et d'audio. Choisis la route :\n"
    "- generate_video : créer/modifier une vidéo. Remplis video_prompt (prompt visuel détaillé, "
    "en anglais) et duration. Si l'utilisateur veut aussi une voix off ou une musique, mets "
    "with_audio=true et remplis audio_kind + audio_text.\n"
    "- generate_audio : audio seul (voix off ou musique). audio_kind='voiceover' avec audio_text = "
    "le texte exact à dire, ou 'music' avec audio_text = description du style.\n"
    "- video_status : demande l'état / le résultat d'une vidéo.\n"
    "- chat : tout le reste (idées, questions, précisions à demander avant de générer)."
))

CHAT_PROMPT = SystemMessage(content=(
    "Tu es un assistant créatif qui aide à concevoir des vidéos et leur audio (voix off, musique). "
    "Réponds en français, pose des questions courtes si le brief est flou (sujet, style, durée, voix)."
))


def _last_human(state: State) -> str:
    return next(m.content for m in reversed(state["messages"]) if m.type == "human")


def orchestrator(state: State) -> dict:
    r = router_llm.invoke([ROUTER_PROMPT] + state["messages"][-10:])
    return {
        "route": r.route,
        "video_prompt": r.video_prompt,
        "duration": r.duration,
        "with_audio": r.with_audio,
        "audio_kind": r.audio_kind,
        "audio_text": r.audio_text,
    }


def chat_node(state: State) -> dict:
    return {"messages": [llm.invoke([CHAT_PROMPT] + state["messages"][-20:])]}


def video_node(state: State) -> dict:
    prompt = state.get("video_prompt") or _last_human(state)
    # Voix intégrée à la vidéo (modèles qui génèrent le son nativement)
    if NATIVE_VOICE and state.get("with_audio") and state.get("audio_kind") == "voiceover" and state.get("audio_text"):
        prompt += f' A calm, energetic narrator speaks in French: "{state["audio_text"]}"'
    job = create_video(prompt, state.get("duration", 5))
    done = job["status"] == "completed" and job["video_url"]
    text = "Voici ta vidéo !" if done else f"Génération lancée (job `{job['job_id']}`). Demande-moi l'état quand tu veux."
    msg = AIMessage(content=text, additional_kwargs={"video_url": job["video_url"], "job_id": job["job_id"]})
    return {"messages": [msg], "last_job_id": job["job_id"]}


def audio_node(state: State) -> dict:
    kind = state.get("audio_kind", "voiceover")
    text = state.get("audio_text") or state.get("video_prompt") or _last_human(state)
    res = create_audio(kind, text, state.get("duration"))
    label = "ta voix off" if kind == "voiceover" else "ta musique"
    if res["status"] == "completed" and res["audio_url"]:
        msg = AIMessage(content=f"Voici {label} !", additional_kwargs={"audio_url": res["audio_url"]})
    else:
        msg = AIMessage(content=f"La génération audio n'a pas abouti (statut : {res['status']}).")
    return {"messages": [msg]}


def status_node(state: State) -> dict:
    job_id = state.get("last_job_id")
    if not job_id:
        return {"messages": [AIMessage(content="Aucune vidéo en cours dans cette conversation.")]}
    job = get_video_status(job_id)
    if job["status"] == "completed" and job["video_url"]:
        msg = AIMessage(content="Ta vidéo est prête !", additional_kwargs={"video_url": job["video_url"]})
    else:
        msg = AIMessage(content=f"Statut du job `{job_id}` : {job['status']}.")
    return {"messages": [msg]}


builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("chat", chat_node)
builder.add_node("generate_video", video_node)
builder.add_node("generate_audio", audio_node)
builder.add_node("video_status", status_node)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges(
    "orchestrator",
    lambda s: s["route"],
    {
        "chat": "chat",
        "generate_video": "generate_video",
        "generate_audio": "generate_audio",
        "video_status": "video_status",
    },
)
# Après la vidéo : enchaîne sur l'audio si demandé
builder.add_conditional_edges(
    "generate_video",
    lambda s: "generate_audio" if s.get("with_audio") and not NATIVE_VOICE else "end",
    {"generate_audio": "generate_audio", "end": END},
)
for n in ("chat", "generate_audio", "video_status"):
    builder.add_edge(n, END)

graph = builder.compile(checkpointer=checkpointer)
