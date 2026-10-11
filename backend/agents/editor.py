"""Décisions de montage : transitions entre scènes, volume du son d'origine sous la voix off.

Par défaut (EDITOR_AGENT=0) : tous les raccords utilisent TRANSITION (« cut » = coupe franche), sans appel LLM.
Avec EDITOR_AGENT=1 : un agent LLM choisit les transitions selon le contenu des scènes et le style demandé.
Sa réponse est validée (liste fermée de transitions, bornes de durée) : il ne produit jamais de commande FFmpeg.
"""
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from utils.ffmpeg_tools import TRANSITIONS
 
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, model_validator
from utils.config import get_conf

EDITOR_AGENT = bool(get_conf("editor_agent"))
DEFAULT_TRANSITION = get_conf("transition")
DEFAULT_TD = float(get_conf("transition_duration_s"))
DEFAULT_DUCK = float(get_conf("narration_ducking"))


llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL",""),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY", ""),
    temperature=0,
)

class _NullTolerant(BaseModel):
    """Classe de base qui ignore les champs inconnus et supprime les champs nulls avant validation."""
    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        return {k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else data

class EditDecision(_NullTolerant):
    transitions: list[str] = []          # un par raccord (n - 1), parmi TRANSITIONS
    transition_duration: float = DEFAULT_TD
    narration_ducking: float = DEFAULT_DUCK   # volume (0-1) du son d'origine sous la voix off


editor_llm = llm.with_structured_output(EditDecision, method="function_calling")

EDITOR_PROMPT = (
    "Tu es monteur vidéo. Pour chaque raccord entre deux scènes consécutives ({n} raccords), choisis une "
    "transition parmi : " + ", ".join(TRANSITIONS) + ".\n"
    "- cut : changement net, rythme dynamique ; fade / dissolve : enchaînement doux et fluide ; "
    "fadeblack : changement de lieu ou de moment marqué ; wipeleft / slideleft : rarement, effet graphique.\n"
    "Respecte le style demandé par l'utilisateur s'il est indiqué. transition_duration entre 0.3 et 1.2 s. "
    "narration_ducking = volume du son d'origine sous la voix off (0.1 à 0.5)."
)


def sanitize(d: dict, n_scenes: int, default: str = DEFAULT_TRANSITION) -> dict:
    """Valide une décision : transitions autorisées, une par raccord, durées bornées."""
    n = max(n_scenes - 1, 0)
    trans = [t if t in TRANSITIONS else DEFAULT_TRANSITION for t in (d.get("transitions") or [])][:n]
    trans += [default] * (n - len(trans))
    return {
        "transitions": trans,
        "transition_duration": min(max(float(d.get("transition_duration") or DEFAULT_TD), 0.2), 1.5),
        "narration_ducking": min(max(float(d.get("narration_ducking") if d.get("narration_ducking") is not None else DEFAULT_DUCK), 0.0), 1.0),
    }


def decide_edit(title: Optional[str], edit_style: Optional[str], scenes: list[dict]) -> dict:
    """scenes : [{"index", "prompt", "narration", "duration"}] dans l'ordre du montage."""
    smooth = any(k in (edit_style or "").lower() for k in ("fluid", "doux", "douce", "fondu", "smooth", "enchaîn", "enchain"))
    default = sanitize({}, len(scenes), "dissolve" if smooth and DEFAULT_TRANSITION == "cut" else DEFAULT_TRANSITION)
    if not EDITOR_AGENT or len(scenes) < 2:
        return default
    listing = "\n".join(f"{i}. ({s['duration']} s) {s['prompt'][:200]}" for i, s in enumerate(scenes, start=1))
    try:
        d = editor_llm.invoke([
            SystemMessage(content=EDITOR_PROMPT.format(n=len(scenes) - 1)),
            HumanMessage(content=f"Projet : {title or 'sans titre'}\nStyle demandé : {edit_style or 'non précisé'}\nScènes :\n{listing}"),
        ])
        return sanitize(d.model_dump(), len(scenes))
    except Exception:   # l'éditeur est un plus : en cas d'échec, montage par défaut
        return default
