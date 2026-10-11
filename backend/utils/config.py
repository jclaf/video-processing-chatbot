import json
import os
from pathlib import Path

# On pointe vers le dossier data à la racine du projet
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CONFIG_FILE = DATA_DIR / "config.json"

DEFAULT_CONFIG = {
    "max_scenes": int(os.getenv("MAX_SCENES", "10")),
    "max_parallel_scenes": int(os.getenv("MAX_PARALLEL_SCENES", "10")),
    "max_plan_cost_usd": float(os.getenv("MAX_PLAN_COST_USD", "2.0")),
    "scene_timeout_s": int(os.getenv("SCENE_TIMEOUT_S", "30")),
    "video_poll_interval_s": int(os.getenv("VIDEO_POLL_INTERVAL_S", "5")),
    "editor_agent": os.getenv("EDITOR_AGENT", "0") == "1",
    "transition": os.getenv("TRANSITION", "cut"),
    "transition_duration_s": float(os.getenv("TRANSITION_DURATION_S", "0.5")),
    "narration_ducking": float(os.getenv("NARRATION_DUCK", "0.25")),
    "video_resolution": os.getenv("VIDEO_RESOLUTION", "768p"),
    "video_aspect_ratio": os.getenv("VIDEO_ASPECT_RATIO", "9:16"),
    "native_voice": os.getenv("NATIVE_VOICE", "1") == "1",
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **data}
        except Exception:
            return DEFAULT_CONFIG
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_config(new_config: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(new_config, indent=4, ensure_ascii=False), encoding="utf-8")

def get_conf(key: str):
    return load_config().get(key, DEFAULT_CONFIG.get(key))