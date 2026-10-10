"""Persistance SQLite : conversations + messages (mémoire du chatbot).

Même interface que la version DuckDB : main.py et graph.py n'ont pas à changer.
"""
import json
import os
import sqlite3
import threading
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

DB_PATH = os.getenv("DB_PATH", "chatbot.db")
DEFAULT_TITLE = "Nouvelle conversation"

_con = sqlite3.connect(DB_PATH, check_same_thread=False)
_con.execute("PRAGMA journal_mode=WAL")
_lock = threading.Lock()


def _run(sql: str, params: tuple | list = ()) -> None:
    with _lock:
        _con.execute(sql, params)
        _con.commit()


def _query(sql: str, params: tuple | list = ()) -> list:
    with _lock:
        return _con.execute(sql, params).fetchall()


_run("""CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    updated_at REAL,
    last_job_id TEXT
)""")
_run("""CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id TEXT NOT NULL,
    role TEXT,
    content TEXT,
    extra TEXT,
    created_at REAL
)""")
_run("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conv_id, id)")


_run("""CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL,
    status TEXT,
    title TEXT,
    scenes TEXT,
    est_low REAL,
    est_high REAL,
    use_avatar INTEGER DEFAULT 0,
    avatar_desc TEXT,
    created_at REAL
)""")
_run("""CREATE TABLE IF NOT EXISTS scene_jobs (
    plan_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    status TEXT,
    job_id TEXT,
    video_url TEXT,
    error TEXT,
    PRIMARY KEY (plan_id, idx)
)""")

# Migration : colonnes ajoutées au fil des versions (base créée par une ancienne version du projet)
_cols = {r[1] for r in _query("PRAGMA table_info(conversations)")}
for _col in ("last_job_id", "avatar_files", "avatar_desc"):
    if _col not in _cols:
        _run(f"ALTER TABLE conversations ADD COLUMN {_col} TEXT")


# ---------- conversations ----------
def create_conversation(conv_id: str) -> None:
    _run("INSERT INTO conversations (id, title, updated_at) VALUES (?, ?, ?)", (conv_id, DEFAULT_TITLE, time.time()))


def exists(conv_id: str) -> bool:
    return bool(_query("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)))


def list_conversations(limit: int = 30) -> list[dict]:
    rows = _query("SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,))
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def get_last_job_id(conv_id: str) -> str | None:
    rows = _query("SELECT last_job_id FROM conversations WHERE id = ?", (conv_id,))
    return rows[0][0] if rows else None


def touch(conv_id: str, first_text: str, last_job_id: str | None) -> None:
    """Met à jour la date, le titre (au 1er message) et le dernier job vidéo."""
    _run(
        "UPDATE conversations SET updated_at = ?, last_job_id = COALESCE(?, last_job_id), "
        "title = CASE WHEN title = ? THEN ? ELSE title END WHERE id = ?",
        (time.time(), last_job_id, DEFAULT_TITLE, first_text[:50], conv_id),
    )


# ---------- avatar de la conversation ----------
def get_avatar(conv_id: str) -> dict:
    """{"files": [noms de fichiers dans MEDIA_DIR], "desc": description ou None}"""
    rows = _query("SELECT avatar_files, avatar_desc FROM conversations WHERE id = ?", (conv_id,))
    if not rows:
        return {"files": [], "desc": None}
    return {"files": json.loads(rows[0][0]) if rows[0][0] else [], "desc": rows[0][1]}


def set_avatar(conv_id: str, files: list[str] | None = None, desc: str | None = None) -> None:
    """Met à jour seulement ce qui est fourni (None = inchangé)."""
    if files is not None:
        _run("UPDATE conversations SET avatar_files = ? WHERE id = ?", (json.dumps(files), conv_id))
    if desc is not None:
        _run("UPDATE conversations SET avatar_desc = ? WHERE id = ?", (desc, conv_id))


def clear_avatar(conv_id: str) -> None:
    _run("UPDATE conversations SET avatar_files = NULL, avatar_desc = NULL WHERE id = ?", (conv_id,))


def delete_conversation(conv_id: str) -> None:
    _run("DELETE FROM scene_jobs WHERE plan_id IN (SELECT id FROM plans WHERE conv_id = ?)", (conv_id,))
    _run("DELETE FROM plans WHERE conv_id = ?", (conv_id,))
    _run("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
    _run("DELETE FROM conversations WHERE id = ?", (conv_id,))


# ---------- plans multi-scènes ----------
# statuts : awaiting_approval -> running -> done | failed   (ou cancelled)
_PLAN_COLS = "id, conv_id, status, title, scenes, est_low, est_high, use_avatar, avatar_desc, created_at"


def _plan(r) -> dict:
    return {
        "id": r[0], "conv_id": r[1], "status": r[2], "title": r[3], "scenes": json.loads(r[4] or "[]"),
        "est_low": r[5], "est_high": r[6], "use_avatar": bool(r[7]), "avatar_desc": r[8], "created_at": r[9],
    }


def create_plan(plan_id: str, conv_id: str, title: str, scenes: list[dict], est_low: float | None,
                est_high: float | None, use_avatar: bool, avatar_desc: str | None) -> None:
    """Crée un plan en attente de validation (remplace le plan précédemment en attente)."""
    _run("UPDATE plans SET status = 'cancelled' WHERE conv_id = ? AND status = 'awaiting_approval'", (conv_id,))
    _run(
        f"INSERT INTO plans ({_PLAN_COLS}) VALUES (?, ?, 'awaiting_approval', ?, ?, ?, ?, ?, ?, ?)",
        (plan_id, conv_id, title, json.dumps(scenes, ensure_ascii=False), est_low, est_high,
         int(use_avatar), avatar_desc, time.time()),
    )


def get_plan(plan_id: str) -> dict | None:
    rows = _query(f"SELECT {_PLAN_COLS} FROM plans WHERE id = ?", (plan_id,))
    return _plan(rows[0]) if rows else None


def get_latest_plan(conv_id: str) -> dict | None:
    rows = _query(f"SELECT {_PLAN_COLS} FROM plans WHERE conv_id = ? ORDER BY created_at DESC LIMIT 1", (conv_id,))
    return _plan(rows[0]) if rows else None


def get_plan_by_status(conv_id: str, status: str) -> dict | None:
    rows = _query(
        f"SELECT {_PLAN_COLS} FROM plans WHERE conv_id = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
        (conv_id, status),
    )
    return _plan(rows[0]) if rows else None


def list_plans_by_status(status: str) -> list[dict]:
    return [_plan(r) for r in _query(f"SELECT {_PLAN_COLS} FROM plans WHERE status = ?", (status,))]


def set_plan_status(plan_id: str, status: str) -> None:
    _run("UPDATE plans SET status = ? WHERE id = ?", (status, plan_id))


_SCENE_FIELDS = ("status", "job_id", "video_url", "error")


def get_scene(plan_id: str, idx: int) -> dict | None:
    rows = _query("SELECT idx, status, job_id, video_url, error FROM scene_jobs WHERE plan_id = ? AND idx = ?", (plan_id, idx))
    return dict(zip(("idx", "status", "job_id", "video_url", "error"), rows[0])) if rows else None


def list_scenes(plan_id: str) -> list[dict]:
    rows = _query("SELECT idx, status, job_id, video_url, error FROM scene_jobs WHERE plan_id = ? ORDER BY idx", (plan_id,))
    return [dict(zip(("idx", "status", "job_id", "video_url", "error"), r)) for r in rows]


def upsert_scene(plan_id: str, idx: int, **fields) -> None:
    """Crée la ligne de scène si besoin, puis met à jour les champs fournis."""
    assert set(fields) <= set(_SCENE_FIELDS), fields
    _run("INSERT OR IGNORE INTO scene_jobs (plan_id, idx) VALUES (?, ?)", (plan_id, idx))
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        _run(f"UPDATE scene_jobs SET {sets} WHERE plan_id = ? AND idx = ?", (*fields.values(), plan_id, idx))


# ---------- messages ----------
def save_messages(conv_id: str, msgs: list[BaseMessage]) -> None:
    now = time.time()
    rows = [
        (
            conv_id,
            m.type,
            m.content if isinstance(m.content, str) else json.dumps(m.content),
            json.dumps(m.additional_kwargs),
            now + i * 1e-3,
        )
        for i, m in enumerate(msgs)
    ]
    with _lock:
        _con.executemany(
            "INSERT INTO messages (conv_id, role, content, extra, created_at) VALUES (?, ?, ?, ?, ?)", rows
        )
        _con.commit()


def load_messages(conv_id: str) -> list[BaseMessage]:
    rows = _query("SELECT role, content, extra FROM messages WHERE conv_id = ? ORDER BY id", (conv_id,))
    out: list[BaseMessage] = []
    for role, content, extra in rows:
        kwargs = json.loads(extra or "{}")
        if role == "human":
            out.append(HumanMessage(content=content, additional_kwargs=kwargs))
        else:
            out.append(AIMessage(content=content, additional_kwargs=kwargs))
    return out