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


# ---------- conversations ----------
def create_conversation(conv_id: str) -> None:
    _run("INSERT INTO conversations VALUES (?, ?, ?, NULL)", (conv_id, DEFAULT_TITLE, time.time()))


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


def delete_conversation(conv_id: str) -> None:
    _run("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
    _run("DELETE FROM conversations WHERE id = ?", (conv_id,))


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
        if role == "human":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content, additional_kwargs=json.loads(extra or "{}")))
    return out
