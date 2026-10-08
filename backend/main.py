import threading
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph import graph, conn, checkpointer
from media import MEDIA_DIR

app = FastAPI(title="Video Chatbot API")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")  # audio générés
lock = threading.Lock()

with lock:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "id TEXT PRIMARY KEY, title TEXT, updated_at REAL)"
    )
    conn.commit()


class ChatIn(BaseModel):
    message: str


def _exists(conv_id: str) -> bool:
    with lock:
        return conn.execute("SELECT 1 FROM conversations WHERE id=?", (conv_id,)).fetchone() is not None


def _serialize(m) -> dict:
    return {
        "role": "user" if m.type == "human" else "assistant",
        "content": m.content if isinstance(m.content, str) else str(m.content),
        "video_url": m.additional_kwargs.get("video_url"),
        "audio_url": m.additional_kwargs.get("audio_url"),
    }


@app.get("/conversations")
def list_conversations(limit: int = 30):
    with lock:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


@app.post("/conversations")
def new_conversation():
    conv_id = str(uuid.uuid4())
    with lock:
        conn.execute("INSERT INTO conversations VALUES (?,?,?)", (conv_id, "Nouvelle conversation", time.time()))
        conn.commit()
    return {"id": conv_id, "title": "Nouvelle conversation"}


@app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str):
    if not _exists(conv_id):
        raise HTTPException(404, "Conversation introuvable")
    state = graph.get_state({"configurable": {"thread_id": conv_id}})
    return [_serialize(m) for m in state.values.get("messages", [])]


@app.post("/conversations/{conv_id}/chat")
def chat(conv_id: str, body: ChatIn):
    if not _exists(conv_id):
        raise HTTPException(404, "Conversation introuvable")
    config = {"configurable": {"thread_id": conv_id}}
    result = graph.invoke({"messages": [HumanMessage(content=body.message)]}, config)

    # Tous les messages produits pendant ce tour (ex. vidéo + audio)
    msgs = result["messages"]
    last_human = max(i for i, m in enumerate(msgs) if m.type == "human")
    new_msgs = [_serialize(m) for m in msgs[last_human + 1:]]

    with lock:  # titre = début du 1er message
        conn.execute(
            "UPDATE conversations SET updated_at=?, title=CASE WHEN title='Nouvelle conversation' THEN ? ELSE title END WHERE id=?",
            (time.time(), body.message[:50], conv_id),
        )
        conn.commit()

    return {"messages": new_msgs}


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    with lock:
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        conn.commit()
    try:
        checkpointer.delete_thread(conv_id)
    except Exception:
        pass
    return {"deleted": conv_id}
