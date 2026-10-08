import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Video Chatbot", page_icon="🎬", layout="wide")


def api(method: str, path: str, **kw):
    r = requests.request(method, f"{API}{path}", timeout=300, **kw)
    r.raise_for_status()
    return r.json()


def open_conversation(conv_id: str):
    st.session_state.conv_id = conv_id
    st.session_state.messages = api("GET", f"/conversations/{conv_id}/messages")


# --- Sidebar : dernières conversations ---
with st.sidebar:
    st.title("🎬 Conversations")
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        open_conversation(api("POST", "/conversations")["id"])
        st.rerun()
    st.divider()
    for c in api("GET", "/conversations"):
        col1, col2 = st.columns([5, 1])
        active = c["id"] == st.session_state.get("conv_id")
        if col1.button(c["title"], key=f"open_{c['id']}", use_container_width=True,
                       type="primary" if active else "secondary"):
            open_conversation(c["id"])
            st.rerun()
        if col2.button("🗑", key=f"del_{c['id']}"):
            api("DELETE", f"/conversations/{c['id']}")
            if active:
                st.session_state.pop("conv_id", None)
                st.session_state.pop("messages", None)
            st.rerun()

# --- Zone de chat ---
if "conv_id" not in st.session_state:
    open_conversation(api("POST", "/conversations")["id"])

st.header("Créateur de vidéos")


def render(m: dict):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("video_url"):
            st.video(m["video_url"])
        if m.get("audio_url"):
            st.audio(m["audio_url"])


for m in st.session_state.messages:
    render(m)

prompt = st.chat_input("Décris ta vidéo, ta voix off ou ta musique…", accept_file="multiple", file_type=["png", "jpg", "pdf", "csv"])

if prompt and prompt.text:
    text_part = prompt.text.strip() if prompt.text else ""
    file_names = [f.name for f in prompt.files] if prompt.files else []
    
    if text_part and file_names:
        display_text = f"{text_part}\n\n📎 **Fichiers joints ({len(file_names)}) :** " + ", ".join(file_names)
    elif text_part:
        display_text = text_part
    
    user_msg = {"role": "user", "content": display_text}

    st.session_state.messages.append(user_msg)
    render(user_msg)
    with st.spinner("Je réfléchis…"):
        payload = {
            "message": text_part, 
            "has_files": len(file_names) > 0, 
            "filenames": file_names
            }
        reply = api("POST", f"/conversations/{st.session_state.conv_id}/chat", json=payload)
    st.session_state.messages.extend(reply["messages"])
    st.rerun()
elif prompt and prompt.files and not (prompt.text and prompt.text.strip()):
    # Optionnel : Afficher un avertissement si l'utilisateur essaie d'envoyer uniquement des fichiers
    st.warning("Veuillez ajouter un texte pour décrire votre fichier ou votre demande.")