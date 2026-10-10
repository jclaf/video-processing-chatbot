import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Video Chatbot", page_icon="🎬", layout="wide")

IMAGE_MODES = {
    "Animer l'image (1re = début, 2e = fin)": "first_frame",
    "Image(s) de référence (style / contenu)": "reference",
}

def api(method: str, path: str, **kw):
    r = requests.request(method, f"{API}{path}", timeout=300, **kw)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"Erreur API {r.status_code} : {detail}")
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
    st.divider()
    mode_label = st.radio("Si tu joins des images", list(IMAGE_MODES), key="image_mode_label")
    st.caption("Pour garder un personnage d'une vidéo à l'autre, joins son image et écris « avec cet avatar ».")
 
    if st.session_state.get("conv_id"):
        av = api("GET", f"/conversations/{st.session_state.conv_id}/avatar")
        if av["image_urls"]:
            st.divider()
            st.subheader("Avatar de la conversation")
            st.image(av["image_urls"], width=100)
            if av["description"]:
                st.caption(av["description"])
            if st.button("Retirer l'avatar", use_container_width=True):
                api("DELETE", f"/conversations/{st.session_state.conv_id}/avatar")
                st.rerun()
                
# --- Zone de chat ---
if "conv_id" not in st.session_state:
    open_conversation(api("POST", "/conversations")["id"])

st.header("Créateur de vidéos")


def render(m: dict):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("image_urls"):
            st.image(m["image_urls"], width=160)
        if m.get("video_url"):
            st.video(m["video_url"])
        if m.get("audio_url"):
            st.audio(m["audio_url"])


for m in st.session_state.messages:
    render(m)

# --- Plan multi-scènes : validation du coût puis suivi de la production ---
STATUS_ICON = {"pending": "🕓", "submitted": "⏳", "completed": "✅", "failed": "❌"}

def cost_text(plan: dict) -> str:
    low, high = plan.get("est_low"), plan.get("est_high")
    if low is None or high is None:
        return "Coût non estimable"
    if high == 0:
        return "Coût estimé : 0 $ (mode mock)"
    if low == high:
        return f"Coût estimé : ≈ {low:.2f} $"
    return f"Coût estimé : entre {low:.2f} et {high:.2f} $"

try:
    _first = api("GET", f"/conversations/{st.session_state.conv_id}/plan")
except RuntimeError:
    _first = {"status": None}
_poll = 5 if _first.get("status") == "running" else None   # on ne sonde l'API que pendant une production

@st.fragment(run_every=_poll)
def plan_panel():
    conv_id = st.session_state.conv_id
    plan = api("GET", f"/conversations/{conv_id}/plan")
    status = plan.get("status")
 
    # Production terminée : recharge la conversation (les scènes y ont été ajoutées)
    if st.session_state.get("watch_plan") == plan.get("id") and status in ("done", "failed"):
        st.session_state.pop("watch_plan")
        st.session_state.messages = api("GET", f"/conversations/{conv_id}/messages")
        st.rerun()
 
    if status == "awaiting_approval":
        with st.container(border=True):
            st.markdown(f"**Plan en attente de validation** : {plan['total']} scène(s)")
            st.caption(cost_text(plan))
            c1, c2 = st.columns(2)
            if c1.button("✅ Lancer la production", type="primary", use_container_width=True, key="plan_ok"):
                try:
                    api("POST", f"/conversations/{conv_id}/plan/approve")
                    st.session_state.watch_plan = plan["id"]
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))
            if c2.button("❌ Annuler", use_container_width=True, key="plan_cancel"):
                api("POST", f"/conversations/{conv_id}/plan/cancel")
                st.rerun()
    elif status == "running":
        st.session_state.watch_plan = plan["id"]
        with st.container(border=True):
            st.markdown(f"**Production en cours** : {plan['done']}/{plan['total']} scène(s) terminée(s)")
            st.progress(plan["done"] / max(plan["total"], 1))
            st.caption("  ".join(f"{STATUS_ICON.get(sc['status'], '⏳')} {sc['index']}" for sc in plan["scenes"]))
 
 
plan_panel()

prompt = st.chat_input("Décris ta vidéo, ta voix off ou ta musique…", accept_file="multiple", file_type=["png", "jpg", "pdf", "csv"])

if prompt:
    text = (prompt.get("text") or "").strip()
    files = prompt.get("files") or []
    if not text:
        st.warning("Ajoute un texte pour décrire ta demande (avec ou sans images).")
    
    user_msg = {"role": "user", "content": text,"image_urls": [f.getvalue() for f in files]}

    st.session_state.messages.append(user_msg)
    render(user_msg)
    try:
        with st.spinner("Je réfléchis…"):
            reply = api(
                "POST",
                f"/conversations/{st.session_state.conv_id}/chat",
                data={"message": text, "image_mode": IMAGE_MODES[mode_label]},
                files=[("images", (f.name, f.getvalue(), f.type)) for f in files],
            )
        st.session_state.messages.extend(reply["messages"])
    except RuntimeError as e:
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}"})
    st.rerun()