import json
import re
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

import file_parser
import embed_and_store
import knowledge_curator
from answer_with_rag import answer  # expects COMPLETIONS_MODEL to be valid (set CHAT_MODEL env var)

# ─────────────────────────────────────────────────────────────
# Paths (Cloud Run safe)
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("DATA_DIR", "/tmp/edbrainai"))
BASE_DIR.mkdir(parents=True, exist_ok=True)
HIST_PATH = BASE_DIR / "chat_history.json"
REFRESH_PATH = BASE_DIR / "last_refresh.txt"

# ─────────────────────────────────────────────────────────────
# App Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI CEO Assistant 🧠", page_icon="🧠", layout="wide")

USERNAME = st.secrets.get("app_user", "admin123")
PASSWORD = st.secrets.get("app_pass", "BestOrg123@#")

# ─────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────
def login():
    st.title("🔐 Login to AI CEO Assistant")
    with st.form("login_form"):
        u = st.text_input("👤 Username")
        p = st.text_input("🔑 Password", type="password")
        submitted = st.form_submit_button("➡️ Login")
        if submitted:
            if u == USERNAME and p == PASSWORD:
                st.session_state["authed"] = True
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

if "authed" not in st.session_state:
    st.session_state["authed"] = False

if not st.session_state["authed"]:
    login()
    st.stop()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_history():
    if HIST_PATH.exists():
        try:
            return json.loads(HIST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(hist):
    HIST_PATH.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

def save_reminder_local(content: str, title_hint: str = "") -> str:
    """
    Save a REMINDER block into DATA_DIR/reminders/*.txt
    """
    rem_dir = BASE_DIR / "reminders"
    rem_dir.mkdir(parents=True, exist_ok=True)

    # Basic title extraction
    m = re.search(r"^Title\s*:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
    title = (m.group(1).strip() if m else title_hint or f"note-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    # Sanitize filename
    fname = re.sub(r"[^A-Za-z0-9_\-]", "_", title.strip().replace(" ", "_"))[:60] or "untitled"
    path = rem_dir / f"{fname}.txt"
    if not content.strip().lower().startswith("title:"):
        content = f"Title: {title}\nTags: \nValidFrom: \nValidTo: \n\nBody:\n{content.strip()}\n"
    path.write_text(content, encoding="utf-8")
    return str(path)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    st.session_state.setdefault("limit_meetings", False)
    st.checkbox("Restrict answers to meetings", key="limit_meetings")
    if REFRESH_PATH.exists():
        st.caption(f"Last refresh: {REFRESH_PATH.read_text(encoding='utf-8')}")
    else:
        st.caption("Last refresh: never")

    st.markdown("---")
    st.subheader("🔁 Refresh Data")
    if st.button("Run File Parser + Embedder", use_container_width=True):
        with st.spinner("Parsing sources and building embeddings…"):
            file_parser.main()
            embed_and_store.main()
            knowledge_curator.main()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        REFRESH_PATH.write_text(now, encoding="utf-8")
        st.success("Refreshed successfully")

# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────
st.title("AI CEO Assistant 🧠")

tabs = st.tabs(["💬 New Chat", "🧾 Knowledge", "🧩 Curator", "⚡ Debug"])
with tabs[0]:
    history = load_history()
    user_msg = st.text_area("Type your message. To add knowledge, start with `REMINDER:`", height=150)
    if st.button("Send", type="primary"):
        ts = datetime.now().strftime("%b-%d-%Y %I:%M%p")
        reply = ""
        if user_msg.strip().lower().startswith("reminder:"):
            # store reminder locally
            body = user_msg.split(":", 1)[1].strip()
            fp = save_reminder_local(body, title_hint=f"reminder-{datetime.now().strftime('%H%M%S')}")
            st.success(f"Saved reminder → {fp}")
            reply = "Reminder saved. Click **Refresh Data** to re-embed."
        else:
            try:
                reply = answer(
                    user_msg,
                    k=7,
                    chat_history=history,
                    restrict_to_meetings=st.session_state["limit_meetings"],
                )
            except Exception as e:
                reply = f"Error: {e}"
        st.markdown(f"[{ts}]  \n{reply}")

        history.append({"role": "user", "content": user_msg, "timestamp": ts})
        history.append({"role": "assistant", "content": reply, "timestamp": ts})
        save_history(history)

    if history:
        with st.expander("Show chat history"):
            st.json(history)

with tabs[1]:
    st.subheader("Parsed Data")
    parsed_root = BASE_DIR / "parsed_data"
    rows = []
    if parsed_root.exists():
        for folder in sorted(parsed_root.iterdir()):
            if folder.is_dir():
                files = list(folder.rglob("*.txt"))
                rows.append([folder.name, len(files)])
    if rows:
        df = pd.DataFrame(rows, columns=["Folder", "Files"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No parsed data yet.")

with tabs[2]:
    st.subheader("Knowledge Registry (if built)")
    reg_csv = BASE_DIR / "embeddings" / "knowledge_registry.csv"
    if reg_csv.exists():
        st.dataframe(pd.read_csv(reg_csv), use_container_width=True)
    else:
        st.caption("Run Refresh to create knowledge registry.")

with tabs[3]:
    st.subheader("Paths")
    st.code(f"BASE_DIR: {BASE_DIR}\nHIST_PATH: {HIST_PATH}\nPARSED_DIR: {BASE_DIR/'parsed_data'}\nEMBED_DIR: {BASE_DIR/'embeddings'}", language="bash")
