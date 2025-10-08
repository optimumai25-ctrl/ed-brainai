import os
import io
import csv
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import docx

# ─────────────────────────────────────────────────────────────
# Paths (Cloud Run safe)
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("DATA_DIR", "/tmp/edbrainai"))
OUTPUT_DIR = BASE_DIR / "parsed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOWTEXT_LOG = OUTPUT_DIR / "low_text_files.csv"

# Optional: Google Drive — only if present in Streamlit secrets
try:
    GDRIVE_CONFIG = st.secrets.get("gdrive", None) if hasattr(st, "secrets") else None
except Exception:
    GDRIVE_CONFIG = None

try:
    if GDRIVE_CONFIG:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from google.oauth2 import service_account

        SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = service_account.Credentials.from_service_account_info(GDRIVE_CONFIG, scopes=SCOPES)
        drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    else:
        drive_service = None
except Exception:
    drive_service = None

KB_FOLDER_NAME = "AI_CEO_KnowledgeBase"
REMINDERS_FOLDER_NAME = "AI_CEO_Reminders"

# ─────────────────────────────────────────────────────────────
# Local reminders (BASE_DIR/reminders/*.txt)
# ─────────────────────────────────────────────────────────────
def _write_parsed(folder: str, filename: str, text: str):
    out_dir = OUTPUT_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{Path(filename).stem}.txt").write_text(text, encoding="utf-8")

def parse_local_reminders():
    rem_dir = BASE_DIR / "reminders"
    if not rem_dir.exists():
        return
    for fp in rem_dir.glob("*.txt"):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        _write_parsed("Reminders", fp.name, text)

# ─────────────────────────────────────────────────────────────
# File text extractors
# ─────────────────────────────────────────────────────────────
def _from_pdf_pymupdf(path: Path) -> str:
    try:
        doc = fitz.open(path)
        txt = []
        for page in doc:
            txt.append(page.get_text())
        return "\n".join(txt)
    except Exception:
        return ""

def _from_pdf_pypdf2(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        out = []
        for page in reader.pages:
            out.append(page.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""

def _from_docx(path: Path) -> str:
    try:
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""

def _from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(errors="ignore")

def extract_text(path: Path) -> str:
    p = path
    if p.suffix.lower() == ".pdf":
        t = _from_pdf_pymupdf(p) or _from_pdf_pypdf2(p)
    elif p.suffix.lower() in {".docx"}:
        t = _from_docx(p)
    elif p.suffix.lower() in {".txt", ".md"}:
        t = _from_txt(p)
    else:
        t = ""
    return t.strip()

# ─────────────────────────────────────────────────────────────
# Google Drive helpers (only used if drive_service exists)
# ─────────────────────────────────────────────────────────────
def _find_folder_id_by_name(name: str) -> Optional[str]:
    if not drive_service:
        return None
    q = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    res = drive_service.files().list(q=q, fields="files(id, name)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def _list_files_in_folder(folder_id: str):
    if not drive_service:
        return []
    q = f"'{folder_id}' in parents and trashed = false"
    res = drive_service.files().list(q=q, fields="files(id, name, mimeType)").execute()
    return res.get("files", [])

def _download_file(file_id: str, filename: str) -> bytes:
    if not drive_service:
        return b""
    request = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return buf.getvalue()

def _extract_from_bytes(name: str, data: bytes) -> str:
    tmp = BASE_DIR / "tmp_dl"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / name
    p.write_bytes(data)
    text = extract_text(p)
    try:
        p.unlink()
    except Exception:
        pass
    return text

def parse_drive():
    if not drive_service:
        return {"drive": "not-configured"}
    low_rows = []
    folders = {
        "KnowledgeBase": KB_FOLDER_NAME,
        "Reminders": REMINDERS_FOLDER_NAME,
    }
    for label, fname in folders.items():
        fid = _find_folder_id_by_name(fname)
        if not fid:
            continue
        files = _list_files_in_folder(fid)
        for f in files:
            name = f["name"]
            if not any(name.lower().endswith(suf) for suf in (".pdf", ".docx", ".txt", ".md")):
                continue
            data = _download_file(f["id"], name)
            text = _extract_from_bytes(name, data)
            if len(text) < 50:
                low_rows.append([label, name, "LOW_TEXT"])
            _write_parsed(label, name, text)
    if low_rows:
        with LOWTEXT_LOG.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["folder", "file", "flag"]] + low_rows)

# ─────────────────────────────────────────────────────────────
# CLI/entry (used by Streamlit "Refresh Data" button)
# ─────────────────────────────────────────────────────────────
def main():
    parse_local_reminders()
    parse_drive()
    # Parse any local uploaded/source docs under BASE_DIR/source (optional)
    local_src = BASE_DIR / "source"
    if local_src.exists():
        for fp in local_src.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in (".pdf", ".docx", ".txt", ".md"):
                _write_parsed("KnowledgeBase", fp.name, extract_text(fp))
    return {"ok": True, "output": str(OUTPUT_DIR)}

if __name__ == "__main__":
    main()
