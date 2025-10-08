import os
import csv
import re
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import faiss
from tqdm import tqdm

from chunk_utils import simple_chunks

# ─────────────────────────────────────────────────────────────
# Paths (Cloud Run safe)
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("DATA_DIR", "/tmp/edbrainai"))
PARSED_DIR = BASE_DIR / "parsed_data"
EMBED_DIR = BASE_DIR / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
INDEX_PATH = EMBED_DIR / "faiss.index"
META_PATH = EMBED_DIR / "metadata.pkl"
REPORT_CSV = EMBED_DIR / "embedding_report.csv"
REG_CSV = EMBED_DIR / "knowledge_registry.csv"  # optional, built by knowledge_curator

# ─────────────────────────────────────────────────────────────
# OpenAI embedding client (new SDK first, legacy fallback)
# ─────────────────────────────────────────────────────────────
_USE_CLIENT = False
try:
    from openai import OpenAI
    _client = OpenAI()
    _USE_CLIENT = True
except Exception:
    _client = None
    try:
        import openai  # type: ignore
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except Exception:
        openai = None

# FAISS index (ID-mapped)
EMBED_DIM = 1536  # text-embedding-3-* is 1536
_base = faiss.IndexFlatL2(EMBED_DIM)
_index = faiss.IndexIDMap2(_base)
_metadata: Dict[int, Dict] = {}
_next_id = 1_000_000  # avoid collisions; simple increasing id


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        try:
            return p.read_text(errors="ignore")
        except Exception:
            return ""

def _extract_header_fields(text: str) -> Dict[str, Optional[str]]:
    """
    Extracts optional metadata from the top of a reminder/knowledge file.
    Accepts lines like:
      Title: ...
      Tags: tag1, tag2
      MeetingDate: YYYY-MM-DD
      ValidFrom: YYYY-MM-DD
      ValidTo: YYYY-MM-DD
    """
    def _grab(label: str) -> Optional[str]:
        m = re.search(rf"^{label}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    return {
        "title": _grab("Title"),
        "tags": _grab("Tags"),
        "meeting_date": _grab("MeetingDate"),
        "valid_from": _grab("ValidFrom"),
        "valid_to": _grab("ValidTo"),
    }

def _embed_chunk(chunk: str) -> Optional[np.ndarray]:
    try:
        if _USE_CLIENT:
            vec = _client.embeddings.create(model=EMBED_MODEL, input=chunk).data[0].embedding
        else:
            assert openai is not None, "OpenAI SDK not available and no API key set."
            vec = openai.Embedding.create(model=EMBED_MODEL, input=chunk)["data"][0]["embedding"]  # type: ignore
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[0] != EMBED_DIM:
            # In case model dim changes
            raise RuntimeError(f"Unexpected embedding dim: {arr.shape[0]} != {EMBED_DIM}")
        return arr
    except Exception as e:
        print(f"[embed] failed: {e}")
        return None

def _add_vector(vec: np.ndarray, meta: Dict):
    global _next_id
    vid = _next_id
    _next_id += 1
    _index.add_with_ids(vec.reshape(1, -1), np.array([vid], dtype=np.int64))
    _metadata[vid] = meta

def _iter_source_files() -> List[Tuple[Path, str]]:
    """
    Returns list of (path, folder_name) to embed.
    We only embed .txt files produced by file_parser into PARSED_DIR.
    """
    out = []
    if not PARSED_DIR.exists():
        return out
    for folder in sorted(PARSED_DIR.iterdir()):
        if not folder.is_dir():
            continue
        for fp in folder.rglob("*.txt"):
            out.append((fp, folder.name))
    return out


# ─────────────────────────────────────────────────────────────
# Main embed pipeline
# ─────────────────────────────────────────────────────────────
def main() -> Dict:
    files = _iter_source_files()
    if not files:
        print(f"[embed] No parsed files found in {PARSED_DIR}.")
        return {"embedded": 0, "files": 0}

    report_rows = [["file", "folder", "num_chunks", "chars", "title", "tags", "meeting_date", "valid_from", "valid_to"]]
    total_chunks = 0

    for fp, folder_name in tqdm(files, desc="Embedding"):
        text = _read_text_file(fp)
        header = _extract_header_fields(text)
        # chunk
        chunks = simple_chunks(text, max_tokens=400, overlap=50)
        good = []
        for i, ch in enumerate(chunks):
            vec = _embed_chunk(ch)
            if vec is None:
                continue
            meta = {
                "file": fp.name,
                "rel_path": str(fp.relative_to(BASE_DIR)),
                "folder": folder_name,
                "chunk_index": i,
                "title": header["title"],
                "tags": header["tags"],
                "meeting_date": header["meeting_date"],
                "valid_from": header["valid_from"],
                "valid_to": header["valid_to"],
                "chars": len(ch),
            }
            _add_vector(vec, meta)
            good.append(ch)

        report_rows.append([
            fp.name, folder_name, len(good), len(text),
            header["title"] or "", header["tags"] or "",
            header["meeting_date"] or "", header["valid_from"] or "", header["valid_to"] or ""
        ])
        total_chunks += len(good)

    # persist index + metadata + report
    faiss.write_index(_index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(report_rows)

    print(f"[embed] Saved index: {INDEX_PATH}")
    print(f"[embed] Saved metadata: {META_PATH} ({len(_metadata)} vectors)")
    print(f"[embed] Report: {REPORT_CSV}")
    return {"embedded": total_chunks, "files": len(files)}

if __name__ == "__main__":
    main()

