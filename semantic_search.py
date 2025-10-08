# semantic_search.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os, pickle, numpy as np

# Cloud Run has a read-only filesystem except /tmp, so we default there
BASE_DIR = Path(os.getenv("DATA_DIR", "/tmp/edbrainai"))
EMBED_DIR = BASE_DIR / "embeddings"
INDEX_PATH = EMBED_DIR / "faiss.index"
META_PATH = EMBED_DIR / "metadata.pkl"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Prefer the modern OpenAI SDK, fall back to legacy if not available
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _client = None
    _use_client = False

_index = None
_metadata: Dict[int, Dict] = {}

def _load_index():
    """Lazy-load FAISS index + metadata."""
    global _index, _metadata
    if _index is not None:
        return
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return
    import faiss
    _index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    _metadata = {int(k): v for k, v in meta.items()}

def _embed(text: str) -> Optional[np.ndarray]:
    try:
        if _use_client:
            vec = _client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
        else:
            vec = openai.Embedding.create(model=EMBED_MODEL, input=text)["data"][0]["embedding"]  # type: ignore
        return np.asarray(vec, dtype=np.float32)
    except Exception:
        return None

def _search_vec(vec: np.ndarray, k: int) -> List[Tuple[int, float, Dict]]:
    if vec is None:
        return []
    if _index is None:
        return []
    import faiss
    D, I = _index.search(vec.reshape(1, -1).astype(np.float32), k)
    out: List[Tuple[int, float, Dict]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        meta = _metadata.get(int(idx), {})
        score = float(1.0 / (1.0 + dist))  # convert L2 distance → similarity-ish score
        out.append((int(idx), score, meta))
    return out

def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    _load_index()
    qv = _embed(query)
    return _search_vec(qv, k)

def search_meetings(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    hits = search(query, k=20)
    mtg = [h for h in hits if str(h[2].get("folder", "")).lower() == "meetings"]
    return mtg[:k] if mtg else hits[:k]

# Optional helper used by your code; keeps same signature
def search_in_date_window(query: str, start, end, k: int = 5) -> List[Tuple[int, float, Dict]]:
    from datetime import datetime
    def _parse_date(s: Optional[str]) -> Optional[datetime]:
        if not s: return None
        for fmt in ("%Y-%m-%d","%Y/%m/%d","%d-%m-%Y","%d/%m/%Y","%b %d %Y","%B %d %Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return None

    _load_index()
    qv = _embed(query)
    hits = _search_vec(qv, 50)
    filtered: List[Tuple[int, float, Dict]] = []
    for idx, score, meta in hits:
        md = _parse_date(meta.get("meeting_date"))
        vf = _parse_date(meta.get("valid_from"))
        vt = _parse_date(meta.get("valid_to"))
        keep = False
        if md:
            keep = (start <= md <= end)
        elif vf and vt:
            keep = not (vt < start or vf > end)
        elif vf and not vt:
            keep = (vf <= end)
        if keep:
            filtered.append((idx, score, meta))
    return filtered[:k]
