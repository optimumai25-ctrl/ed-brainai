# knowledge_curator.py
from __future__ import annotations
import csv, json, re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import os

BASE_DIR = Path(os.getenv("DATA_DIR", "/tmp/edbrainai"))
REM_DIR = BASE_DIR / "reminders"
PARSED_DIR = BASE_DIR / "parsed_data"
REG_CSV = BASE_DIR / "embeddings" / "knowledge_registry.csv"
HIST_PATH = BASE_DIR / "chat_history.json"
REG_CSV.parent.mkdir(parents=True, exist_ok=True)

ISO = "%Y-%m-%d"

def _coerce_iso(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip()
    for fmt in (ISO, "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%b %d %Y", "%B %d %Y"):
        try: return datetime.strptime(s, fmt).strftime(ISO)
        except Exception: pass
    return None

def _mtime_iso(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime(ISO)

@dataclass
class Record:
    kind: str
    title: str
    file: str
    folder: str
    tags: str
    meeting_date: Optional[str]
    valid_from: Optional[str]
    valid_to: Optional[str]
    updated: str

def _parse_header(text: str) -> Dict[str, Optional[str]]:
    def grab(label: str) -> Optional[str]:
        import re
        m = re.search(rf"^{label}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None
    return {
        "title": grab("Title"),
        "tags": grab("Tags"),
        "meeting_date": _coerce_iso(grab("MeetingDate")),
        "valid_from": _coerce_iso(grab("ValidFrom")),
        "valid_to": _coerce_iso(grab("ValidTo")),
    }

def _collect() -> List[Record]:
    recs: List[Record] = []

    # Reminders
    if REM_DIR.exists():
        for fp in REM_DIR.glob("*.txt"):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            h = _parse_header(text)
            recs.append(Record(
                kind="reminder",
                title=h["title"] or fp.stem,
                file=fp.name,
                folder="Reminders",
                tags=h["tags"] or "",
                meeting_date=h["meeting_date"],
                valid_from=h["valid_from"],
                valid_to=h["valid_to"],
                updated=_mtime_iso(fp),
            ))

    # Parsed folders
    if PARSED_DIR.exists():
        for folder in sorted(PARSED_DIR.iterdir()):
            if not folder.is_dir(): continue
            for fp in folder.rglob("*.txt"):
                text = fp.read_text(encoding="utf-8", errors="ignore")
                h = _parse_header(text)
                recs.append(Record(
                    kind="parsed",
                    title=h["title"] or fp.stem,
                    file=fp.name,
                    folder=folder.name,
                    tags=h["tags"] or "",
                    meeting_date=h["meeting_date"],
                    valid_from=h["valid_from"],
                    valid_to=h["valid_to"],
                    updated=_mtime_iso(fp),
                ))
    return recs

def main():
    recs = _collect()
    with open(REG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Kind","Title","File","Folder","Tags","MeetingDate","ValidFrom","ValidTo","Updated"])
        for r in recs:
            w.writerow([r.kind, r.title, r.file, r.folder, r.tags, r.meeting_date or "", r.valid_from or "", r.valid_to or "", r.updated])
    print(f"[curator] Wrote {len(recs)} rows → {REG_CSV}")
    return {"rows": len(recs), "csv": str(REG_CSV)}

if __name__ == "__main__":
    main()
