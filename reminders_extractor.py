# reminders_extractor.py
from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime
import os

BASE_DIR = Path(os.getenv("DATA_DIR", "/tmp/edbrainai"))
REM_DIR = BASE_DIR / "reminders"
REM_DIR.mkdir(parents=True, exist_ok=True)

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SCHEMA_HELP = """\
Expected reminder format (top lines):
Title: <short title>
Tags: tag1, tag2, ...
MeetingDate: YYYY-MM-DD    # optional
ValidFrom: YYYY-MM-DD      # optional
ValidTo: YYYY-MM-DD        # optional
Body:
<one or a few lines of the actual fact/policy/decision>
"""

def _sanitize_filename(s: str, maxlen: int = 60) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    return s[:maxlen] if s else "untitled"

def _coerce_iso(d: str | None) -> str | None:
    if not d: return None
    d = d.strip()
    if ISO_DATE.match(d):
        return d
    for fmt in ("%Y/%m/%d","%d-%m-%Y","%d/%m/%Y","%b %d %Y","%B %d %Y"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None

def write_reminder(title: str, tags: str = "", body: str = "", meeting_date: str | None = None,
                   valid_from: str | None = None, valid_to: str | None = None) -> str:
    title = title.strip() or "untitled"
    fname = _sanitize_filename(title)
    path = REM_DIR / f"{fname}.txt"
    content = [
        f"Title: {title}",
        f"Tags: {tags}",
        f"MeetingDate: {_coerce_iso(meeting_date) or ''}",
        f"ValidFrom: {_coerce_iso(valid_from) or ''}",
        f"ValidTo: {_coerce_iso(valid_to) or ''}",
        "",
        "Body:",
        body.strip(),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return str(path)

def example():
    return write_reminder(
        title="Quarterly Hiring Freeze",
        tags="policy, finance, headcount",
        body="A temporary hiring freeze is in effect for Q4 except for critical security roles.",
        valid_from="2025-10-01",
        valid_to="2026-01-01"
    )

if __name__ == "__main__":
    p = example()
    print(f"Created example reminder at: {p}")
