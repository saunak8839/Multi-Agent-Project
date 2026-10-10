"""Search the project's simulated technical log records."""

import json
import re
from pathlib import Path

from langchain_core.tools import tool


LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "seed" / "technical_logs.json"


@tool
def search_technical_logs(search_term: str) -> str:
    """Search simulated support logs for a device, service, error code, or issue keyword."""
    terms = {part for part in re.findall(r"[a-z0-9_-]+", search_term.lower()) if len(part) > 2}
    try:
        records = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return json.dumps({"error": "Simulated technical log file is missing."})

    matches = [
        record for record in records
        if not terms or terms.intersection(re.findall(r"[a-z0-9_-]+", json.dumps(record).lower()))
    ]
    return json.dumps({"query": search_term, "matches": matches}, ensure_ascii=False)


TECHNICAL_TOOLS = [search_technical_logs]
