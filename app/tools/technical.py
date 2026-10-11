"""Search the project's simulated technical log records."""

import json
import re
from pathlib import Path

from langchain_core.tools import tool


LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "seed" / "technical_logs.json"


def extract_device_ids(message: str) -> set[str]:
    """Accept an explicit identifier, never infer one from issue keywords."""
    identifiers = set(re.findall(r"\b(?:router-[a-z0-9]+|web-client)\b", message.lower()))
    identifiers.update(re.findall(
        r"\bdevice\s+id\s*(?:is\s+|[:=#]\s*)?([a-z0-9]+(?:[-_][a-z0-9]+)*)\b",
        message.lower(),
    ))
    return identifiers


@tool
def search_technical_logs(search_term: str, device_id: str) -> str:
    """Search simulated logs only for an exact customer-provided device ID and issue keywords."""
    device_id = device_id.strip().lower()
    if not device_id:
        return json.dumps({"error": "Ask the customer for their device ID before searching logs.", "matches": []})
    terms = {part for part in re.findall(r"[a-z0-9_-]+", search_term.lower()) if len(part) > 2}
    try:
        records = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return json.dumps({"error": "Simulated technical log file is missing."})

    matches = [
        record for record in records
        if record.get("device_id", "").lower() == device_id
        and (not terms or terms.intersection(re.findall(r"[a-z0-9_-]+", json.dumps(record).lower())))
    ]
    return json.dumps({"query": search_term, "device_id": device_id, "matches": matches}, ensure_ascii=False)
