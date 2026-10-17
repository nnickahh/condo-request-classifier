import csv
import io
import json
from typing import Any

from pydantic import ValidationError

from condo_jev.schema import ResidentRequest
from condo_jev.workflow import JevTriageService

MAX_BATCH_BYTES = 512_000
MAX_BATCH_ROWS = 250
ALLOWED_CHANNELS = {
    "Web",
    "WhatsApp",
    "Email",
    "Phone transcript",
    "Resident App",
    "Intercom / Call Log",
    "Telegram",
}


def parse_batch(raw_bytes: bytes) -> list[dict[str, str]]:
    """Validate and parse a UTF-8 CSV file containing a `text` column into raw row dicts."""
    if len(raw_bytes) > 2_000_000:
        raise ValueError("Use a CSV smaller than 2 MB.")
    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Could not read the CSV. Save it as UTF-8 with a header row.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames or "text" not in reader.fieldnames:
        raise ValueError("CSV must include a 'text' column header.")
    rows = list(reader)
    if not rows:
        raise ValueError("CSV file contains no data rows.")
    if len(rows) > MAX_BATCH_ROWS:
        raise ValueError(f"Batch upload supports at most {MAX_BATCH_ROWS} rows per run.")
    return rows


def spreadsheet_cell(value: Any) -> Any:
    """Escape formula-like text for both CSV export and Streamlit's table display."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def classify_csv_bytes(raw_bytes: bytes, service: JevTriageService) -> list[dict[str, Any]]:
    """Parse a UTF-8 CSV file containing a `text` column and run Jev triage on every row."""
    if len(raw_bytes) > MAX_BATCH_BYTES:
        raise ValueError(f"CSV file exceeds maximum size of {MAX_BATCH_BYTES // 1000} KB.")

    rows = parse_batch(raw_bytes)

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        raw_text = (row.get("text") or "").strip()
        raw_channel = (row.get("channel") or "Resident App").strip()
        channel = raw_channel if raw_channel in ALLOWED_CHANNELS else "Resident App"
        raw_location = (row.get("location") or "").strip() or None

        try:
            req = ResidentRequest(
                text=raw_text,
                channel=channel,  # type: ignore[arg-type]
                location=raw_location,
            )
            decision = service.classify(req)
            results.append(
                {
                    "row": index,
                    "status": "ok",
                    "text": raw_text,
                    "channel": channel,
                    "location": decision.extracted_location or "",
                    "category": decision.category.value,
                    "category_title": decision.category_title,
                    "assigned_team": decision.assigned_team,
                    "priority": decision.priority.value,
                    "choice_confidence": round(decision.choice_confidence, 4),
                    "urgency_score": round(decision.urgency_score, 4),
                    "noul_life_safety_hazard": round(decision.nouls.get("life_safety_hazard", 0.0), 4),
                    "noul_multi_intent_request": round(decision.nouls.get("multi_intent_request", 0.0), 4),
                    "noul_adversarial_or_sensitive": round(decision.nouls.get("adversarial_or_sensitive", 0.0), 4),
                    "noul_requires_human_triage": round(decision.nouls.get("requires_human_triage", 0.0), 4),
                    "needs_human_review": decision.needs_human_review,
                    "review_reasons": "; ".join(decision.review_reasons),
                    "calibration_status": decision.calibration_status,
                    "backend": decision.backend,
                    "model": decision.model,
                    "latency_ms": decision.latency_ms,
                }
            )
        except ValidationError as exc:
            first_msg = exc.errors()[0].get("msg", "Invalid row") if exc.errors() else "Invalid row"
            results.append(
                {
                    "row": index,
                    "status": "validation_error",
                    "text": raw_text,
                    "channel": channel,
                    "location": raw_location or "",
                    "category": "other",
                    "category_title": "Validation Error",
                    "assigned_team": "Triage Supervisor Desk",
                    "priority": "high",
                    "choice_confidence": 0.0,
                    "urgency_score": 0.0,
                    "noul_life_safety_hazard": 0.0,
                    "noul_multi_intent_request": 0.0,
                    "noul_adversarial_or_sensitive": 0.0,
                    "noul_requires_human_triage": 1.0,
                    "needs_human_review": True,
                    "review_reasons": f"Input validation failed: {first_msg}",
                    "calibration_status": "deterministic_safety_rule",
                    "backend": "validation_gate",
                    "model": "pydantic-v2",
                    "latency_ms": 0.0,
                }
            )

    return results


def batch_results_to_csv(records: list[dict[str, Any]]) -> str:
    """Serialize batch triage records into a downloadable CSV string."""
    if not records:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def batch_results_to_json(records: list[dict[str, Any]]) -> str:
    """Serialize batch triage records into formatted JSON."""
    return json.dumps(records, indent=2)
