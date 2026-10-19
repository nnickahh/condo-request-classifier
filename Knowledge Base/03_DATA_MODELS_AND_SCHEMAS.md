# 03. Data Models & Schemas

This document defines the core data contracts, Pydantic schemas, and database entity relationships used across the platform.

---

## 1. Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    TICKETS ||--o{ DISPATCHES : triggers
    TICKETS ||--o{ TIMELINE_EVENTS : contains
    TICKETS ||--o| EVIDENCE_ATTACHMENTS : includes
    CONTRACTORS ||--o{ DISPATCHES : receives

    TICKETS {
        string ticket_id PK "e.g. WO-20261014-4821"
        string source "TELEGRAM | WEB | WHATSAPP"
        string telegram_chat_id "Optional chat ID"
        string resident_name
        string unit_number "e.g. #08-14"
        string contact_number
        string category "maintenance, security, etc."
        string priority "emergency, high, normal, low"
        string issue_summary
        string raw_message
        string status "SUBMITTED | DISPATCHED | IN_PROGRESS | RESOLVED"
        string safety_instructions
        datetime created_at
        datetime updated_at
    }

    DISPATCHES {
        string dispatch_id PK "e.g. DISP-9821"
        string ticket_id FK
        string contractor_id FK
        string contractor_name
        string contact_channel "SMS | WHATSAPP | EMAIL"
        string recipient_phone
        string payload_text
        integer eta_minutes
        string delivery_status "SENT | DELIVERED | ACKNOWLEDGED"
        datetime dispatched_at
    }

    TIMELINE_EVENTS {
        string event_id PK
        string ticket_id FK
        string event_type "STATUS_CHANGE | NOTE_ADDED | DISPATCH"
        string actor "AI_AGENT | STAFF | CONTRACTOR | RESIDENT"
        string description
        datetime timestamp
    }

    CONTRACTORS {
        string contractor_id PK
        string company_name
        string primary_category
        string phone
        string email
        integer emergency_sla_minutes
        boolean is_24_7
        boolean is_active
    }

    EVIDENCE_ATTACHMENTS {
        string attachment_id PK
        string ticket_id FK
        string file_path
        string mime_type
        integer file_size_bytes
        datetime uploaded_at
    }
```

---

## 2. Pydantic Models (`condo_classifier/schema.py`)

### Gemini Structured Output Model
```python
from pydantic import BaseModel, Field


class TriageDecision(BaseModel):
    category: str = Field(
        description="One of: maintenance, security, noise, cleanliness, facilities, parking, billing, general_enquiry, other"
    )
    priority: str = Field(description="One of: emergency, high, normal, low")
    issue_summary: str = Field(description="Clear 1-sentence summary of the incident")
    unit_number: str | None = Field(
        default=None, description="Resident unit number if mentioned (e.g. #08-14)"
    )
    is_emergency: bool = Field(
        description="True if pipe burst, major leak, lift entrapment, electrical fire, or intruder"
    )
    assigned_entity: str = Field(description="Name of the contractor or internal team assigned")
    entity_contact: str = Field(description="Phone or contact number of the assigned entity")
    eta_minutes: int = Field(
        description="Estimated arrival time in minutes (15-30 for emergencies, 120+ for normal)"
    )
    safety_instruction: str | None = Field(
        default=None,
        description="Urgent interim advice (e.g. turn off water valve, do not use lift)",
    )
```

---

## 3. SQLite DDL Schema (`data/tickets.db`)

```sql
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    telegram_chat_id INTEGER,
    resident_name TEXT,
    unit_number TEXT,
    contact_number TEXT,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    issue_summary TEXT NOT NULL,
    raw_message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'SUBMITTED',
    assigned_entity TEXT,
    entity_contact TEXT,
    eta_minutes INTEGER,
    safety_instruction TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dispatches (
    dispatch_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    contractor_name TEXT NOT NULL,
    contact_channel TEXT NOT NULL,
    recipient_phone TEXT NOT NULL,
    message_text TEXT NOT NULL,
    delivery_status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);

CREATE TABLE IF NOT EXISTS timeline_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);
```
