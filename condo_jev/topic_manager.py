from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from condo_jev.schema import Category

logger = logging.getLogger("condo_jev.topic_manager")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOPICS_FILE = DATA_DIR / "telegram_topics.json"

_LOCK = threading.RLock()

# Mapping of Category enums to standard topic routing keys
CATEGORY_TO_TOPIC_KEY: dict[str, str] = {
    Category.MAINTENANCE.value: "maintenance",
    Category.NOISE.value: "noise",
    Category.CLEANLINESS.value: "cleanliness",
    Category.PARKING.value: "parking",
    Category.SECURITY.value: "parking",
    Category.FACILITIES.value: "facilities",
    Category.BILLING.value: "enquiries",
    Category.RENOVATION.value: "maintenance",
    Category.GENERAL.value: "enquiries",
    Category.OTHER.value: "enquiries",
}

# User-friendly aliases for command inputs (e.g. /settopic gym)
ALIAS_TO_TOPIC_KEY: dict[str, str] = {
    "maint": "maintenance",
    "maintenance": "maintenance",
    "repair": "maintenance",
    "repairs": "maintenance",
    "plumbing": "maintenance",
    "lift": "maintenance",
    "electrical": "maintenance",
    "noise": "noise",
    "disturbance": "noise",
    "sound": "noise",
    "loud": "noise",
    "clean": "cleanliness",
    "cleaning": "cleanliness",
    "cleanliness": "cleanliness",
    "pest": "cleanliness",
    "pests": "cleanliness",
    "trash": "cleanliness",
    "rubbish": "cleanliness",
    "waste": "cleanliness",
    "park": "parking",
    "parking": "parking",
    "carpark": "parking",
    "access": "parking",
    "barrier": "parking",
    "security": "parking",
    "gate": "parking",
    "gym": "facilities",
    "pool": "facilities",
    "amenity": "facilities",
    "amenities": "facilities",
    "facility": "facilities",
    "facilities": "facilities",
    "clubhouse": "facilities",
    "bbq": "facilities",
    "general": "general",
    "misc": "general",
    "enquiry": "enquiries",
    "enquiries": "enquiries",
    "condo enquiry": "enquiries",
    "condo enquiries": "enquiries",
    "condo_enquiries": "enquiries",
    "faq": "enquiries",
    "info": "enquiries",
}

TOPIC_DESCRIPTIONS: dict[str, str] = {
    "maintenance": "Maintenance & Repairs (plumbing, lifts, leaks, electrical)",
    "noise": "Noise & Disturbance (loud music, late noise, shouting)",
    "cleanliness": "Cleaning & Pests (trash, spills, insect sightings)",
    "parking": "Carpark & Access (gates, visitor lots, vehicle obstruction)",
    "facilities": "Gym, Pool & Amenities (equipment breakdown, pool, booking)",
    "enquiries": "Condo Enquiries & FAQ (facilities hours, visitor parking, fees, general rules)",
    "general": "General & Management Office",
}


def normalize_topic_key(input_str: str) -> str | None:
    """Normalize user input or category name to a valid topic key."""
    clean = input_str.strip().lower().replace("-", " ").replace("_", " ")
    if clean in TOPIC_DESCRIPTIONS:
        return clean
    if clean in ALIAS_TO_TOPIC_KEY:
        return ALIAS_TO_TOPIC_KEY[clean]
    for word in clean.split():
        if word in TOPIC_DESCRIPTIONS:
            return word
        if word in ALIAS_TO_TOPIC_KEY:
            return ALIAS_TO_TOPIC_KEY[word]
    return None


class TopicManager:
    """Thread-safe persistent forum topic manager for Telegram staff supergroups."""

    def __init__(self, file_path: Path | None = None):
        self.file_path = file_path or TOPICS_FILE
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.file_path.exists():
            default_config: dict[str, Any] = {
                "maintenance": None,
                "noise": None,
                "cleanliness": None,
                "parking": None,
                "facilities": 10,  # Pre-configured from user's screenshot
                "general": None,
            }
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=2)
            except Exception as exc:
                logger.error(f"Failed to initialize topics file at {self.file_path}: {exc}")

    def load_topics(self) -> dict[str, int | None]:
        with _LOCK:
            self._ensure_file()
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {k: (int(v) if v is not None else None) for k, v in data.items()}
            except Exception as exc:
                logger.warning(f"Error reading topics file {self.file_path}: {exc}")
            return {
                "maintenance": None,
                "noise": None,
                "cleanliness": None,
                "parking": None,
                "facilities": 10,
                "general": None,
            }

    def save_topic(self, topic_key: str, thread_id: int | None) -> bool:
        normalized = normalize_topic_key(topic_key) or topic_key.lower().strip()
        with _LOCK:
            try:
                current = self.load_topics()
                current[normalized] = int(thread_id) if thread_id is not None else None
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(current, f, indent=2)
                return True
            except Exception as exc:
                logger.error(f"Failed to save topic {topic_key}={thread_id}: {exc}")
                return False

    def get_thread_id_for_category(self, category: Category | str | None) -> int | None:
        """Resolve message_thread_id for a given issue category."""
        if not category:
            return None
        cat_val = category.value if isinstance(category, Category) else str(category).lower().strip()
        topic_key = CATEGORY_TO_TOPIC_KEY.get(cat_val) or normalize_topic_key(cat_val) or "general"

        # Check environment variable overrides first (e.g. TELEGRAM_TOPIC_MAINTENANCE)
        env_key = f"TELEGRAM_TOPIC_{topic_key.upper()}"
        env_val = os.getenv(env_key)
        if env_val and env_val.strip().isdigit():
            return int(env_val.strip())

        topics = self.load_topics()
        thread_id = topics.get(topic_key)
        if thread_id is not None:
            return thread_id

        # Fallback for enquiries / general
        if topic_key == "enquiries":
            return topics.get("enquiries") or topics.get("general")
        if topic_key == "general":
            return topics.get("general") or topics.get("enquiries") or topics.get("maintenance")
        return None


# Global singleton
topic_manager = TopicManager()
