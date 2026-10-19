from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from condo_jev.backends import LocalJevSimulatorBackend
from condo_jev.condo_enquiries_rag import (
    CONDO_NAME,
    MCST_NO,
    answer_condo_enquiry,
    format_staff_enquiry_log,
    get_article_by_id,
)
from condo_jev.estate_directory import (
    extract_estate_components,
    extract_noise_details,
    format_estate_levels_clarification,
    format_noise_clarification_prompt,
    is_ambiguous_carpark_location,
    is_concrete_estate_location,
    is_noise_complaint,
    validate_estate_location,
)
from condo_jev.media_screening import screen_media_file
from condo_jev.schema import Category, Priority, ResidentRequest
from condo_jev.signals import extract_location
from condo_jev.work_order import (
    ESTATE_TECHNICIANS,
    WorkOrder,
    WorkOrderStatus,
    WorkOrderStore,
    build_quick_location_markup,
    build_resident_card_markup,
    build_staff_alert_markup,
    format_resident_assigned_notification,
    format_resident_closed_notification,
    format_resident_receipt,
    format_resident_resolved_notification,
    format_resident_status_notification,
    format_staff_dispatch_alert,
    format_telegram_resident_card,
    send_telegram_direct_message,
)
from condo_jev.topic_manager import (
    ALIAS_TO_TOPIC_KEY,
    TOPIC_DESCRIPTIONS,
    normalize_topic_key,
    topic_manager,
)
from condo_jev.translator import (
    SUPPORTED_LANGUAGES,
    detect_language,
    transcribe_audio_to_english,
    translate_text,
)
from condo_jev.workflow import JevTriageService

logger = logging.getLogger("condo_jev.telegram_bot")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _is_probable_location(text: str) -> bool:
    """Check if a short resident reply is providing a missing unit or block location."""
    clean = text.strip()
    if len(clean) > 80:
        return False
    # Patterns like #05-12, 08-14, #12-34, Blk B, Tower 2, Level 3, Gym, Basement, B1, Pillar
    loc_patterns = [
        r"#?[B0-9]{1,2}[-\s]\d{1,4}",
        r"(?i)\b(block|blk|tower|lobby|level|lvl|floor|unit|apt|b\d|lot|pillar|basement|gym|pool|carpark|chute|pit|bbq|pavilion|court)\b",
        r"^[1-4]$",
        r"(?i)^b[12]$",
    ]
    return any(re.search(p, clean) for p in loc_patterns)


_CONVERSATIONAL_WORDS = {
    "ok", "okay", "k", "noted", "got it", "thanks", "thank you", "thx",
    "alright", "sure", "all good", "cool", "great", "fine", "received",
    "understood", "yes", "no", "nope", "yep", "hi", "hello", "hey",
    "morning", "good morning", "afternoon", "good afternoon", "evening",
    "good evening", "test", "bye", "goodbye", "good", "nice", "ty",
    "thank you very much", "thank you so much", "thanks a lot", "many thanks",
    "noted with thanks", "ok noted", "okay thanks", "alright thanks"
}

_CONVERSATIONAL_TOKENS = {
    "ok", "okay", "k", "noted", "got", "it", "thanks", "thank", "you", "thx",
    "alright", "sure", "all", "good", "cool", "great", "fine", "received",
    "understood", "yes", "no", "nope", "yep", "hi", "hello", "hey",
    "morning", "afternoon", "evening", "test", "bye", "goodbye", "nice", "ty",
    "very", "much", "so", "lot", "a", "with", "sir", "team", "to", "again",
    "for", "the", "help", "update", "man", "mate", "bro"
}


def _is_conversational_reply(text: str) -> bool:
    """Detect if incoming text is a conversational acknowledgment or greeting rather than an issue."""
    clean = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not clean:
        return True
    if clean in _CONVERSATIONAL_WORDS:
        return True
    words = clean.split()
    if len(words) <= 6 and all(w in _CONVERSATIONAL_TOKENS for w in words):
        return True
    if len(clean) <= 3 and not re.search(r"\d", clean):
        return True
    return False


def _clean_bot_mention(text: str, bot_username: str | None) -> str:
    """Remove bot username mention and command prefixes from resident message."""
    if not text:
        return ""
    cleaned = text
    if bot_username:
        cleaned = re.sub(rf"@{re.escape(bot_username)}\b", "", cleaned, flags=re.IGNORECASE)
    # Also strip any generic leading bot mention or slash command
    cleaned = re.sub(r"^@\w+\s*", "", cleaned.strip())
    return cleaned.strip()


class TelegramSessionStore:
    """Thread-safe persistent session storage for Telegram chats."""

    def __init__(self, file_path: Path | None = None):
        if file_path is None:
            self.file_path = Path(__file__).resolve().parent.parent / "data" / "telegram_sessions.json"
        else:
            self.file_path = Path(file_path)
        self._lock = threading.Lock()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_data({})

    def _read_data(self) -> dict[str, dict[str, Any]]:
        if not self.file_path.exists():
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _write_data(self, data: dict[str, dict[str, Any]]) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Failed to persist Telegram session: {exc}")

    def get_session(self, resident_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._read_data()
            return data.get(str(resident_id), {})

    def set_pending_location(self, resident_id: str, wo_id: str | None) -> None:
        with self._lock:
            data = self._read_data()
            sess = data.setdefault(str(resident_id), {})
            sess["pending_location_wo_id"] = wo_id
            if wo_id:
                sess["last_wo_id"] = wo_id
            sess["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_data(data)

    def pop_pending_location(self, resident_id: str) -> str | None:
        with self._lock:
            data = self._read_data()
            sess = data.get(str(resident_id), {})
            val = sess.pop("pending_location_wo_id", None)
            if val is not None:
                self._write_data(data)
            return val

    def set_pending_handover(self, resident_id: str, wo_id: str | None) -> None:
        with self._lock:
            data = self._read_data()
            sess = data.setdefault(str(resident_id), {})
            sess["pending_handover_wo_id"] = wo_id
            if wo_id:
                sess["last_wo_id"] = wo_id
            sess["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_data(data)

    def pop_pending_handover(self, resident_id: str) -> str | None:
        with self._lock:
            data = self._read_data()
            sess = data.get(str(resident_id), {})
            val = sess.pop("pending_handover_wo_id", None)
            if val is not None:
                self._write_data(data)
            return val

    def set_pending_photo(self, resident_id: str, photo_path: str | None) -> None:
        with self._lock:
            data = self._read_data()
            sess = data.setdefault(str(resident_id), {})
            sess["pending_photo_path"] = photo_path
            sess["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_data(data)

    def pop_pending_photo(self, resident_id: str) -> str | None:
        with self._lock:
            data = self._read_data()
            sess = data.get(str(resident_id), {})
            val = sess.pop("pending_photo_path", None)
            if val is not None:
                self._write_data(data)
            return val

    def set_pending_noise(self, resident_id: str, data: dict[str, Any] | None) -> None:
        with self._lock:
            s_data = self._read_data()
            sess = s_data.setdefault(str(resident_id), {})
            sess["pending_noise"] = data
            sess["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_data(s_data)

    def get_pending_noise(self, resident_id: str) -> dict[str, Any] | None:
        with self._lock:
            s_data = self._read_data()
            sess = s_data.get(str(resident_id), {})
            return sess.get("pending_noise")

    def pop_pending_noise(self, resident_id: str) -> dict[str, Any] | None:
        with self._lock:
            s_data = self._read_data()
            sess = s_data.get(str(resident_id), {})
            val = sess.pop("pending_noise", None)
            if val is not None:
                self._write_data(s_data)
            return val

    def set_last_work_order(self, resident_id: str, wo_id: str) -> None:
        with self._lock:
            data = self._read_data()
            sess = data.setdefault(str(resident_id), {})
            sess["last_wo_id"] = wo_id
            sess["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_data(data)

    def get_last_work_order(self, resident_id: str) -> str | None:
        with self._lock:
            data = self._read_data()
            sess = data.get(str(resident_id), {})
            return sess.get("last_wo_id")

    def set_language(self, resident_id: str, lang_code: str) -> None:
        with self._lock:
            data = self._read_data()
            sess = data.setdefault(str(resident_id), {})
            sess["language"] = lang_code.lower().strip()
            sess["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_data(data)

    def get_language(self, resident_id: str) -> str:
        with self._lock:
            data = self._read_data()
            sess = data.get(str(resident_id), {})
            return sess.get("language", "en")


class CondoManagementTelegramBot:
    """
    Telegram Bot interface for the Condominium Management Agent.
    Collects resident issue reports, executes Jev System-One triage,
    and automatically creates and dispatches structured Work Orders.
    """

    def __init__(
        self,
        triage_service: JevTriageService | None = None,
        work_order_store: WorkOrderStore | None = None,
        staff_chat_id: str | int | None = None,
        token: str | None = None,
        session_store: TelegramSessionStore | None = None,
    ):
        self.triage_service = triage_service or JevTriageService(backend=LocalJevSimulatorBackend())
        self.store = work_order_store or WorkOrderStore()
        self.staff_chat_id = staff_chat_id or os.getenv("TELEGRAM_STAFF_CHAT_ID")
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.session_store = session_store or TelegramSessionStore()

        self.bot_username: str | None = None
        self.bot_id: int | None = None

        # In-memory fast cache
        self._pending_location_requests: dict[str, str] = {}
        self._pending_handover_questions: dict[str, str] = {}
        self._pending_noise_requests: dict[str, dict[str, Any]] = {}

    def process_resident_report(
        self,
        text: str,
        resident_id: str,
        resident_username: str | None = None,
        resident_name: str | None = None,
        location_hint: str | None = None,
        photo_path: str | None = None,
    ) -> tuple[WorkOrder, str]:
        """
        Core triage and work-order generation pipeline:
        1. Parse input with ResidentRequest(channel='Telegram').
        2. Run LangGraph Jev System-One classifier.
        3. Persist Work Order in WorkOrderStore.
        4. Return WorkOrder and formatted Telegram message.
        """
        extracted_loc = location_hint or extract_location(text)
        request = ResidentRequest(
            text=text,
            channel="Telegram",
            location=extracted_loc,
        )

        classification = self.triage_service.classify(request)
        work_order = self.store.create_work_order(
            request=request,
            classification=classification,
            resident_id=resident_id,
            resident_username=resident_username,
            resident_name=resident_name,
            photo_path=photo_path,
            resident_chat_id=resident_id,
            notify_staff=False,
        )

        self.session_store.set_last_work_order(resident_id, work_order.work_order_id)

        has_loc = is_concrete_estate_location(work_order.location, text)
        if not has_loc or work_order.missing_details:
            self._pending_location_requests[resident_id] = work_order.work_order_id
            self.session_store.set_pending_location(resident_id, work_order.work_order_id)
        else:
            self._pending_location_requests.pop(resident_id, None)
            self.session_store.pop_pending_location(resident_id)

        card_text = format_telegram_resident_card(work_order)
        return work_order, card_text

    def handle_status_query(self, query_id: str, resident_id: str | None = None) -> str:
        """Lookup work order status by ID without emojis."""
        wo = self.store.get(query_id)
        if not wo:
            return (
                f"Work Order `{query_id}` was not found in our estate registry.\n\n"
                "Please verify the reference number (e.g. `WO-EST-1042` or `1042`) or use `/myrequests` "
                "to see your recent logged tickets."
            )

        prio_label = "EMERGENCY" if wo.priority == Priority.EMERGENCY else str(wo.priority.value).upper()
        loc_display = wo.location or "Pending unit confirmation"
        tech_line = f"\n• *Assigned Officer/Tech:* {wo.assigned_technician}" if wo.assigned_technician else ""
        if wo.assigned_technician and wo.technician_phone:
            tech_line += f" (`{wo.technician_phone}`)"
        if wo.assigned_technician and wo.technician_vendor:
            tech_line += f" · {wo.technician_vendor}"
        res_line = f"\n• *Resolution:* {wo.resolution_summary}" if wo.resolution_summary else ""
        rating_line = f"\n• *Resident Rating:* {wo.resident_feedback_rating} / 5 Stars" if wo.resident_feedback_rating else ""
        photo_line = f"\n• *Photo Evidence:* Attached (`{Path(wo.photo_path).name}`)" if wo.photo_path else ""
        notes_str = "\n".join(f"  • {n}" for n in wo.notes[-3:]) if wo.notes else "No additional notes."

        header = "*URGENT EMERGENCY WORK ORDER STATUS*" if wo.priority == Priority.EMERGENCY else "*ESTATE MANAGEMENT WORK ORDER STATUS*"
        return (
            f"{header}\n\n"
            f"• *Work Order:* `{wo.work_order_id}`\n"
            f"• *Department:* {wo.category_title}\n"
            f"• *Assigned Team:* {wo.assigned_team}"
            f"{tech_line}{photo_line}\n"
            f"• *Priority:* {prio_label}\n"
            f"• *Target SLA:* {wo.sla_target}\n"
            f"• *Location:* {loc_display}\n"
            f"• *Current Status:* *{wo.status}*{res_line}{rating_line}\n"
            f"• *Logged At:* {wo.created_at[:19].replace('T', ' ')} UTC\n\n"
            f"*Original Report:*\n_{wo.text}_\n\n"
            f"*Recent Updates:*\n{notes_str}"
        )

    def handle_my_requests(self, resident_id: str) -> str:
        """List all work orders logged by the current resident without emojis."""
        orders = self.store.list_for_resident(resident_id)
        if not orders:
            return (
                "*ESTATE MANAGEMENT & RESIDENT CONCIERGE*\n\n"
                "You do not have any active or past work orders on file.\n\n"
                "To report an issue, simply send a message describing the problem "
                "(e.g. _'Corridor light flickering outside #05-12'_)."
            )

        lines = [
            f"*YOUR LOGGED WORK ORDERS ({len(orders)})*",
            "",
        ]
        for wo in orders[:6]:
            prio_tag = "[EMERGENCY]" if wo.priority == Priority.EMERGENCY else f"[{str(wo.priority.value).upper()}]"
            loc = f" · {wo.location}" if wo.location else ""
            tech_note = f" | Tech: `{wo.assigned_technician}`" if wo.assigned_technician else ""
            lines.append(
                f"• *{wo.work_order_id}* {prio_tag} — {wo.category_title}{loc}\n"
                f"  Status: `{wo.status}` | Team: `{wo.assigned_team}`{tech_note}\n"
            )
        return "\n".join(lines)

    def get_welcome_message(self, resident_name: str | None = None) -> str:
        """Executive, professional 24/7 resident concierge greeting."""
        salutation = f", {resident_name}" if resident_name else ""
        return (
            f"*{CONDO_NAME}*\n"
            f"*Resident Concierge Desk*\n\n"
            f"Good day{salutation}.\n\n"
            f"Welcome to your 24/7 Estate Concierge. How may our management team assist you today?\n\n"
            f"You may send a message or attach a photo directly to:\n"
            f"• *Report an estate defect or repair* (e.g. _\"Corridor light flickering outside #05-12\"_)\n"
            f"• *Enquire about estate facilities* (e.g. pool hours, BBQ booking, visitor parking)\n\n"
            f"To check on your existing requests at any time, type `/status`."
        )

    def get_resident_welcome_markup(self) -> Any:
        """Clear any persistent custom keyboard buttons to keep native clean chat input."""
        try:
            from telegram import ReplyKeyboardRemove
            return ReplyKeyboardRemove()
        except Exception:
            return None

    def get_resident_reply_keyboard(self) -> Any:
        """Clear any persistent custom keyboard buttons to keep native clean chat input."""
        try:
            from telegram import ReplyKeyboardRemove
            return ReplyKeyboardRemove()
        except Exception:
            return None

    def get_emergency_message(self) -> str:
        """Emergency contact info without emojis."""
        return (
            "*24/7 ESTATE EMERGENCY HOTLINES*\n\n"
            "If there is immediate danger to life or personal safety, please contact emergency services immediately:\n\n"
            "• *Police Emergency:* `999`\n"
            "• *SCDF (Fire & Ambulance):* `995`\n"
            "• *Estate 24/7 Guardhouse & Security:* `+65 6789 0123`\n"
            "• *Lift Trapped Passenger 24/7 Desk:* `+65 6789 0124`\n"
            "• *Estate Management Office:* `+65 6789 0125` (Mon–Fri 9:00 AM – 6:00 PM)\n\n"
            "Any urgent incident reported here is escalated with immediate operational priority."
        )

    def get_staff_dashboard(self) -> str:
        """Overview for estate staff & technicians without emojis."""
        stats = self.store.get_cmms_stats()
        return (
            "*ESTATE OPERATIONS DESK*\n\n"
            f"• Total Work Orders: *{stats['total_work_orders']}*\n"
            f"• Active / Open: *{stats['open_work_orders']}*\n"
            f"• Pending Review: *{stats['pending_review']}*\n"
            f"• Assigned to Tech: *{stats['assigned']}*\n"
            f"• Emergency Alerts: *{stats['emergency']}*\n"
            f"• Resolved / Closed: *{stats['resolved']}*\n\n"
            f"*Staff Commands:*\n"
            f"• `/pending` — View tickets awaiting dispatch or review\n"
            f"• `/techs` — Directory of available technicians & contacts\n"
            f"• `/assign <id> <tech>` — Assign technician (e.g. `/assign 1058 Ah Seng`)\n"
            f"• `/progress <id>` — Advance ticket to In Progress\n"
            f"• `/resolve <id> <summary>` — Mark resolved with repair summary\n"
            f"• `/note <id> <text>` — Append field progress note\n"
            f"• `/status <id>` — View full work order dossier\n"
            f"• `/topics` — View forum topic routing status\n"
            f"• `/settopic <category>` — Bind topic thread to an issue category"
        )

    async def _broadcast_emergency_to_general(
        self,
        context: Any,
        full_text: str,
        category: Category | str | None,
        priority_str: str,
    ) -> None:
        """Broadcast high-priority emergency alert banner to the general staff chat."""
        try:
            from telegram.constants import ParseMode
            cat_display = category.value if isinstance(category, Category) else str(category or "General")
            summary_lines = [line.strip() for line in full_text.splitlines() if line.strip()][:4]
            summary_snippet = "\n".join(summary_lines)

            banner = (
                "🚨 *[CRITICAL EMERGENCY BROADCAST]* 🚨\n"
                "──────────────────────────────\n"
                f"• *Priority:* {priority_str.upper() or 'HIGH'}\n"
                f"• *Department:* {cat_display.title()}\n\n"
                f"{summary_snippet}\n\n"
                f"_⚡ Full work order details and live technician actions are routed to the department topic thread._"
            )
            await context.bot.send_message(
                chat_id=self.staff_chat_id,
                text=banner,
                parse_mode=ParseMode.MARKDOWN,
            )
            print("[STAFF DISPATCH] Critical emergency broadcasted to general chat.", flush=True)
        except Exception as exc:
            logger.warning(f"Could not broadcast emergency banner to general chat: {exc}")

    async def _dispatch_to_staff(
        self,
        context: Any,
        text: str,
        reply_markup: Any = None,
        category: Category | str | None = None,
        priority: Priority | str | None = None,
        is_emergency: bool = False,
        attachment_path: str | None = None,
    ) -> None:
        """
        Route staff dispatch alert directly to relevant topic thread.
        CRITICAL RULE: The general chat is ONLY used for important/emergency messages.
        Routine tickets are strictly confined to their department topic threads.
        """
        if not self.staff_chat_id:
            return

        priority_str = str(priority.value if isinstance(priority, Priority) else priority or "").lower().strip()
        is_critical = bool(is_emergency) or priority_str == "emergency"

        thread_id = topic_manager.get_thread_id_for_category(category)
        if thread_id is not None:
            try:
                await context.bot.send_message(
                    chat_id=self.staff_chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    message_thread_id=thread_id,
                )
                print(f"[STAFF DISPATCH SUCCESS] Dispatched to chat {self.staff_chat_id}, topic thread {thread_id}", flush=True)

                if attachment_path and Path(attachment_path).exists():
                    try:
                        ext = Path(attachment_path).suffix.lower()
                        with open(attachment_path, "rb") as media_f:
                            if ext in (".ogg", ".oga"):
                                await context.bot.send_voice(chat_id=self.staff_chat_id, voice=media_f, message_thread_id=thread_id)
                            elif ext in (".mp3", ".wav", ".m4a", ".aac"):
                                await context.bot.send_audio(chat_id=self.staff_chat_id, audio=media_f, message_thread_id=thread_id)
                            elif ext in (".mp4", ".mov", ".avi"):
                                await context.bot.send_video(chat_id=self.staff_chat_id, video=media_f, message_thread_id=thread_id)
                            elif ext in (".jpg", ".jpeg", ".png"):
                                await context.bot.send_photo(chat_id=self.staff_chat_id, photo=media_f, message_thread_id=thread_id)
                    except Exception as media_err:
                        logger.warning(f"Could not forward media attachment to staff thread: {media_err}")

                # If this is a critical emergency, ALSO broadcast an alert to General!
                if is_critical:
                    await self._broadcast_emergency_to_general(context, text, category, priority_str)
                return
            except Exception as exc:
                logger.warning(f"Could not forward to topic thread {thread_id} ({exc})")

        # Fallback when topic thread is not mapped:
        # ONLY send to General chat IF this is an important / emergency issue!
        if is_critical:
            try:
                await context.bot.send_message(
                    chat_id=self.staff_chat_id,
                    text=text,
                    reply_markup=reply_markup,
                )
                print(f"[STAFF DISPATCH SUCCESS] Dispatched critical alert to general chat {self.staff_chat_id}", flush=True)
                if attachment_path and Path(attachment_path).exists():
                    try:
                        ext = Path(attachment_path).suffix.lower()
                        with open(attachment_path, "rb") as media_f:
                            if ext in (".ogg", ".oga"):
                                await context.bot.send_voice(chat_id=self.staff_chat_id, voice=media_f)
                            elif ext in (".mp3", ".wav", ".m4a", ".aac"):
                                await context.bot.send_audio(chat_id=self.staff_chat_id, audio=media_f)
                            elif ext in (".mp4", ".mov", ".avi"):
                                await context.bot.send_video(chat_id=self.staff_chat_id, video=media_f)
                            elif ext in (".jpg", ".jpeg", ".png"):
                                await context.bot.send_photo(chat_id=self.staff_chat_id, photo=media_f)
                    except Exception as media_err:
                        logger.warning(f"Could not forward media attachment to general staff chat: {media_err}")
            except Exception as exc:
                logger.warning(f"Failed to deliver critical alert to staff chat {self.staff_chat_id}: {exc}")
        else:
            print(f"[STAFF DISPATCH SUPPRESSED] Category '{category}' has no topic thread and priority is routine ({priority_str}). Suppressed from General chat.", flush=True)

    async def _dispatch_enquiry_log_to_staff(
        self,
        context: Any,
        text: str,
    ) -> None:
        """Route informational resident enquiry log to condo enquiries topic thread (no general fallback)."""
        if not self.staff_chat_id:
            return
        thread_id = topic_manager.get_thread_id_for_category("enquiries")
        if thread_id is not None:
            try:
                await context.bot.send_message(
                    chat_id=self.staff_chat_id,
                    text=text,
                    message_thread_id=thread_id,
                )
                print(f"[STAFF DISPATCH SUCCESS] Enquiry log sent to thread {thread_id}", flush=True)
            except Exception as exc:
                logger.warning(f"Failed to deliver enquiry log to topic thread {thread_id}: {exc}")
        else:
            # When no dedicated enquiries topic thread is configured, suppress sending to General chat
            # so the staff General chat remains completely clean.
            print("[STAFF DISPATCH INFO] No dedicated enquiries topic configured; suppressed from General chat.", flush=True)

    def build_application(self) -> Any:
        """Construct the python-telegram-bot Application instance."""
        try:
            from telegram import (
                BotCommand,
                InlineKeyboardButton,
                InlineKeyboardMarkup,
                KeyboardButton,
                ReplyKeyboardMarkup,
                Update,
            )
            from telegram.constants import ParseMode
            from telegram.ext import (
                ApplicationBuilder,
                CallbackQueryHandler,
                CommandHandler,
                ContextTypes,
                MessageHandler,
                filters,
            )
        except ImportError as exc:
            raise RuntimeError(
                "python-telegram-bot is required to run the live bot. "
                "Install it via `pip install python-telegram-bot`."
            ) from exc

        if not self.token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set. Please provide it via constructor or in .env."
            )

        async def post_init(application: Any) -> None:
            """Register bot command menu with Telegram so users have quick autocomplete."""
            try:
                me = await application.bot.get_me()
                self.bot_username = me.username
                self.bot_id = me.id
            except Exception as exc:
                logger.warning(f"Could not fetch bot identity: {exc}")

            commands = [
                BotCommand("new", "Log a new maintenance request or issue"),
                BotCommand("report", "Report a maintenance or security issue"),
                BotCommand("faq", "Estate bylaws, facilities & rules Q&A"),
                BotCommand("status", "Check the live progress of your work order"),
                BotCommand("myrequests", "View all your logged work orders"),
                BotCommand("emergency", "24/7 Security & Emergency Hotlines"),
                BotCommand("receipt", "Official incident receipt & complaint copy"),
                BotCommand("staff", "Estate staff & technician operations desk"),
                BotCommand("help", "Concierge guide & instructions"),
            ]
            try:
                await application.bot.set_my_commands(commands)
            except Exception as exc:
                logger.warning(f"Could not register Telegram bot commands: {exc}")

        app = (
            ApplicationBuilder()
            .token(self.token)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .pool_timeout(30.0)
            .post_init(post_init)
            .build()
        )

        async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            user = update.effective_user
            user_name = user.first_name if user else None
            welcome_text = self.get_welcome_message(user_name)

            await update.message.reply_text(
                welcome_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=self.get_resident_welcome_markup(),
            )

        async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text(
                self.get_welcome_message(),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=self.get_resident_welcome_markup(),
            )

        async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Inform resident of the estate's official language standard."""
            text = (
                "*ESTATE COMMUNICATIONS STANDARD*\n\n"
                "• *Official Language:* **English (MCST 4521 Standard)**\n\n"
                "All official estate notices, work order records, status tracking, and formal incident receipts are issued in English.\n\n"
                "You are welcome to submit reports or descriptions in any language; our concierge will translate your input into English for estate triage and staff dispatch."
            )
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

        async def faq_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Interactive menu for estate FAQs and RAG knowledge base."""
            buttons = [
                [
                    InlineKeyboardButton("Pool & Gym Hours", callback_data="faq:art:FAC-POOL"),
                    InlineKeyboardButton("BBQ Pit Booking", callback_data="faq:art:FAC-BBQ"),
                ],
                [
                    InlineKeyboardButton("Visitor Parking", callback_data="faq:art:PARK-VISITOR"),
                    InlineKeyboardButton("EV & Season Parking", callback_data="faq:art:PARK-SEASON"),
                ],
                [
                    InlineKeyboardButton("Renovation Hours", callback_data="faq:art:RENO-HOURS"),
                    InlineKeyboardButton("Pet By-Laws", callback_data="faq:art:PET-RULES"),
                ],
                [
                    InlineKeyboardButton("Maintenance Fees", callback_data="faq:art:FEE-PAYMENT"),
                    InlineKeyboardButton("Access Cards & Fobs", callback_data="faq:art:CARD-ACCESS"),
                ],
                [
                    InlineKeyboardButton("Parcel Lockers", callback_data="faq:art:PARCEL-LOCKER"),
                    InlineKeyboardButton("Office & Guardhouse", callback_data="faq:art:OFFICE-CONTACT"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(buttons)
            await update.message.reply_text(
                "*ESTATE KNOWLEDGE BASE & RESIDENT FAQ*\n"
                "──────────────────────────────\n"
                "Select a topic below, or *type any question directly* in chat:\n\n"
                "• _\"What time does the pool close?\"_\n"
                "• _\"How much is the BBQ deposit?\"_\n"
                "• _\"Are visitors allowed to park overnight?\"_\n"
                "• _\"Can contractors drill on Saturday?\"_",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )

        async def emergency_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text(
                self.get_emergency_message(),
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=self.get_resident_welcome_markup(),
            )

        async def staff_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text(
                self.get_staff_dashboard(),
                parse_mode=ParseMode.MARKDOWN,
            )

        async def techs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            lines = [
                "*ESTATE TECHNICIANS & CONTRACTOR DIRECTORY*",
                "──────────────────────────────",
            ]
            for t in ESTATE_TECHNICIANS:
                status_dot = "[Available]" if t.get("available", True) else "[On Duty]"
                lines.append(
                    f"{status_dot} *{t['name']}* ({t['department']})\n"
                    f"   Role: {t['role']} · {t['vendor']}\n"
                    f"   Phone: `{t['phone']}`\n"
                )
            lines.append("_To assign: `/assign <work_order_id> <tech_name>`_")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

        async def pending_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            all_orders = self.store.list_all(limit=50)
            pending = [
                o for o in all_orders
                if o.status in (
                    WorkOrderStatus.PENDING_REVIEW.value,
                    WorkOrderStatus.EMERGENCY_ESCALATION.value,
                    WorkOrderStatus.AUTO_DISPATCHED.value,
                )
            ]
            if not pending:
                await update.message.reply_text("All clear. There are no pending work orders requiring staff review.")
                return

            lines = [
                f"*PENDING WORK ORDERS QUEUE ({len(pending)})*",
                "",
            ]
            keyboard = []
            for o in pending[:6]:
                prio_tag = "[EMERGENCY]" if o.priority == Priority.EMERGENCY else f"[{str(o.priority.value).upper()}]"
                lines.append(
                    f"• *{o.work_order_id}* {prio_tag} — {o.category_title}\n"
                    f"  Loc: {o.location or 'Unspecified'} | Status: `{o.status}`\n"
                )
                keyboard.append([
                    InlineKeyboardButton(f"Assign {o.work_order_id}", callback_data=f"status:{o.work_order_id}"),
                    InlineKeyboardButton("In Progress", callback_data=f"staff_prog:{o.work_order_id}"),
                ])
            markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            await update.message.reply_text("\n".join(lines), reply_markup=markup, parse_mode=ParseMode.MARKDOWN)

        async def assign_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text(
                    "Usage: `/assign <work_order_id> <technician_name>`\n"
                    "Example: `/assign 1058 Ah Seng` or `/assign WO-EST-1058 John Tan`\n"
                    "Type `/techs` to see the available roster.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return
            wo_id = context.args[0]
            tech_input = " ".join(context.args[1:])

            matched_tech = next(
                (t for t in ESTATE_TECHNICIANS if tech_input.lower() in t["name"].lower()),
                None,
            )
            tech_name = matched_tech["name"] if matched_tech else tech_input
            phone = matched_tech["phone"] if matched_tech else None
            vendor = matched_tech["vendor"] if matched_tech else "Contractor"

            updated = self.store.assign_technician(
                wo_id,
                technician_name=tech_name,
                phone=phone,
                vendor=vendor,
                note=f"Assigned via staff Telegram command by user {update.effective_user.id if update.effective_user else 'staff'}",
            )
            if updated:
                await update.message.reply_text(
                    f"**Assigned {tech_name}** to Work Order `#{updated.work_order_id}`.\n"
                    f"• Vendor: {vendor}\n"
                    f"• Contact: `{phone or 'None'}`\n"
                    f"• Resident has been automatically notified on Telegram.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text(f"Work Order `{wo_id}` was not found.")

        async def progress_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not context.args:
                await update.message.reply_text("Usage: `/progress <work_order_id>`")
                return
            wo_id = context.args[0]
            updated = self.store.update_status(
                wo_id,
                WorkOrderStatus.IN_PROGRESS.value,
                note="Technician arrived on site and commenced work.",
            )
            if updated:
                await update.message.reply_text(f"Work Order `#{updated.work_order_id}` is now **In Progress**.")
            else:
                await update.message.reply_text(f"Work Order `{wo_id}` was not found.")

        async def resolve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text(
                    "Usage: `/resolve <work_order_id> <resolution_summary>`\n"
                    "Example: `/resolve 1058 Replaced faulty tap washer. Tested normal.`",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return
            wo_id = context.args[0]
            summary = " ".join(context.args[1:])
            updated = self.store.resolve_work_order(
                wo_id,
                resolution_summary=summary,
                note="Resolved via technician Telegram command",
            )
            if updated:
                await update.message.reply_text(
                    f"Work Order `#{updated.work_order_id}` has been marked **Resolved**.\n"
                    f"Resident has received notification with confirmation and feedback options.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text(f"Work Order `{wo_id}` was not found.")

        async def note_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text("Usage: `/note <work_order_id> <message>`")
                return
            wo_id = context.args[0]
            note_body = " ".join(context.args[1:])
            wo = self.store.get(wo_id)
            if not wo:
                await update.message.reply_text(f"Work Order `{wo_id}` was not found.")
                return
            self.store.update_status(wo.work_order_id, wo.status, note=note_body, notify_resident=False)
            await update.message.reply_text(f"Added note to Work Order `#{wo.work_order_id}`.")

        async def delete_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message:
                return
            if not context.args:
                await update.message.reply_text(
                    "*Delete Work Order (Admin)*\n\n"
                    "Usage: `/delete <work_order_id>`\n"
                    "Example: `/delete WO-EST-1062` or `/delete 1062`",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return
            wo_id = context.args[0].strip()
            success = self.store.delete_work_order(wo_id)
            if success:
                await update.message.reply_text(
                    f"*Work Order Deleted*\n\n"
                    f"Work Order `{wo_id}` has been permanently deleted from the estate database.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text(
                    f"*Record Not Found*\n\n"
                    f"No active work order found matching `{wo_id}`.",
                    parse_mode=ParseMode.MARKDOWN,
                )

        async def chatid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message:
                return
            chat = update.effective_chat
            user = update.effective_user
            if not chat:
                return
            chat_type = chat.type.title()
            title = chat.title or (f"@{user.username}" if user and user.username else (user.first_name if user else "Direct Chat"))
            thread_id = update.message.message_thread_id
            is_topic = getattr(update.message, "is_topic_message", False)
            topic_line = f"• *Topic / Thread ID:* `{thread_id}`\n" if thread_id else ""
            bind_hint = (
                f"\nTo bind this topic to a category, run:\n"
                f"`/settopic <category>` inside this topic!\n"
            ) if thread_id else ""
            print(f"[CHATID COMMAND] Chat ID: {chat.id}, Thread ID: {thread_id}, is_topic: {is_topic}, Title: '{title}'", flush=True)
            await update.message.reply_text(
                f"*TELEGRAM CHAT & TOPIC IDENTIFICATION*\n"
                f"──────────────────────────────\n"
                f"• *Chat Title:* {title}\n"
                f"• *Chat Type:* {chat_type}\n"
                f"• *Chat ID:* `{chat.id}`\n"
                f"{topic_line}"
                f"{bind_hint}\n"
                f"To configure this chat, set in your `.env` file:\n"
                f"```bash\nTELEGRAM_STAFF_CHAT_ID={chat.id}\n```",
                parse_mode=ParseMode.MARKDOWN,
            )

        async def topics_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message:
                return
            current_topics = topic_manager.load_topics()
            lines = [
                "*ESTATE STAFF FORUM TOPIC ROUTING*",
                "──────────────────────────────",
            ]
            for key, desc in TOPIC_DESCRIPTIONS.items():
                tid = current_topics.get(key)
                status = f"Thread `{tid}`" if tid is not None else "_Not bound_"
                lines.append(f"• *{key.title()}*: {status}\n  _{desc}_")
            lines.append(
                "\nTo bind a topic, type `/settopic <category>` inside that topic (e.g. `/settopic maintenance`)."
            )
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

        async def settopic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message:
                return
            chat = update.effective_chat
            thread_id = update.message.message_thread_id
            is_topic = getattr(update.message, "is_topic_message", False)
            print(f"[SETTOPIC COMMAND] Chat ID: {chat.id if chat else None}, Thread ID: {thread_id}, is_topic: {is_topic}", flush=True)

            if not context.args:
                current_topics = topic_manager.load_topics()
                lines = [
                    "*ESTATE STAFF TOPIC CONFIGURATION*",
                    "──────────────────────────────",
                    "To bind a category to this topic, run:",
                    "`/settopic <category>` inside the topic!\n",
                    "*Available Categories:*",
                ]
                for key, desc in TOPIC_DESCRIPTIONS.items():
                    tid = current_topics.get(key)
                    status = f"Thread `{tid}`" if tid is not None else "_Not bound_"
                    lines.append(f"• *{key}*: {desc} (Status: {status})")
                if thread_id:
                    lines.append(f"\n_Current Topic Thread ID:_ `{thread_id}`")
                await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
                return

            cat_input = context.args[0].lower().strip()
            topic_key = normalize_topic_key(cat_input)
            if not topic_key:
                await update.message.reply_text(
                    f"Unknown category `{cat_input}`. Choose from:\n"
                    "• `maintenance` (repairs, plumbing, lifts, leaks)\n"
                    "• `noise` (noise disturbances, shouting, music)\n"
                    "• `cleanliness` (trash, spills, pest control)\n"
                    "• `parking` (carpark, gates, access cards)\n"
                    "• `facilities` (gym, pool, bbq, clubhouse)\n"
                    "• `enquiries` (condo enquiries, FAQ, visitor parking rates, fees)\n"
                    "• `general` (billing, general queries)",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            target_thread = thread_id
            if len(context.args) >= 2 and context.args[1].isdigit():
                target_thread = int(context.args[1])

            if target_thread is None:
                await update.message.reply_text(
                    "You ran this command without a topic thread context.\n"
                    "Run this command directly inside the intended topic, or specify thread ID:\n"
                    f"`/settopic {topic_key} <thread_id>`\n"
                    f"Example: `/settopic {topic_key} 10`",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            success = topic_manager.save_topic(topic_key, target_thread)
            if success:
                desc = TOPIC_DESCRIPTIONS.get(topic_key, topic_key.title())
                await update.message.reply_text(
                    f"*Topic Bound Successfully!*\n"
                    f"• Category: *{desc}*\n"
                    f"• Topic Thread ID: `{target_thread}`\n\n"
                    f"All incoming tickets for this category will now automatically post into this topic.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text(
                    f"Could not save topic configuration for `{topic_key}`. Please check logs.",
                    parse_mode=ParseMode.MARKDOWN,
                )

        async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            user = update.effective_user
            res_id = str(user.id) if user else "unknown"

            if not context.args:
                # Intelligent status lookup: check resident's latest active work order!
                orders = self.store.list_for_resident(res_id)
                if not orders:
                    await update.message.reply_text(
                        "You do not have any logged work orders yet. To report an issue, simply describe it here!\n"
                        "Or specify a reference number: `/status WO-EST-1042`",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return
                # Show latest work order
                latest_wo = orders[0]
                resp = self.handle_status_query(latest_wo.work_order_id, res_id)
                markup_dict = build_resident_card_markup(latest_wo)
                inline_rows = [
                    [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                    for row in markup_dict.get("inline_keyboard", [])
                ]
                markup = InlineKeyboardMarkup(inline_rows) if inline_rows else None
                try:
                    await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)
                except Exception:
                    await update.message.reply_text(resp)
                return

            query_id = context.args[0]
            resp = self.handle_status_query(query_id, res_id)
            wo = self.store.get(query_id)
            markup = None
            if wo:
                markup_dict = build_resident_card_markup(wo)
                inline_rows = [
                    [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                    for row in markup_dict.get("inline_keyboard", [])
                ]
                markup = InlineKeyboardMarkup(inline_rows) if inline_rows else None
            try:
                await update.message.reply_text(resp, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)
            except Exception:
                await update.message.reply_text(resp)

        async def myrequests_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            res_id = str(update.effective_user.id if update.effective_user else "unknown")
            orders = self.store.list_for_resident(res_id)
            text = self.handle_my_requests(res_id)
            if not orders:
                await update.message.reply_text(
                    text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=self.get_resident_welcome_markup(),
                )
                return

            keyboard = []
            for wo in orders[:5]:
                keyboard.append([
                    InlineKeyboardButton(f"Inspect {wo.work_order_id}", callback_data=f"status:{wo.work_order_id}")
                ])
            markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            await update.message.reply_text(text, reply_markup=markup, parse_mode=ParseMode.MARKDOWN)

        async def report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not context.args:
                await update.message.reply_text(
                    "*LOG A NEW ISSUE / REQUEST*\n"
                    "──────────────────────────────\n"
                    "Please state your issue and unit number (e.g. `/new Water leaking at #08-14`), or describe it directly.\n\n"
                    "• *Tip:* You can also send a photo directly with your description as a caption!",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=self.get_resident_welcome_markup(),
                )
                return
            full_text = " ".join(context.args)
            await _handle_text_report(update, context, full_text)

        async def receipt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Allow resident to retrieve an official incident receipt for their work order."""
            user = update.effective_user
            res_id = str(user.id) if user else "unknown"
            pref_lang = self.session_store.get_language(res_id)

            if context.args:
                wo_id = context.args[0]
            else:
                orders = self.store.list_for_resident(res_id)
                if not orders:
                    await update.message.reply_text(
                        "You do not have any logged work orders yet. To report an issue, simply describe it here!\n"
                        "Or specify a ticket number: `/receipt WO-EST-1042`",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return
                wo_id = orders[0].work_order_id

            wo = self.store.get(wo_id)
            if not wo:
                await update.message.reply_text(f"Work Order `{wo_id}` was not found.")
                return

            receipt_text = format_resident_receipt(wo)
            await update.message.reply_text(receipt_text, parse_mode=ParseMode.MARKDOWN)

        async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            query = update.callback_query
            await query.answer()
            data = query.data or ""
            user = update.effective_user
            res_id = str(user.id) if user else "unknown"

            if data.startswith("faq:art:"):
                art_id = data.split(":", 2)[2]
                art_res = get_article_by_id(art_id)
                if art_res:
                    back_markup = InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("« Back to FAQ Topics", callback_data="nav:faq"),
                            InlineKeyboardButton("Main Menu", callback_data="nav:welcome"),
                        ]
                    ])
                    await query.edit_message_text(
                        art_res["reply"],
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=back_markup,
                    )
                return

            if data in ("faq:menu", "nav:faq"):
                buttons = [
                    [
                        InlineKeyboardButton("Pool & Gym Hours", callback_data="faq:art:FAC-POOL"),
                        InlineKeyboardButton("BBQ Pit Booking", callback_data="faq:art:FAC-BBQ"),
                    ],
                    [
                        InlineKeyboardButton("Visitor Parking", callback_data="faq:art:PARK-VISITOR"),
                        InlineKeyboardButton("EV & Season Parking", callback_data="faq:art:PARK-SEASON"),
                    ],
                    [
                        InlineKeyboardButton("Renovation Hours", callback_data="faq:art:RENO-HOURS"),
                        InlineKeyboardButton("Pet By-Laws", callback_data="faq:art:PET-RULES"),
                    ],
                    [
                        InlineKeyboardButton("Maintenance Fees", callback_data="faq:art:FEE-PAYMENT"),
                        InlineKeyboardButton("Access Cards & Fobs", callback_data="faq:art:CARD-ACCESS"),
                    ],
                    [
                        InlineKeyboardButton("Parcel Lockers", callback_data="faq:art:PARCEL-LOCKER"),
                        InlineKeyboardButton("Office & Guardhouse", callback_data="faq:art:OFFICE-CONTACT"),
                    ],
                    [
                        InlineKeyboardButton("« Back to Main Menu", callback_data="nav:welcome"),
                    ],
                ]
                await query.edit_message_text(
                    "*ESTATE KNOWLEDGE BASE & RESIDENT FAQ*\n"
                    "──────────────────────────────\n"
                    "Select a topic below, or *type any question directly* in chat:\n\n"
                    "• _\"What time does the pool close?\"_\n"
                    "• _\"How much is the BBQ deposit?\"_\n"
                    "• _\"Are visitors allowed to park overnight?\"_\n"
                    "• _\"Can contractors drill on Saturday?\"_",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(buttons),
                )
                return

            if data == "nav:myrequests":
                orders = self.store.list_for_resident(res_id)
                text = self.handle_my_requests(res_id)
                keyboard = []
                for wo in orders[:5]:
                    keyboard.append([
                        InlineKeyboardButton(f"Inspect {wo.work_order_id}", callback_data=f"status:{wo.work_order_id}")
                    ])
                keyboard.append([
                    InlineKeyboardButton("« Back to Main Menu", callback_data="nav:welcome")
                ])
                await query.edit_message_text(
                    text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            if data == "nav:emergency":
                keyboard = [
                    [InlineKeyboardButton("« Back to Main Menu", callback_data="nav:welcome")]
                ]
                await query.edit_message_text(
                    self.get_emergency_message(),
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            if data == "nav:welcome":
                user_name = user.first_name if user else None
                await query.edit_message_text(
                    self.get_welcome_message(user_name),
                    reply_markup=self.get_resident_welcome_markup(),
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            if data.startswith("person:"):
                wo_id = data.split(":", 1)[1]
                self.store.update_status(wo_id, "Pending Staff Review", note="Resident requested officer callback")
                self.session_store.pop_pending_handover(res_id)
                self._pending_handover_questions.pop(res_id, None)
                await query.edit_message_text(
                    f"*ESTATE MANAGEMENT & CONCIERGE*\n"
                    f"──────────────────────────────\n"
                    f"Certainly. Work Order `#{wo_id}` has been escalated to our on-duty estate manager for direct follow-up.\n\n"
                    f"*Direct Contacts:*\n"
                    f"• Estate Management Office: `+65 6789 0125` (Mon–Fri 9:00 AM – 6:00 PM)\n"
                    f"• 24/7 Guardhouse & Intercom: `+65 6789 0123`",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data.startswith("team:"):
                wo_id = data.split(":", 1)[1]
                self.session_store.pop_pending_handover(res_id)
                self._pending_handover_questions.pop(res_id, None)
                await query.edit_message_text(
                    f"*ESTATE MANAGEMENT & CONCIERGE*\n"
                    f"──────────────────────────────\n"
                    f"Thank you. Our duty team has taken charge of Work Order `#{wo_id}`. "
                    f"You can check real-time progress anytime via `/status {wo_id}`.\n\n"
                    f"Have a pleasant day ahead.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data.startswith("status:"):
                wo_id = data.split(":", 1)[1]
                resp = self.handle_status_query(wo_id, res_id)
                wo = self.store.get(wo_id)
                markup = None
                if wo:
                    markup_dict = build_resident_card_markup(wo)
                    inline_rows = [
                        [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                        for row in markup_dict.get("inline_keyboard", [])
                    ]
                    markup = InlineKeyboardMarkup(inline_rows) if inline_rows else None
                try:
                    await query.message.reply_text(resp, reply_markup=markup, parse_mode=ParseMode.MARKDOWN)
                except Exception:
                    await query.message.reply_text(resp)
            elif data.startswith("loc:"):
                wo_id = data.split(":", 1)[1]
                self.session_store.set_pending_location(res_id, wo_id)
                self._pending_location_requests[res_id] = wo_id
                await query.message.reply_text(
                    f"Please reply with your unit or block number for Work Order `#{wo_id}` (e.g. `#08-14` or `Block B Level 4`).\n\n"
                    f"{format_estate_levels_clarification()}",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data.startswith("close:"):
                wo_id = data.split(":", 1)[1]
                self.store.close_work_order(wo_id, note="Resident confirmed resolution via Telegram", notify_resident=False)
                rating_keyboard = [
                    [
                        InlineKeyboardButton("1 Star", callback_data=f"rate:{wo_id}:1"),
                        InlineKeyboardButton("2 Stars", callback_data=f"rate:{wo_id}:2"),
                        InlineKeyboardButton("3 Stars", callback_data=f"rate:{wo_id}:3"),
                        InlineKeyboardButton("4 Stars", callback_data=f"rate:{wo_id}:4"),
                        InlineKeyboardButton("5 Stars", callback_data=f"rate:{wo_id}:5"),
                    ],
                ]
                await query.edit_message_text(
                    f"*Work Order `#{wo_id}` is officially closed.*\n\n"
                    f"Thank you for confirming resolution. How would you rate the speed and quality of our service?\n\n"
                    f"_Whenever you need to make a new report or inquiry, simply type `/new` anytime._",
                    reply_markup=InlineKeyboardMarkup(rating_keyboard),
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data.startswith("rate:"):
                parts = data.split(":")
                wo_id = parts[1]
                stars = int(parts[2])
                updated_wo = self.store.record_feedback(wo_id, rating=stars)
                await query.edit_message_text(
                    f"*Rating Received:* {stars} / 5 Stars\n\n"
                    f"Thank you for your feedback on Work Order `#{wo_id}`. "
                    f"Your rating helps our estate management team maintain high community standards.\n\n"
                    f"──────────────────────────────\n"
                    f"Whenever you want to submit a new report or inquiry, you can type `/new` or send a message anytime.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                if self.staff_chat_id and updated_wo:
                    res_name = updated_wo.resident_name or (f"@{updated_wo.resident_username}" if updated_wo.resident_username else "Resident")
                    staff_alert = (
                        f"*[RESIDENT SERVICE RATING RECEIVED]*\n"
                        f"──────────────────────────────\n"
                        f"• *Ticket:* `#{updated_wo.work_order_id}`\n"
                        f"• *Rating:* {stars} / 5 Stars\n"
                        f"• *Resident:* {res_name}\n"
                        f"• *Department:* {updated_wo.category_title}\n"
                        f"• *Location:* `{updated_wo.location or 'Estate Common Area'}`"
                    )
                    await self._dispatch_to_staff(
                        context=context,
                        text=staff_alert,
                        category=updated_wo.category,
                        priority=Priority.HIGH if stars <= 2 else Priority.LOW,
                    )
            elif data.startswith("clean_bbq:"):
                parts = data.split(":")
                wo_id = parts[1]
                pit_num = parts[2]
                self.session_store.pop_pending_location(res_id)
                self._pending_location_requests.pop(res_id, None)
                updated_wo = self.store.update_location(wo_id, f"BBQ Pit {pit_num}")
                if updated_wo:
                    # Dispatch to staff topic thread 13 (cleanliness)
                    if self.staff_chat_id:
                        staff_alert = format_staff_dispatch_alert(updated_wo)
                        staff_markup_dict = build_staff_alert_markup(updated_wo)
                        staff_markup = InlineKeyboardMarkup([
                            [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                            for row in staff_markup_dict.get("inline_keyboard", [])
                        ])
                        await self._dispatch_to_staff(
                            context=context,
                            text=staff_alert,
                            reply_markup=staff_markup,
                            category=Category.CLEANLINESS,
                            priority=Priority.HIGH,
                        )

                    reply_msg_text = (
                        f"*Housekeeping Dispatched: BBQ Pit {pit_num}*\n\n"
                        f"• *Ticket:* `#{updated_wo.work_order_id}`\n"
                        f"• Cleaners are on the way to scrub and sanitize BBQ Pit {pit_num} (SLA: Within 30 minutes).\n"
                        f"• The previous user's $50 cleaning deposit will be forfeited following booking and CCTV review.\n\n"
                        f"_Guardhouse 24/7 hotline: +65 6789 0123_"
                    )
                    await query.edit_message_text(
                        reply_msg_text,
                        parse_mode=ParseMode.MARKDOWN,
                    )
            elif data.startswith("quick_loc:"):
                parts = data.split(":", 2)
                wo_id = parts[1]
                loc_text = parts[2]
                self.session_store.pop_pending_location(res_id)
                self._pending_location_requests.pop(res_id, None)
                updated_wo = self.store.update_location(wo_id, loc_text)
                if updated_wo:
                    # Dispatch to staff now that location is confirmed
                    if self.staff_chat_id:
                        staff_alert = format_staff_dispatch_alert(updated_wo)
                        staff_markup_dict = build_staff_alert_markup(updated_wo)
                        staff_markup = InlineKeyboardMarkup([
                            [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                            for row in staff_markup_dict.get("inline_keyboard", [])
                        ])
                        await self._dispatch_to_staff(
                            context=context,
                            text=staff_alert,
                            reply_markup=staff_markup,
                            category=updated_wo.category,
                            priority=updated_wo.priority,
                        )

                    reply_msg_text = (
                        f"*Location Confirmed:* `{loc_text}`\n\n"
                        f"Thank you. Your Work Order `#{updated_wo.work_order_id}` "
                        f"has been dispatched to our **{updated_wo.assigned_team}** for prompt attention.\n\n"
                        f"• *Priority:* {updated_wo.priority.value.upper()} (Estimated SLA: {updated_wo.sla_target})\n"
                        f"• *Status:* In Progress\n\n"
                        f"• Track live status: `/status {updated_wo.work_order_id}`\n"
                        f"• Make another report: `/new`"
                    )
                    await query.edit_message_text(
                        reply_msg_text,
                        parse_mode=ParseMode.MARKDOWN,
                    )
            elif data.startswith("reopen:"):
                wo_id = data.split(":", 1)[1]
                self.store.reopen_work_order(wo_id, reason="Resident reported issue persists via Telegram callback")
                await query.edit_message_text(
                    f"*WORK ORDER REOPENED: `#{wo_id}`*\n\n"
                    f"Your ticket has been reopened and escalated to our duty estate manager for urgent inspection.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data.startswith("staff_assign:"):
                parts = data.split(":")
                if len(parts) >= 3:
                    wo_id, tech_name = parts[1], parts[2]
                    matched_tech = next((t for t in ESTATE_TECHNICIANS if tech_name.lower() in t["name"].lower()), None)
                    phone = matched_tech["phone"] if matched_tech else None
                    vendor = matched_tech["vendor"] if matched_tech else "Contractor"
                    self.store.assign_technician(
                        wo_id,
                        technician_name=tech_name,
                        phone=phone,
                        vendor=vendor,
                        note="Assigned via staff Telegram one-touch button",
                    )
                    await query.edit_message_text(
                        f"**Assigned to {tech_name}** (`{wo_id}`)\n"
                        f"• Vendor: {vendor}\n"
                        f"• Contact: `{phone or 'None'}`\n"
                        f"Resident has been automatically notified.",
                        parse_mode=ParseMode.MARKDOWN,
                    )
            elif data.startswith("staff_prog:"):
                wo_id = data.split(":", 1)[1]
                self.store.update_status(
                    wo_id,
                    WorkOrderStatus.IN_PROGRESS.value,
                    note="Staff marked in progress via Telegram",
                )
                await query.edit_message_text(
                    f"Work Order `#{wo_id}` marked as **In Progress**.\nResident has been notified.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data.startswith("staff_res:"):
                wo_id = data.split(":", 1)[1]
                staff_user = query.from_user
                staff_name = staff_user.first_name or staff_user.username or "Duty Staff"
                self.store.resolve_work_order(
                    wo_id,
                    resolution_summary=f"Rectification works completed and verified on-site by {staff_name}.",
                )
                await query.edit_message_text(
                    f"*WORK ORDER RESOLVED: `#{wo_id}`*\n\n"
                    f"• *Verified by:* {staff_name}\n"
                    f"• *Status:* Resolved\n\n"
                    f"Resident has been automatically notified with satisfaction survey.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data.startswith("staff_xfer_menu:"):
                wo_id = data.split(":", 1)[1]
                xfer_keyboard = [
                    [
                        InlineKeyboardButton("Maintenance (Repairs/Leaks)", callback_data=f"staff_xfer_do:{wo_id}:maintenance"),
                        InlineKeyboardButton("Noise & Nuisance", callback_data=f"staff_xfer_do:{wo_id}:noise"),
                    ],
                    [
                        InlineKeyboardButton("Cleanliness & Pests", callback_data=f"staff_xfer_do:{wo_id}:cleanliness"),
                        InlineKeyboardButton("Facilities & Amenities", callback_data=f"staff_xfer_do:{wo_id}:facilities"),
                    ],
                    [
                        InlineKeyboardButton("Carpark & Access", callback_data=f"staff_xfer_do:{wo_id}:parking"),
                        InlineKeyboardButton("Condo Enquiries", callback_data=f"staff_xfer_do:{wo_id}:enquiries"),
                    ],
                    [
                        InlineKeyboardButton("Cancel / Keep Current", callback_data=f"staff_xfer_cancel:{wo_id}"),
                    ],
                ]
                await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(xfer_keyboard))
            elif data.startswith("staff_xfer_cancel:"):
                wo_id = data.split(":", 1)[1]
                target_wo = self.store.get(wo_id)
                if target_wo:
                    staff_markup_dict = build_staff_alert_markup(target_wo)
                    staff_markup = InlineKeyboardMarkup([
                        [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                        for row in staff_markup_dict.get("inline_keyboard", [])
                    ])
                    await query.edit_message_reply_markup(reply_markup=staff_markup)
            elif data.startswith("staff_xfer_do:"):
                parts = data.split(":")
                wo_id = parts[1]
                target_cat_str = parts[2]
                staff_user = query.from_user
                staff_name = staff_user.first_name or staff_user.username or "Staff"
                updated_wo = self.store.update_category(
                    wo_id,
                    target_cat_str,
                    note=f"Transferred to {target_cat_str.title()} by {staff_name}",
                )
                if updated_wo:
                    await query.edit_message_text(
                        f"*[WORK ORDER TRANSFERRED]*\n\n"
                        f"Work Order `#{wo_id}` has been transferred to **{updated_wo.category_title}** ({updated_wo.assigned_team}) by {staff_name}.\n\n"
                        f"Dispatched into the {target_cat_str.title()} topic thread.",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    # Dispatch fresh alert into destination topic
                    if self.staff_chat_id:
                        staff_alert = format_staff_dispatch_alert(updated_wo)
                        staff_markup_dict = build_staff_alert_markup(updated_wo)
                        staff_markup = InlineKeyboardMarkup([
                            [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                            for row in staff_markup_dict.get("inline_keyboard", [])
                        ])
                        await self._dispatch_to_staff(
                            context=context,
                            text=staff_alert,
                            reply_markup=staff_markup,
                            category=updated_wo.category,
                            priority=updated_wo.priority,
                        )
            elif data == "cmd:emergency":
                await query.message.reply_text(self.get_emergency_message(), parse_mode=ParseMode.MARKDOWN)
            elif data == "cmd:myrequests":
                await myrequests_cmd(update, context)
            elif data.startswith("setlang:"):
                chosen_lang = data.split(":", 1)[1]
                user = query.from_user
                res_id = str(user.id) if user else "unknown"
                self.session_store.set_language(res_id, chosen_lang)
                lang_name = SUPPORTED_LANGUAGES.get(chosen_lang, "English")
                confirm_msg = f"*Language Preference Updated: {lang_name}*\n\nYour concierge responses will now be provided in {lang_name}. You can change this anytime with `/lang`."
                if chosen_lang != "en":
                    trans = translate_text(confirm_msg, target_lang=chosen_lang)
                    if trans.is_translated:
                        confirm_msg = trans.translated_text
                await query.edit_message_text(confirm_msg, parse_mode=ParseMode.MARKDOWN)
            elif data.startswith("receipt:"):
                wo_id = data.split(":", 1)[1]
                wo = self.store.get(wo_id)
                if wo:
                    receipt_text = format_resident_receipt(wo)
                    await query.message.reply_text(receipt_text, parse_mode=ParseMode.MARKDOWN)
                else:
                    await query.message.reply_text(f"Could not locate Work Order `#{wo_id}`.", parse_mode=ParseMode.MARKDOWN)

        async def location_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle GPS location pins shared by residents via Telegram."""
            loc = update.message.location
            if not loc:
                return
            user = update.effective_user
            res_id = str(user.id) if user else "unknown"
            coords_str = f"GPS Pin (Lat: {loc.latitude:.5f}, Lon: {loc.longitude:.5f})"

            pending_wo_id = self.session_store.pop_pending_location(res_id) or self._pending_location_requests.pop(res_id, None)
            if pending_wo_id:
                updated_wo = self.store.update_location(pending_wo_id, coords_str)
                await update.message.reply_text(
                    f"*Location Pin Recorded*\n\n"
                    f"Coordinates `{coords_str}` attached to Work Order `#{pending_wo_id}`.\n"
                    f"Our technician will use this for on-site navigation.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text(
                    f"Location received: `{coords_str}`. Please describe the issue you would like to report.",
                    parse_mode=ParseMode.MARKDOWN,
                )

        async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle photos and photo evidence attachments sent by residents."""
            if not update.message or not update.message.photo:
                return

            chat = update.effective_chat
            is_group = bool(chat and chat.type in ("group", "supergroup"))
            user = update.effective_user
            res_id = str(user.id) if user else "unknown"
            caption = (update.message.caption or "").strip()

            bot_username = getattr(self, "bot_username", None) or (context.bot.username if context.bot else None)

            # Check if bot was mentioned or swiped to reply
            has_mention = False
            if bot_username and f"@{bot_username.lower()}" in caption.lower():
                has_mention = True
            elif update.message.caption_entities:
                for ent in update.message.caption_entities:
                    if ent.type == "mention":
                        m_txt = caption[ent.offset : ent.offset + ent.length].lower()
                        if bot_username and m_txt == f"@{bot_username.lower()}":
                            has_mention = True
                    elif ent.type == "text_mention" and ent.user and ent.user.id == context.bot.id:
                        has_mention = True

            reply_msg = update.message.reply_to_message
            is_reply_to_bot = False
            if reply_msg and reply_msg.from_user:
                if reply_msg.from_user.id == context.bot.id:
                    is_reply_to_bot = True
                elif bot_username and reply_msg.from_user.username and reply_msg.from_user.username.lower() == bot_username.lower():
                    is_reply_to_bot = True

            # In group chats, only respond if mentioned or swiped to reply to bot
            if is_group and not (has_mention or is_reply_to_bot):
                return

            clean_caption = _clean_bot_mention(caption, bot_username)

            attachments_dir = Path(__file__).resolve().parent.parent / "data" / "attachments"
            attachments_dir.mkdir(parents=True, exist_ok=True)

            photo = update.message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{res_id}_{timestamp_str}.jpg"
            saved_path = attachments_dir / filename
            await photo_file.download_to_drive(str(saved_path))

            # Screen photo for human faces and valid maintenance evidence
            screening = screen_media_file(saved_path)
            if not screening.is_acceptable:
                try:
                    os.remove(saved_path)
                except Exception:
                    pass
                await update.message.reply_text(
                    screening.rejection_reason or "Image was rejected.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # If user swiped to reply to an existing Work Order message with this photo
            if reply_msg:
                replied_text = reply_msg.text or reply_msg.caption or ""
                wo_match = re.search(r"#?(WO-EST-\d{4,6}|EST-\d{4,6})", replied_text)
                if wo_match:
                    raw_wo_id = wo_match.group(1)
                    target_wo_id = raw_wo_id if raw_wo_id.startswith("WO-") else f"WO-{raw_wo_id}"
                    updated = self.store.attach_photo(target_wo_id, str(saved_path), filename=filename)
                    if updated:
                        note_suffix = f"\n• Note: _{clean_caption}_" if clean_caption else ""
                        if clean_caption:
                            self.store.update_status(target_wo_id, updated.status, note=f"Resident photo note: {clean_caption}", notify_resident=False)
                        await update.message.reply_text(
                            f"*Photo Evidence Attached*\n\n"
                            f"Your photo has been attached to Work Order `#{updated.work_order_id}`.{note_suffix}",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        return

            if clean_caption and len(clean_caption.strip()) >= 3:
                # Caption included: triage report immediately with photo attached
                await _handle_text_report(update, context, clean_caption.strip(), photo_path=str(saved_path))
            else:
                # Save photo as pending for the resident's next text description
                self.session_store.set_pending_photo(res_id, str(saved_path))
                await update.message.reply_text(
                    "*Photo Evidence Received*\n\n"
                    "Thank you for sharing the photo. Please reply with a brief description of the issue and your unit or location "
                    "(e.g. _'Corridor ceiling outside #08-14 is leaking'_).\n\n"
                    f"{format_estate_levels_clarification()}\n\n"
                    "I will automatically attach your photo to the official work order.",
                    parse_mode=ParseMode.MARKDOWN,
                )

        async def video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle video evidence sent by residents with screening for faces and relevance."""
            if not update.message or not update.message.video:
                return

            video = update.message.video
            caption = update.message.caption or ""
            user = update.effective_user
            chat = update.effective_chat
            res_id = str(user.id) if user else "unknown"
            username = user.username if user else None
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() if user else None

            is_group = chat.type in ("group", "supergroup")
            has_mention = False
            bot_username = self.bot_username
            if caption and bot_username and f"@{bot_username.lower()}" in caption.lower():
                has_mention = True

            reply_msg = update.message.reply_to_message
            is_reply_to_bot = False
            if reply_msg and reply_msg.from_user:
                if reply_msg.from_user.id == context.bot.id:
                    is_reply_to_bot = True
                elif bot_username and reply_msg.from_user.username and reply_msg.from_user.username.lower() == bot_username.lower():
                    is_reply_to_bot = True

            if is_group and not (has_mention or is_reply_to_bot):
                return

            clean_caption = _clean_bot_mention(caption, bot_username)

            attachments_dir = Path(__file__).resolve().parent.parent / "data" / "attachments"
            attachments_dir.mkdir(parents=True, exist_ok=True)

            video_file = await context.bot.get_file(video.file_id)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{res_id}_{timestamp_str}.mp4"
            saved_path = attachments_dir / filename
            await video_file.download_to_drive(str(saved_path))

            # Screen video for human faces and valid maintenance content
            screening = screen_media_file(saved_path)
            if not screening.is_acceptable:
                try:
                    os.remove(saved_path)
                except Exception:
                    pass
                await update.message.reply_text(
                    screening.rejection_reason or "Video was rejected.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # If user swiped to reply to an existing Work Order message with this video
            if reply_msg:
                replied_text = reply_msg.text or reply_msg.caption or ""
                wo_match = re.search(r"#?(WO-EST-\d{4,6}|EST-\d{4,6})", replied_text)
                if wo_match:
                    raw_wo_id = wo_match.group(1)
                    target_wo_id = raw_wo_id if raw_wo_id.startswith("WO-") else f"WO-{raw_wo_id}"
                    updated = self.store.attach_photo(target_wo_id, str(saved_path), filename=filename)
                    if updated:
                        note_suffix = f"\n• Note: _{clean_caption}_" if clean_caption else ""
                        if clean_caption:
                            self.store.update_status(target_wo_id, updated.status, note=f"Resident video note: {clean_caption}", notify_resident=False)
                        await update.message.reply_text(
                            f"*Video Evidence Attached*\n\n"
                            f"Your video has been attached to Work Order `#{updated.work_order_id}`.{note_suffix}",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        return

            if clean_caption and len(clean_caption.strip()) >= 3:
                await _handle_text_report(update, context, clean_caption.strip(), photo_path=str(saved_path))
            else:
                self.session_store.set_pending_photo(res_id, str(saved_path))
                await update.message.reply_text(
                    "*Video Evidence Received*\n\n"
                    "Thank you for sharing the video. Please reply with a brief description of the issue and your unit or location "
                    "(e.g. _'Corridor ceiling outside #08-14 is leaking'_).\n\n"
                    f"{format_estate_levels_clarification()}\n\n"
                    "I will automatically attach your video to the official work order.",
                    parse_mode=ParseMode.MARKDOWN,
                )

        async def audio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle voice notes and audio evidence sent by residents."""
            if not update.message:
                return

            voice = update.message.voice
            audio = update.message.audio
            if not voice and not audio:
                return

            caption = update.message.caption or ""
            user = update.effective_user
            chat = update.effective_chat
            res_id = str(user.id) if user else "unknown"
            username = user.username if user else None
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() if user else None

            is_group = chat.type in ("group", "supergroup")
            has_mention = False
            bot_username = self.bot_username
            if caption and bot_username and f"@{bot_username.lower()}" in caption.lower():
                has_mention = True

            reply_msg = update.message.reply_to_message
            is_reply_to_bot = False
            if reply_msg and reply_msg.from_user:
                if reply_msg.from_user.id == context.bot.id:
                    is_reply_to_bot = True
                elif bot_username and reply_msg.from_user.username and reply_msg.from_user.username.lower() == bot_username.lower():
                    is_reply_to_bot = True

            if is_group and not (has_mention or is_reply_to_bot):
                return

            clean_caption = _clean_bot_mention(caption, bot_username)

            attachments_dir = Path(__file__).resolve().parent.parent / "data" / "attachments"
            attachments_dir.mkdir(parents=True, exist_ok=True)

            media_obj = voice or audio
            file_id = media_obj.file_id
            audio_file = await context.bot.get_file(file_id)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = ".ogg" if voice else (Path(audio.file_name).suffix.lower() if audio and audio.file_name else ".mp3")
            if not ext or ext not in (".ogg", ".oga", ".mp3", ".wav", ".m4a", ".aac"):
                ext = ".ogg" if voice else ".mp3"

            filename = f"audio_{res_id}_{timestamp_str}{ext}"
            saved_path = attachments_dir / filename
            await audio_file.download_to_drive(str(saved_path))

            # If user swiped to reply to an existing Work Order message with this audio
            if reply_msg:
                replied_text = reply_msg.text or reply_msg.caption or ""
                wo_match = re.search(r"#?(WO-EST-\d{4,6}|EST-\d{4,6})", replied_text)
                if wo_match:
                    raw_wo_id = wo_match.group(1)
                    target_wo_id = raw_wo_id if raw_wo_id.startswith("WO-") else f"WO-{raw_wo_id}"
                    updated = self.store.attach_photo(target_wo_id, str(saved_path), filename=filename)
                    if updated:
                        note_suffix = f"\n• Note: _{clean_caption}_" if clean_caption else ""
                        if clean_caption:
                            self.store.update_status(target_wo_id, updated.status, note=f"Resident audio note: {clean_caption}", notify_resident=False)
                        await update.message.reply_text(
                            f"*Audio Evidence Attached*\n\n"
                            f"Your audio recording has been attached to Work Order `#{updated.work_order_id}`.{note_suffix}",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        return

            if clean_caption and len(clean_caption.strip()) >= 3:
                await _handle_text_report(update, context, clean_caption.strip(), photo_path=str(saved_path))
            else:
                # Transcribe audio to speech text
                media_type = "voice message" if voice else "audio recording"
                audio_transcript = transcribe_audio_to_english(saved_path)

                if audio_transcript and len(audio_transcript.strip()) >= 3:
                    await _handle_text_report(
                        update,
                        context,
                        audio_transcript.strip(),
                        photo_path=str(saved_path),
                        is_transcribed=True,
                    )
                else:
                    self.session_store.set_pending_photo(res_id, str(saved_path))
                    await update.message.reply_text(
                        f"*Audio Evidence Received*\n\n"
                        f"Thank you for sharing the {media_type}. Your recording has been securely saved as evidence.\n\n"
                        f"Please reply with a brief text transcription or description of your issue, including your unit or location "
                        f"(e.g. _'Water pipe bursting outside #04-12'_).\n\n"
                        f"{format_estate_levels_clarification()}\n\n"
                        f"Once you reply, we will register your issue and attach your recording to the official work order.",
                        parse_mode=ParseMode.MARKDOWN,
                    )

        async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message or not update.message.text:
                return

            raw_text = update.message.text
            user = update.effective_user
            chat = update.effective_chat
            res_id = str(user.id) if user else "unknown"
            username = user.username if user else None
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() if user else None

            is_group = bool(chat and chat.type in ("group", "supergroup"))
            bot_username = getattr(self, "bot_username", None) or (context.bot.username if context.bot else None)

            # Check if bot was mentioned
            has_mention = False
            if bot_username and f"@{bot_username.lower()}" in raw_text.lower():
                has_mention = True
            elif update.message.entities:
                for ent in update.message.entities:
                    if ent.type == "mention":
                        m_txt = raw_text[ent.offset : ent.offset + ent.length].lower()
                        if bot_username and m_txt == f"@{bot_username.lower()}":
                            has_mention = True
                    elif ent.type == "text_mention" and ent.user and ent.user.id == context.bot.id:
                        has_mention = True

            # Check if this is a swipe-to-reply
            reply_msg = update.message.reply_to_message
            is_reply_to_bot = False
            if reply_msg and reply_msg.from_user:
                if reply_msg.from_user.id == context.bot.id:
                    is_reply_to_bot = True
                elif bot_username and reply_msg.from_user.username and reply_msg.from_user.username.lower() == bot_username.lower():
                    is_reply_to_bot = True

            # In group chats: respond if mentioned OR swiped to reply to bot (or replying to someone else with mention)
            if is_group and not (has_mention or is_reply_to_bot):
                return

            clean_text = _clean_bot_mention(raw_text, bot_username)

            # 1. Check if user swiped to reply to a message referencing a Work Order
            if reply_msg:
                replied_text = reply_msg.text or reply_msg.caption or ""
                wo_match = re.search(r"#?(WO-EST-\d{4,6}|EST-\d{4,6})", replied_text)
                if wo_match:
                    raw_wo_id = wo_match.group(1)
                    target_wo_id = raw_wo_id if raw_wo_id.startswith("WO-") else f"WO-{raw_wo_id}"
                    target_wo = self.store.get(target_wo_id)
                    if target_wo:
                        # Case 0: Staff swipe-to-reply from internal staff group!
                        is_staff_group = bool(
                            self.staff_chat_id
                            and str(chat.id).strip() == str(self.staff_chat_id).strip()
                        )
                        if is_staff_group:
                            # Relay directly to resident
                            res = self.store.send_resident_message(
                                target_wo.work_order_id,
                                clean_text,
                                sender_name=f"Estate Officer ({full_name or 'Staff'})",
                            )
                            if res.get("success"):
                                res_handle = f"@{target_wo.resident_username}" if target_wo.resident_username else f"User {target_wo.resident_id}"
                                await update.message.reply_text(
                                    f"*Message Dispatched to Resident*\n\n"
                                    f"Delivered to {res_handle} for Work Order `#{target_wo.work_order_id}`:\n"
                                    f"_{clean_text}_\n\n"
                                    f"Logged in estate audit timeline.",
                                    parse_mode=ParseMode.MARKDOWN,
                                )
                            else:
                                await update.message.reply_text(
                                    f"Could not relay to resident: {res.get('error')}",
                                )
                            return

                        # Case A: Location update
                        if _is_probable_location(clean_text):
                            val_res = validate_estate_location(clean_text)
                            if not val_res.is_valid:
                                await update.message.reply_text(val_res.error_message, parse_mode=ParseMode.MARKDOWN)
                                return
                            updated = self.store.update_location(target_wo.work_order_id, clean_text)
                            if updated:
                                await update.message.reply_text(
                                    f"*Location Recorded:* `{clean_text}`\n\n"
                                    f"Work Order `#{updated.work_order_id}` location has been updated for **{updated.assigned_team}**.",
                                    parse_mode=ParseMode.MARKDOWN,
                                )
                                return

                        # Case B: Status inquiry
                        if re.search(r"\b(status|progress|update|how is it|check)\b", clean_text, re.IGNORECASE) and len(clean_text.split()) <= 6:
                            await update.message.reply_text(
                                self.handle_status_query(target_wo.work_order_id),
                                parse_mode=ParseMode.MARKDOWN,
                            )
                            return

                        # Case C: Pleasantry
                        if _is_conversational_reply(clean_text):
                            await update.message.reply_text(
                                f"You are welcome. We are monitoring Work Order `#{target_wo.work_order_id}`. "
                                f"Feel free to check status anytime with `/status` or reply if you require further assistance.",
                                parse_mode=ParseMode.MARKDOWN,
                            )
                            return

                        # Case D: Additional follow-up note
                        if len(clean_text) >= 2:
                            self.store.update_status(
                                target_wo.work_order_id,
                                target_wo.status,
                                note=f"Resident update: {clean_text}",
                                notify_resident=False,
                            )
                            if self.staff_chat_id:
                                note_text = (
                                    f"[WORK ORDER UPDATE NOTE]\n"
                                    f"• Work Order ID: {target_wo.work_order_id}\n"
                                    f"• Resident: @{username or res_id} ({full_name or 'Resident'})\n"
                                    f"• Note: \"{clean_text}\""
                                )
                                await self._dispatch_to_staff(
                                    context=context,
                                    text=note_text,
                                    category=target_wo.category,
                                    priority=target_wo.priority,
                                )

                            await update.message.reply_text(
                                f"*Update Recorded*\n\n"
                                f"Your note has been added to Work Order `#{target_wo.work_order_id}`:\n"
                                f"_{clean_text}_\n\n"
                                f"Our on-duty estate team has been notified.",
                                parse_mode=ParseMode.MARKDOWN,
                            )
                            return

                # If user swiped to reply to another user's message reporting an issue (e.g. in group)
                if replied_text and not is_reply_to_bot:
                    if len(clean_text.split()) <= 4 or re.search(r"\b(fix|check|report|lodge|help|look into|attend|this)\b", clean_text, re.IGNORECASE):
                        clean_text = f"{replied_text} ({clean_text})" if clean_text else replied_text

            # 1.5 Check for resident quick action buttons
            clean_btn = clean_text.strip().lower()

            # Button: Report Issue (legacy fallback)
            if clean_btn in (
                "🛠️ report issue",
                "🛠 report issue",
                "report issue",
                "report an issue",
                "🛠️ report an issue",
            ):
                await update.message.reply_text(
                    "*REPORT AN ESTATE DEFECT / ISSUE*\n"
                    "──────────────────────────────\n"
                    "Please reply with a brief description of the issue and your unit or location.\n\n"
                    "*Examples:*\n"
                    "• _\"Water leaking from corridor ceiling outside #08-14\"_\n"
                    "• _\"Gym treadmill #2 belt is slipping at Level 2 clubhouse\"_\n"
                    "• _\"Car parked across two lots at B2 near Lot 104\"_\n\n"
                    "📷 *Photo Evidence:* You can also snap or attach a photo directly with your description as the caption!",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=self.get_resident_welcome_markup(),
                )
                return

            # Button: FAQ & Bylaws
            if clean_btn in (
                "faqs & bylaws",
                "faq & bylaws",
                "faqs",
                "faq",
                "📖 faq & bylaws",
                "📖 faq",
                "📖 estate faq",
                "❓ faq & bylaws",
                "❓ faq",
                "condo faq & bylaws",
                "faq / bylaws",
                "condo faq",
                "bylaws",
                "estate faq",
            ):
                await faq_cmd(update, context)
                return

            # Button: My Requests
            if clean_btn in (
                "📋 my requests",
                "my requests",
                "📋 my request",
                "my request",
                "my tickets",
                "my work orders",
            ):
                await myrequests_cmd(update, context)
                return

            # Button: Emergency Hotlines
            if clean_btn in (
                "🚨 24/7 emergency hotlines",
                "24/7 emergency hotlines",
                "🚨 emergency hotlines",
                "emergency hotlines",
                "🚨 hotlines",
                "hotlines",
                "emergency contacts",
                "emergency",
                "🚨 emergency",
            ):
                await emergency_cmd(update, context)
                return

            # 2. Check if text is just a mention / greeting
            if not clean_text or clean_text.lower() in ("hi", "hello", "hey", "help", "@"):
                user_name = user.first_name if user else None
                await update.message.reply_text(
                    self.get_welcome_message(user_name),
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=self.get_resident_welcome_markup(),
                )
                return

            # Check if resident is asking about estate blocks or level information
            if re.search(r"\b(how many (levels|floors)|what (blocks|levels)|which block|estate blocks|block info|levels in block)\b", clean_text, re.IGNORECASE):
                await update.message.reply_text(
                    f"*ESTATE DIRECTORY & LEVELS*\n\n"
                    f"{format_estate_levels_clarification()}\n\n"
                    f"• Common Facilities: Gym, Swimming Pool, Clubhouse, Tennis Court, BBQ Pits, Guardhouse, Carpark (B1 & B2)\n\n"
                    f"To report an issue, simply describe what happened and your unit or location.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # 3. Check if resident is responding to "talking to a person or team?"
            pending_handover = self.session_store.pop_pending_handover(res_id) or self._pending_handover_questions.pop(res_id, None)
            if pending_handover:
                wo_id = pending_handover
                if re.search(r"\b(person|human|officer|staff|talk|speak|call|manager)\b", clean_text, re.IGNORECASE):
                    self.store.update_status(wo_id, "Pending Staff Review", note="Resident requested officer callback")
                    await update.message.reply_text(
                        f"Work Order `#{wo_id}` has been escalated to our on-duty estate manager for direct follow-up.\n\n"
                        f"*Direct Contacts:*\n"
                        f"• Estate Management Office: `+65 6789 0125` (Mon–Fri 9:00 AM – 6:00 PM)\n"
                        f"• 24/7 Guardhouse & Intercom: `+65 6789 0123`",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return
                elif re.search(r"\b(team|leave|fine|ok|okay|all good|no|thanks|thank you)\b", clean_text, re.IGNORECASE):
                    rating_keyboard = [
                        [
                            InlineKeyboardButton("1 Star", callback_data=f"rate:{wo_id}:1"),
                            InlineKeyboardButton("2 Stars", callback_data=f"rate:{wo_id}:2"),
                            InlineKeyboardButton("3 Stars", callback_data=f"rate:{wo_id}:3"),
                            InlineKeyboardButton("4 Stars", callback_data=f"rate:{wo_id}:4"),
                            InlineKeyboardButton("5 Stars", callback_data=f"rate:{wo_id}:5"),
                        ]
                    ]
                    await update.message.reply_text(
                        f"Thank you. Our duty team has taken charge of Work Order `#{wo_id}`. "
                        f"You can check real-time progress anytime via `/status {wo_id}`.\n\n"
                        f"How would you rate our concierge service today?\n\n"
                        f"_Whenever you need to make a new report, simply type `/new` or send a message anytime._",
                        reply_markup=InlineKeyboardMarkup(rating_keyboard),
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return

            # 3.5 Check if resident is replying to a pending noise clarification
            pending_noise = self.session_store.get_pending_noise(res_id) or self._pending_noise_requests.get(res_id)
            if pending_noise and not clean_text.startswith("/"):
                # Resident is providing noise timing / location details!
                self.session_store.pop_pending_noise(res_id)
                self._pending_noise_requests.pop(res_id, None)

                orig_text = pending_noise.get("original_text", "")
                orig_loc = pending_noise.get("location_hint")

                # Combine resident's reply with original complaint
                combined_report = f"{orig_text}. {clean_text}"

                # Extract timing details
                noise_info = extract_noise_details(combined_report)
                if not noise_info.start_time:
                    noise_info_reply = extract_noise_details(f"noise {clean_text}")
                    if noise_info_reply.start_time:
                        noise_info = noise_info._replace(start_time=noise_info_reply.start_time)
                if not noise_info.duration:
                    noise_info_reply = extract_noise_details(f"noise {clean_text}")
                    if noise_info_reply.duration:
                        noise_info = noise_info._replace(duration=noise_info_reply.duration)

                # Extract estate coordinates (block, level, unit, etc.)
                comps = extract_estate_components(combined_report, orig_loc)

                # Check level validity if level/block was mentioned
                val_res = validate_estate_location(clean_text)
                if not val_res.is_valid and any(w in clean_text.lower() for w in ("block", "blk", "level", "lvl", "floor", "#")):
                    # Re-store pending noise and ask user to resend valid level
                    self.session_store.set_pending_noise(res_id, pending_noise)
                    self._pending_noise_requests[res_id] = pending_noise
                    await update.message.reply_text(val_res.error_message, parse_mode=ParseMode.MARKDOWN)
                    return

                # Build structured reason
                timing_parts = []
                if noise_info.start_time:
                    timing_parts.append(f"Started: {noise_info.start_time}")
                if noise_info.duration:
                    timing_parts.append(f"Duration: {noise_info.duration}")
                timing_str = f" ({', '.join(timing_parts)})" if timing_parts else ""

                base_reason = noise_info.noise_type.title() if noise_info.noise_type else "Noise disturbance"
                full_reason = f"{base_reason} reported at night{timing_str}"

                final_loc = comps.formatted_location if comps.formatted_location and comps.formatted_location != combined_report else (orig_loc or "Pending confirmation")

                # Create Work Order directly with Category.NOISE
                full_issue_text = f"Noise disturbance: {orig_text}. Details: {clean_text}{timing_str}"
                req = ResidentRequest(
                    text=full_issue_text,
                    channel="Telegram",
                    location=final_loc,
                )
                dec = self.triage_service.classify(req).model_copy(
                    update={
                        "category": Category.NOISE,
                        "category_title": "Noise & nuisance",
                        "assigned_team": "Security desk & Resident relations",
                        "priority": Priority.HIGH,
                        "sla_target": "Within 30 minutes (Active night noise patrol)",
                    }
                )
                wo = self.store.create_work_order(
                    request=req,
                    classification=dec,
                    resident_id=res_id,
                    resident_username=username,
                    resident_name=full_name,
                    resident_chat_id=res_id,
                    notify_staff=False,
                )
                if comps.block:
                    wo.block = comps.block
                if comps.level:
                    wo.level = comps.level
                if comps.unit:
                    wo.unit = comps.unit
                wo.issue_reason = full_reason

                self.session_store.set_last_work_order(res_id, wo.work_order_id)

                has_loc = is_concrete_estate_location(wo.location, combined_report)
                if not has_loc:
                    self.session_store.set_pending_location(res_id, wo.work_order_id)
                    if is_ambiguous_carpark_location(wo.location, combined_report):
                        carpark_map_path = Path(__file__).resolve().parent.parent / "output" / "estate_carpark_map.jpg"
                        carpark_prompt = (
                            f"*CARPARK LEVEL REQUIRED FOR NOISE PATROL*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"We have registered your noise report: \"{wo.text}\"\n\n"
                            f"• *Reference:* `#{wo.work_order_id}`\n"
                            f"• *Team:* {wo.assigned_team}\n"
                            f"• *Status:* Carpark Location Details Required\n\n"
                            f"Our estate has multiple underground carpark levels (**Basement 1** & **Basement 2**).\n\n"
                            f"Please check the attached directory map and reply with:\n"
                            f"1. *Level:* **B1** or **B2**\n"
                            f"2. *Pillar / Lot / Lobby:* (e.g. `B1 near Lot 20` or `B2 Tower B Lobby`)\n\n"
                            f"Once provided, our security patrol will be dispatched immediately to investigate."
                        )
                        if carpark_map_path.exists():
                            try:
                                with open(carpark_map_path, "rb") as map_file:
                                    await update.message.reply_photo(
                                        photo=map_file,
                                        caption=carpark_prompt,
                                        parse_mode=ParseMode.MARKDOWN,
                                    )
                                    return
                            except Exception as exc:
                                logger.warning(f"Failed to send carpark directory photo: {exc}")

                        await update.message.reply_text(carpark_prompt, parse_mode=ParseMode.MARKDOWN)
                        return

                    noise_loc_dict = build_quick_location_markup(wo.work_order_id, is_noise=True)
                    noise_inline_rows = [
                        [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                        for row in noise_loc_dict.get("inline_keyboard", [])
                    ]
                    noise_loc_markup = InlineKeyboardMarkup(noise_inline_rows) if noise_inline_rows else None
                    await update.message.reply_text(
                        f"*NOISE DISTURBANCE REPORT REGISTERED*\n\n"
                        f"We have registered your noise report: \"{wo.text}\"\n\n"
                        f"• *Reference:* `#{wo.work_order_id}`\n"
                        f"• *Team:* {wo.assigned_team}\n"
                        f"• *Status:* Location Details Required\n\n"
                        f"Please reply with the specific unit or block where the disturbance is occurring (e.g. `Block C, #08-12`).\n\n"
                        f"{format_estate_levels_clarification()}\n\n"
                        f"Once provided, your report will be dispatched immediately to our on-duty security patrol.",
                        reply_markup=noise_loc_markup,
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return

                card_text = format_telegram_resident_card(wo)
                markup_dict = build_resident_card_markup(wo)
                inline_rows = [
                    [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                    for row in markup_dict.get("inline_keyboard", [])
                ]
                markup = InlineKeyboardMarkup(inline_rows) if inline_rows else None

                try:
                    await update.message.reply_text(card_text, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)
                except Exception:
                    await update.message.reply_text(card_text, reply_markup=markup)

                # Send staff alert
                if self.staff_chat_id:
                    staff_alert = format_staff_dispatch_alert(wo)
                    staff_markup_dict = build_staff_alert_markup(wo)
                    staff_markup = InlineKeyboardMarkup([
                        [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                        for row in staff_markup_dict.get("inline_keyboard", [])
                    ])
                    await self._dispatch_to_staff(
                        context=context,
                        text=staff_alert,
                        reply_markup=staff_markup,
                        category=Category.NOISE,
                        priority=wo.priority,
                    )
                return

            # 4. Check if resident is providing the missing location
            pending_loc_wo = self.session_store.get_session(res_id).get("pending_location_wo_id") or self._pending_location_requests.get(res_id)
            if pending_loc_wo and _is_probable_location(clean_text):
                clean_loc = clean_text.strip()
                if clean_loc in ("1", "2", "3", "4"):
                    clean_loc = f"BBQ Pit {clean_loc}"
                elif re.match(r"^pit\s*([1-4])$", clean_loc, re.IGNORECASE):
                    m = re.match(r"^pit\s*([1-4])$", clean_loc, re.IGNORECASE)
                    clean_loc = f"BBQ Pit {m.group(1)}"

                # Check if resident replied with ambiguous carpark (e.g. "carpark" or "in carpark")
                if is_ambiguous_carpark_location(clean_loc):
                    carpark_map_path = Path(__file__).resolve().parent.parent / "output" / "estate_carpark_map.jpg"
                    carpark_prompt = (
                        f"*CARPARK LEVEL & LOCATION REQUIRED*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"Our estate has multiple underground carpark levels:\n"
                        f"• *Basement 1 (B1):* Visitor Lots (V01–V20), EV Charging, Resident Lots #01–#80\n"
                        f"• *Basement 2 (B2):* Resident Season Lots #81–#180, Car Wash Bay\n\n"
                        f"Please check the carpark directory map above and reply with:\n"
                        f"1. *Level:* **B1** or **B2**\n"
                        f"2. *Pillar or Lot Number:* (e.g. `B1 near Lot 42` or `B2 near Pillar BB3`)\n\n"
                        f"Once provided, our team will be dispatched directly to the exact spot."
                    )
                    if carpark_map_path.exists():
                        try:
                            with open(carpark_map_path, "rb") as map_file:
                                await update.message.reply_photo(
                                    photo=map_file,
                                    caption=carpark_prompt,
                                    parse_mode=ParseMode.MARKDOWN,
                                )
                                return
                        except Exception as exc:
                            logger.warning(f"Failed to send carpark directory photo: {exc}")

                    await update.message.reply_text(carpark_prompt, parse_mode=ParseMode.MARKDOWN)
                    return

                val_res = validate_estate_location(clean_loc)
                if not val_res.is_valid:
                    await update.message.reply_text(val_res.error_message, parse_mode=ParseMode.MARKDOWN)
                    return
                wo_id = self.session_store.pop_pending_location(res_id) or self._pending_location_requests.pop(res_id, None)
                updated_wo = self.store.update_location(wo_id, clean_loc)
                if updated_wo:
                    # Dispatch complete ticket to staff topic now that location is confirmed!
                    if self.staff_chat_id:
                        staff_alert = format_staff_dispatch_alert(updated_wo)
                        staff_markup_dict = build_staff_alert_markup(updated_wo)
                        staff_markup = InlineKeyboardMarkup([
                            [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                            for row in staff_markup_dict.get("inline_keyboard", [])
                        ])
                        await self._dispatch_to_staff(
                            context=context,
                            text=staff_alert,
                            reply_markup=staff_markup,
                            category=updated_wo.category,
                            priority=updated_wo.priority,
                        )

                    reply_msg_text = (
                        f"*Location Confirmed:* `{clean_text.strip()}`\n\n"
                        f"Thank you for providing your location details. Your Work Order `#{updated_wo.work_order_id}` "
                        f"has been dispatched to our **{updated_wo.assigned_team}** for prompt attention.\n\n"
                        f"• *Priority:* {updated_wo.priority.value.upper()} (Estimated SLA: {updated_wo.sla_target})\n"
                        f"• *Status:* In Progress\n\n"
                        f"• Track live status: `/status {updated_wo.work_order_id}`\n"
                        f"• Make another report: `/new`"
                    )
                    await update.message.reply_text(
                        reply_msg_text,
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return

            # 5. Direct request to speak with a human/person
            if re.search(r"\b(talk\s+to\s+a\s+person|speak\s+to\s+a\s+(person|human)|human\s+agent|contact\s+a\s+person|talk\s+to\s+human)\b", clean_text, re.IGNORECASE):
                await update.message.reply_text(
                    "Our estate management team is at your service.\n\n"
                    "*Direct Contacts:*\n"
                    "• Estate Management Office: `+65 6789 0125` (Mon–Fri 9:00 AM – 6:00 PM)\n"
                    "• 24/7 Security Guardhouse: `+65 6789 0123`\n"
                    "• Lift Emergency Desk: `+65 6789 0124`\n\n"
                    "If you have an active ticket number (e.g., `WO-EST-1042`), please let me know and I will flag it for our on-duty manager.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # 6. Filter casual greetings and pleasantries ("okay", "thanks", "hello")
            if _is_conversational_reply(clean_text):
                last_wo_id = self.session_store.get_last_work_order(res_id)
                clean_norm = re.sub(r"[^\w\s]", "", clean_text.strip().lower())
                if clean_norm in ("ok", "okay", "k", "noted", "got it", "thanks", "thank you", "thx", "sure", "alright", "good", "bye", "goodbye"):
                    if last_wo_id:
                        self.store.update_status(last_wo_id, self.store.get(last_wo_id).status if self.store.get(last_wo_id) else "In Progress", note=f"Resident acknowledged: \"{clean_text.strip()}\"", notify_resident=False)
                        rating_keyboard = [
                            [
                                InlineKeyboardButton("1 Star", callback_data=f"rate:{last_wo_id}:1"),
                                InlineKeyboardButton("2 Stars", callback_data=f"rate:{last_wo_id}:2"),
                                InlineKeyboardButton("3 Stars", callback_data=f"rate:{last_wo_id}:3"),
                                InlineKeyboardButton("4 Stars", callback_data=f"rate:{last_wo_id}:4"),
                                InlineKeyboardButton("5 Stars", callback_data=f"rate:{last_wo_id}:5"),
                            ]
                        ]
                        await update.message.reply_text(
                            f"You are most welcome. Our duty team is attending to Work Order `#{last_wo_id}`.\n\n"
                            f"How would you rate our concierge service today?\n\n"
                            f"_Whenever you need to make a new report or inquiry, simply type `/new` or send a message anytime._",
                            reply_markup=InlineKeyboardMarkup(rating_keyboard),
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    else:
                        await update.message.reply_text(
                            "You are most welcome! How may our estate management team assist you today?\n\n"
                            "_Whenever you need to make a new report or inquiry, simply type `/new` or describe it here anytime._",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    return
                else:
                    user_name = f", {user.first_name}" if user and user.first_name else ""
                    await update.message.reply_text(
                        f"Good day{user_name}. How may our estate management team assist you today?\n\n"
                        "To report a maintenance or estate issue, simply send a description with your unit or location "
                        "(e.g. _'Corridor light flickering outside #08-14'_ or _'Gym aircon leaking water'_).\n\n"
                        "• Check ticket: `/status`\n"
                        "• Your tickets: `/myrequests`\n"
                        "• Emergency hotlines: `/emergency`",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return

            # Check if there is a pending photo waiting to be attached
            pending_photo = self.session_store.pop_pending_photo(res_id)
            await _handle_text_report(update, context, clean_text, photo_path=pending_photo)

        async def _handle_text_report(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            text_body: str,
            photo_path: str | None = None,
            is_transcribed: bool = False,
        ) -> None:
            user = update.effective_user
            res_id = str(user.id) if user else "unknown"
            username = user.username if user else None
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() if user else None

            # If text is in a foreign language, translate to English for internal triage and staff dispatch
            det_lang, det_lang_name = detect_language(text_body)
            original_input_text = text_body
            english_text = text_body
            if det_lang != "en":
                trans_res = translate_text(text_body, target_lang="en", source_lang=det_lang)
                if trans_res.is_translated:
                    english_text = trans_res.translated_text

            # 1. First check if resident is asking a condo enquiry (RAG Knowledge Base)
            rag_answer = answer_condo_enquiry(english_text) or answer_condo_enquiry(text_body)
            if rag_answer and rag_answer.get("answered"):
                rag_reply = rag_answer["reply"]
                if rag_answer.get("article_id") == "FAC-BBQ-CLEAN":
                    pit_match = re.search(r"\b(?:bbq\s*pit|pit)\s*([1-4])\b", text_body, re.IGNORECASE)
                    if pit_match:
                        pit_num = pit_match.group(1)
                        issue_text = f"BBQ Pit cleanliness issue: previous booking did not clean up. Cleaners required for BBQ Pit {pit_num}."
                        req = ResidentRequest(
                            text=issue_text,
                            channel="Telegram",
                            location=f"BBQ Pit {pit_num}",
                        )
                        dec = self.triage_service.classify(req).model_copy(
                            update={
                                "category": Category.CLEANLINESS,
                                "category_title": "Cleanliness & pest control",
                                "assigned_team": "Housekeeping & waste management",
                                "priority": Priority.HIGH,
                                "sla_target": "Within 30 minutes (Duty Cleaners)",
                            }
                        )
                        wo = self.store.create_work_order(
                            request=req,
                            classification=dec,
                            resident_id=res_id,
                            resident_username=username,
                            resident_name=full_name,
                            resident_chat_id=res_id,
                            notify_staff=False,
                        )
                        wo.issue_reason = f"Previous user did not clean up BBQ Pit {pit_num}. Flag previous booking for $50 deposit deduction."
                        self.session_store.set_last_work_order(res_id, wo.work_order_id)

                        dispatch_msg = (
                            f"*Housekeeping Dispatched: BBQ Pit {pit_num}*\n\n"
                            f"• *Ticket:* `#{wo.work_order_id}`\n"
                            f"• Cleaners are on the way to scrub and sanitize BBQ Pit {pit_num} (SLA: Within 30 minutes).\n"
                            f"• The previous user's $50 cleaning deposit will be forfeited following booking and CCTV review.\n\n"
                            f"_Guardhouse 24/7 hotline: +65 6789 0123_"
                        )
                        await update.message.reply_text(dispatch_msg, parse_mode=ParseMode.MARKDOWN)

                        if self.staff_chat_id:
                            staff_alert = format_staff_dispatch_alert(wo)
                            staff_markup_dict = build_staff_alert_markup(wo)
                            staff_markup = InlineKeyboardMarkup([
                                [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                                for row in staff_markup_dict.get("inline_keyboard", [])
                            ])
                            await self._dispatch_to_staff(
                                context=context,
                                text=staff_alert,
                                reply_markup=staff_markup,
                                category=Category.CLEANLINESS,
                                priority=Priority.HIGH,
                            )
                        return
                    else:
                        issue_text = f"BBQ Pit cleanliness issue: previous booking did not clean up ({text_body})."
                        req = ResidentRequest(
                            text=issue_text,
                            channel="Telegram",
                            location="BBQ Pavilion",
                        )
                        dec = self.triage_service.classify(req).model_copy(
                            update={
                                "category": Category.CLEANLINESS,
                                "category_title": "Cleanliness & pest control",
                                "assigned_team": "Housekeeping & waste management",
                                "priority": Priority.HIGH,
                                "sla_target": "Within 30 minutes (Duty Cleaners)",
                            }
                        )
                        wo = self.store.create_work_order(
                            request=req,
                            classification=dec,
                            resident_id=res_id,
                            resident_username=username,
                            resident_name=full_name,
                            resident_chat_id=res_id,
                            notify_staff=False,
                        )
                        wo.issue_reason = "Previous user did not clean up BBQ pit. Cleaners required on-site. Flag previous booking for $50 deposit deduction."
                        self.session_store.set_last_work_order(res_id, wo.work_order_id)
                        self.session_store.set_pending_location(res_id, wo.work_order_id)
                        self._pending_location_requests[res_id] = wo.work_order_id

                        reply_text = (
                            f"*BBQ Pit Cleanliness & Deposit Notice*\n"
                            f"──────────────────────────────\n"
                            f"• The previous user's $50 cleaning deposit will be forfeited following CCTV and booking log review.\n"
                            f"• Our on-duty cleaners will be dispatched immediately to clean and sanitize the pit for your session.\n\n"
                            f"*Which BBQ Pit are you currently at?*\n"
                            f"Please reply with your BBQ pit number (e.g. `Pit 1` to `4`) to dispatch our cleaners."
                        )
                        await update.message.reply_text(
                            reply_text,
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        return

                await update.message.reply_text(rag_answer["reply"], parse_mode=ParseMode.MARKDOWN)
                if self.staff_chat_id:
                    enquiry_log = format_staff_enquiry_log(
                        resident_name=full_name or "Resident",
                        resident_username=username,
                        resident_id=res_id,
                        query=text_body,
                        answer_result=rag_answer,
                    )
                    await self._dispatch_enquiry_log_to_staff(context, enquiry_log)
                return

            # Validate location in report against estate directory
            val_res = validate_estate_location(english_text) or validate_estate_location(text_body)
            if not val_res.is_valid:
                await update.message.reply_text(val_res.error_message, parse_mode=ParseMode.MARKDOWN)
                return

            # Check if this is a noise disturbance complaint
            extracted_loc = extract_location(english_text) or extract_location(text_body)
            if is_noise_complaint(english_text) or is_noise_complaint(text_body):
                noise_details = extract_noise_details(english_text)
                if not noise_details.has_timing_details:
                    # Save pending noise in session and ask clarifying questions
                    self.session_store.set_pending_noise(
                        res_id,
                        {
                            "original_text": english_text,
                            "location_hint": extracted_loc,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    self._pending_noise_requests[res_id] = {
                        "original_text": english_text,
                        "location_hint": extracted_loc,
                    }
                    prompt = format_noise_clarification_prompt(location_str=extracted_loc)
                    await update.message.reply_text(prompt, parse_mode=ParseMode.MARKDOWN)
                    return

            preview_req = ResidentRequest(
                text=english_text,
                channel="Telegram",
                location=extracted_loc,
            )
            preview_decision = self.triage_service.classify(preview_req)
            is_emergency = preview_decision.priority == Priority.EMERGENCY or preview_decision.nouls.get("life_safety_hazard", 0.0) >= 0.50

            # Only reject if not an emergency and category is GENERAL or OTHER without noise complaint
            if not is_emergency and preview_decision.category in (Category.GENERAL, Category.OTHER) and not is_noise_complaint(english_text):
                unrec_msg = (
                    "*CONDOMINIUM RESIDENT CONCIERGE*\n\n"
                    "I didn't recognize that as an active estate defect or recognized condo policy enquiry.\n\n"
                    "I am programmed to assist with:\n"
                    "• *Reporting Estate Defects:* Maintenance, plumbing, electrical faults, leaks, lift issues, or noise disturbances (e.g. _\"Water leaking outside #08-14\"_).\n"
                    "• *Condo Rules & Bylaws:* Type `/faq` or ask about pool hours, BBQ booking, visitor parking, renovation hours, or pet policies.\n\n"
                    "*Direct Contacts:*\n"
                    "• Estate Management Office: `+65 6789 0125` (Mon–Fri 9:00 AM – 6:00 PM)\n"
                    "• 24/7 Security Guardhouse: `+65 6789 0123`"
                )
                await update.message.reply_text(unrec_msg, parse_mode=ParseMode.MARKDOWN)
                return

            try:
                # Process triage
                wo, card_text = self.process_resident_report(
                    text=english_text,
                    resident_id=res_id,
                    resident_username=username,
                    resident_name=full_name,
                    photo_path=photo_path,
                )
                wo.original_text = original_input_text
                wo.detected_language = det_lang
                wo.is_transcribed_audio = is_transcribed

                # LOCATION GATING: Non-emergencies require concrete location before staff dispatch
                has_loc = is_concrete_estate_location(wo.location, english_text)
                if not has_loc and wo.priority != Priority.EMERGENCY:
                    self.session_store.set_pending_location(res_id, wo.work_order_id)
                    self._pending_location_requests[res_id] = wo.work_order_id

                    # Check if carpark is mentioned without specific level / lot
                    if is_ambiguous_carpark_location(wo.location, english_text):
                        carpark_map_path = Path(__file__).resolve().parent.parent / "output" / "estate_carpark_map.jpg"
                        carpark_prompt = (
                            f"*CARPARK LEVEL & LOCATION REQUIRED*\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"We have registered your issue: \"{wo.text}\"\n\n"
                            f"• *Reference:* `#{wo.work_order_id}`\n"
                            f"• *Assigned Team:* {wo.assigned_team}\n"
                            f"• *Status:* Carpark Location Details Required\n\n"
                            f"Our estate has multiple underground carpark levels:\n"
                            f"• *Basement 1 (B1):* Visitor Lots (V01–V20), EV Charging, Resident Lots #01–#80\n"
                            f"• *Basement 2 (B2):* Resident Season Lots #81–#180, Car Wash Bay\n\n"
                            f"Please check the carpark directory map above and reply with:\n"
                            f"1. *Level:* **B1** or **B2**\n"
                            f"2. *Pillar or Lot Number:* (e.g. `B1 near Lot 42` or `B2 near Pillar BB3`)\n\n"
                            f"Once provided, our team will be dispatched directly to the exact spot."
                        )
                        if carpark_map_path.exists():
                            try:
                                with open(carpark_map_path, "rb") as map_file:
                                    await update.message.reply_photo(
                                        photo=map_file,
                                        caption=carpark_prompt,
                                        parse_mode=ParseMode.MARKDOWN,
                                    )
                                    return
                            except Exception as exc:
                                logger.warning(f"Failed to send carpark directory photo: {exc}")

                        await update.message.reply_text(carpark_prompt, parse_mode=ParseMode.MARKDOWN)
                        return

                    loc_dict = build_quick_location_markup(wo.work_order_id, is_noise=False)
                    loc_inline_rows = [
                        [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                        for row in loc_dict.get("inline_keyboard", [])
                    ]
                    loc_markup = InlineKeyboardMarkup(loc_inline_rows) if loc_inline_rows else None
                    loc_prompt = (
                        f"*MAINTENANCE REPORT REGISTERED*\n\n"
                        f"We have registered your issue: \"{wo.text}\"\n\n"
                        f"• *Reference:* `#{wo.work_order_id}`\n"
                        f"• *Assigned Team:* {wo.assigned_team}\n"
                        f"• *Status:* Location Details Required\n\n"
                        f"Please reply with your unit or block number (e.g. `Block B, #08-14` or `Clubhouse Level 2`).\n\n"
                        f"{format_estate_levels_clarification()}\n\n"
                        f"Once you provide your location, this ticket will be dispatched immediately to our on-duty team."
                    )
                    await update.message.reply_text(loc_prompt, reply_markup=loc_markup, parse_mode=ParseMode.MARKDOWN)
                    return

                # Send confirmation card without interactive inline buttons
                markup_dict = build_resident_card_markup(wo)
                inline_rows = [
                    [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                    for row in markup_dict.get("inline_keyboard", [])
                ]
                markup = InlineKeyboardMarkup(inline_rows) if inline_rows else None

                try:
                    await update.message.reply_text(card_text, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)
                except Exception:
                    # Fallback if markdown triggers parsing anomaly
                    await update.message.reply_text(card_text, reply_markup=markup)

                # If staff chat ID is configured, forward dispatch brief with one-touch action buttons
                if self.staff_chat_id:
                    staff_alert = format_staff_dispatch_alert(wo)
                    staff_markup_dict = build_staff_alert_markup(wo)
                    staff_markup = InlineKeyboardMarkup([
                        [InlineKeyboardButton(btn["text"], callback_data=btn["callback_data"]) for btn in row]
                        for row in staff_markup_dict.get("inline_keyboard", [])
                    ])
                    await self._dispatch_to_staff(
                        context=context,
                        text=staff_alert,
                        reply_markup=staff_markup,
                        category=wo.category,
                        priority=wo.priority,
                        is_emergency=(wo.priority == Priority.EMERGENCY),
                        attachment_path=wo.photo_path,
                    )
            except Exception as exc:
                logger.error(f"Error handling text report: {exc}")
                await update.message.reply_text(
                    "Please describe the maintenance issue you would like to report, including your unit or location "
                    "(e.g. _'Corridor light flickering outside #08-14'_). Type `/help` for guidance.",
                    parse_mode=ParseMode.MARKDOWN,
                )

        async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
            logger.error(f"Telegram handler exception: {context.error}")

        # Register handlers
        app.add_error_handler(global_error_handler)
        app.add_handler(CommandHandler("start", start_cmd))
        app.add_handler(CommandHandler("help", help_cmd))
        app.add_handler(CommandHandler("faq", faq_cmd))
        app.add_handler(CommandHandler("emergency", emergency_cmd))
        app.add_handler(CommandHandler("lang", lang_cmd))
        app.add_handler(CommandHandler("language", lang_cmd))
        app.add_handler(CommandHandler("status", status_cmd))
        app.add_handler(CommandHandler("myrequests", myrequests_cmd))
        app.add_handler(CommandHandler("myworkorders", myrequests_cmd))
        app.add_handler(CommandHandler("report", report_cmd))
        app.add_handler(CommandHandler("new", report_cmd))
        app.add_handler(CommandHandler("newticket", report_cmd))
        app.add_handler(CommandHandler("ticket", report_cmd))
        app.add_handler(CommandHandler("receipt", receipt_cmd))
        app.add_handler(CommandHandler("staff", staff_cmd))
        app.add_handler(CommandHandler("pending", pending_cmd))
        app.add_handler(CommandHandler("techs", techs_cmd))
        app.add_handler(CommandHandler("assign", assign_cmd))
        app.add_handler(CommandHandler("progress", progress_cmd))
        app.add_handler(CommandHandler("in_progress", progress_cmd))
        app.add_handler(CommandHandler("resolve", resolve_cmd))
        app.add_handler(CommandHandler("note", note_cmd))
        app.add_handler(CommandHandler("delete", delete_cmd))
        app.add_handler(CommandHandler("chatid", chatid_cmd))
        app.add_handler(CommandHandler("id", chatid_cmd))
        app.add_handler(CommandHandler("topics", topics_cmd))
        app.add_handler(CommandHandler("topic", topics_cmd))
        app.add_handler(CommandHandler("settopic", settopic_cmd))
        async def new_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message:
                return
            chat = update.effective_chat
            if not chat:
                return
            new_members = update.message.new_chat_members or []
            bot_id = context.bot.id
            if any(m.id == bot_id for m in new_members):
                print(f"[BOT ADDED TO GROUP] Title: '{chat.title}', Chat ID: {chat.id}", flush=True)
                await update.message.reply_text(
                    f"*CONDO MANAGEMENT BOT CONNECTED*\n"
                    f"──────────────────────────────\n"
                    f"• *Group Title:* {chat.title}\n"
                    f"• *Staff Group Chat ID:* `{chat.id}`\n\n"
                    f"To enable staff dispatch alerts, set in `.env`:\n"
                    f"```bash\nTELEGRAM_STAFF_CHAT_ID={chat.id}\n```\n\n"
                    f"• Type `/topics` to see current topic routes.\n"
                    f"• Type `/settopic <category>` inside any topic to link it!",
                    parse_mode=ParseMode.MARKDOWN,
                )

        app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_chat_member_handler))
        app.add_handler(CallbackQueryHandler(callback_handler))
        app.add_handler(MessageHandler(filters.LOCATION, location_handler))
        app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
        app.add_handler(MessageHandler(filters.VIDEO, video_handler))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, audio_handler))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

        return app

    def run_polling(self) -> None:
        """Start resilient 24/7 long-polling mode with auto-reconnection and exponential backoff."""
        print("==================================================", flush=True)
        print("  CONDO MANAGEMENT AGENT — TELEGRAM BOT SERVICE   ", flush=True)
        print("==================================================", flush=True)
        print("Bot starting in resilient 24/7 long-polling mode...", flush=True)
        if self.staff_chat_id:
            print(f"Staff dispatch notifications enabled for chat: {self.staff_chat_id}", flush=True)
        print("Press Ctrl+C to terminate.", flush=True)

        consecutive_errors = 0
        while True:
            try:
                app = self.build_application()
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now_str}] Online & listening for resident messages...", flush=True)
                consecutive_errors = 0
                app.run_polling(drop_pending_updates=False)
                break
            except KeyboardInterrupt:
                print("\n[Service Stopped] Telegram bot service terminated by user.", flush=True)
                break
            except Exception as exc:
                consecutive_errors += 1
                backoff = min(60, 2 ** min(consecutive_errors, 5))
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.error(f"Telegram polling exception: {exc}. Reconnecting in {backoff}s...")
                print(f"[{now_str}] Connection issue: {exc}. Auto-reconnecting in {backoff}s...", flush=True)
                time.sleep(backoff)


# Convenient alias for developer imports and test suites
TelegramBot = CondoManagementTelegramBot


def run_test_simulation(sample_text: str, location: str | None = None) -> None:
    """Run simulated resident request triage without requiring live Telegram API."""
    bot = CondoManagementTelegramBot()
    print("=" * 65)
    print("  TELEGRAM BOT MANAGEMENT AGENT — TEST SIMULATION")
    print("=" * 65)
    print(f"Incoming Resident Message:\n  \"{sample_text}\"")
    if location:
        print(f"Provided Location: {location}")
    print("-" * 65)

    wo, card_text = bot.process_resident_report(
        text=sample_text,
        resident_id="tg_mock_987654",
        resident_username="test_resident",
        resident_name="Simulated Resident",
        location_hint=location,
    )

    print("\n[1] GENERATED WORK ORDER RECORD:")
    print(f"  • Work Order ID : {wo.work_order_id}")
    print(f"  • Ticket ID     : {wo.ticket_id}")
    print(f"  • Category      : {wo.category.value} ({wo.category_title})")
    print(f"  • Assigned Team : {wo.assigned_team}")
    print(f"  • Priority      : {wo.priority.value.upper()} (Score: {wo.urgency_score:.0%})")
    print(f"  • Target SLA    : {wo.sla_target}")
    print(f"  • Extracted Loc : {wo.location or 'None'}")
    print(f"  • Status        : {wo.status}")
    print(f"  • Needs Review  : {wo.needs_human_review}")
    if wo.review_reasons:
        print(f"  • Review Reasons: {wo.review_reasons}")

    print("\n[2] RESIDENT TELEGRAM CARD (PREVIEW):")
    print("-" * 65)
    print(card_text)
    print("-" * 65)

    print("\n[3] STAFF DISPATCH ALERT (PREVIEW):")
    print("-" * 65)
    print(format_staff_dispatch_alert(wo))
    print("-" * 65)


def run_interactive_cli() -> None:
    """Run an interactive terminal session mimicking resident Telegram chats."""
    bot = CondoManagementTelegramBot()
    print("==================================================")
    print("  CONDO MANAGEMENT TELEGRAM BOT — INTERACTIVE CLI  ")
    print("==================================================")
    print(bot.get_welcome_message("Resident"))
    print("\nType your issue below (or type 'quit' / 'exit'):\n")

    user_id = "cli_tester_01"
    while True:
        try:
            line = input("\n[Resident Telegram]> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("quit", "exit"):
            break

        if line.startswith("/status"):
            query = line.replace("/status", "").strip()
            print("\n" + bot.handle_status_query(query or "WO-EST-1044"))
        elif line == "/myrequests":
            text = bot.handle_my_requests(user_id)
            print("\n" + text)
        elif line == "/emergency":
            print("\n" + bot.get_emergency_message())
        elif line == "/staff":
            print("\n" + bot.get_staff_dashboard())
        elif line.startswith("/new") or line.startswith("/report"):
            content = re.sub(r"^/(newticket|ticket|new|report)\s*", "", line).strip()
            if not content:
                print("\nPlease provide your issue after /new (e.g. /new Water leaking at #08-14)")
            else:
                wo, card = bot.process_resident_report(
                    text=content,
                    resident_id=user_id,
                    resident_username="cli_resident",
                    resident_name="Resident Tester",
                )
                print("\n[Bot Telegram Response]:")
                print(card)
        else:
            wo, card = bot.process_resident_report(
                text=line,
                resident_id=user_id,
                resident_username="cli_resident",
                resident_name="Resident Tester",
            )
            print("\n[Bot Telegram Response]:")
            print(card)


def main() -> None:
    load_dotenv(".env")
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        load_dotenv(".env.example")
    parser = argparse.ArgumentParser(
        description="Condominium Management Agent Telegram Bot.",
    )
    parser.add_argument(
        "--test",
        nargs="?",
        const="The corridor light outside #05-12 is flickering and going off.",
        help="Run a test triage on sample resident message.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive terminal session mimicking the Telegram Bot.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Telegram Bot Token (or set TELEGRAM_BOT_TOKEN in .env).",
    )
    parser.add_argument(
        "--staff-chat",
        default=None,
        help="Staff Telegram Chat ID for internal dispatch alerts.",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive_cli()
        return

    if args.test:
        run_test_simulation(args.test)
        return

    token = (
        args.token
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or "8846460366:AAFdtbHpsotNqHRuwT_TORvSRlB0JJ0YhcA"
    )
    staff_chat = (
        args.staff_chat
        or os.getenv("TELEGRAM_STAFF_CHAT_ID")
        or "-1004358326140"
    )

    bot = CondoManagementTelegramBot(
        token=token,
        staff_chat_id=staff_chat,
    )
    bot.run_polling()


if __name__ == "__main__":
    main()
