import tempfile
from pathlib import Path

import pytest

from condo_jev.backends import LocalJevSimulatorBackend
from condo_jev.schema import Category, Priority, ResidentRequest
from condo_jev.telegram_bot import (
    CondoManagementTelegramBot,
    TelegramSessionStore,
    _clean_bot_mention,
    _is_conversational_reply,
    _is_probable_location,
)
from condo_jev.work_order import (
    WorkOrder,
    WorkOrderStatus,
    WorkOrderStore,
    format_staff_dispatch_alert,
    format_telegram_resident_card,
)
from condo_jev.workflow import JevTriageService


@pytest.fixture
def temp_store(tmp_path: Path) -> WorkOrderStore:
    test_json = tmp_path / "test_work_orders.json"
    return WorkOrderStore(file_path=test_json)


@pytest.fixture
def triage_service() -> JevTriageService:
    return JevTriageService(backend=LocalJevSimulatorBackend())


def test_work_order_creation_and_retrieval(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(
        text="The corridor light outside #08-14 keeps flickering.",
        channel="Telegram",
        location="#08-14",
    )
    decision = triage_service.classify(req)
    wo = temp_store.create_work_order(
        request=req,
        classification=decision,
        resident_id="tg_12345",
        resident_username="john_doe",
        resident_name="John Doe",
    )

    assert wo.work_order_id.startswith("WO-EST-")
    assert wo.category == Category.MAINTENANCE
    assert wo.location == "#08-14"
    assert wo.resident_id == "tg_12345"
    assert wo.channel == "Telegram"

    # Lookup by full WO id
    fetched = temp_store.get(wo.work_order_id)
    assert fetched is not None
    assert fetched.work_order_id == wo.work_order_id

    # Lookup by ticket id
    by_ticket = temp_store.get(wo.ticket_id)
    assert by_ticket is not None
    assert by_ticket.work_order_id == wo.work_order_id

    # Lookup by numeric part
    numeric = wo.ticket_id.replace("EST-", "")
    by_num = temp_store.get(numeric)
    assert by_num is not None


def test_work_order_status_and_location_update(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(
        text="Water leaking under kitchen sink.",
        channel="Telegram",
    )
    decision = triage_service.classify(req)
    wo = temp_store.create_work_order(
        request=req,
        classification=decision,
        resident_id="tg_67890",
    )

    # Update location
    updated = temp_store.update_location(wo.work_order_id, "#12-04")
    assert updated is not None
    assert updated.location == "#12-04"

    # Update status
    dispatched = temp_store.update_status(
        wo.work_order_id,
        WorkOrderStatus.IN_PROGRESS.value,
        note="Technician assigned to inspect #12-04",
    )
    assert dispatched is not None
    assert dispatched.status == WorkOrderStatus.IN_PROGRESS.value
    assert any("Technician assigned" in n for n in dispatched.notes)


def test_telegram_bot_report_and_status(temp_store: WorkOrderStore, triage_service: JevTriageService):
    bot = CondoManagementTelegramBot(
        triage_service=triage_service,
        work_order_store=temp_store,
    )

    # 1. Process resident report
    wo, card_text = bot.process_resident_report(
        text="A car is blocking my assigned parking lot B1-22.",
        resident_id="tg_resident_1",
        resident_username="parking_victim",
        resident_name="Alice Wang",
    )

    assert wo.category == Category.PARKING
    assert wo.assigned_team == "Parking desk"
    assert "WO-EST-" in card_text
    assert "Parking" in card_text

    # 2. Check /status
    status_resp = bot.handle_status_query(wo.work_order_id)
    assert wo.work_order_id in status_resp
    assert "Parking" in status_resp
    assert "Parking desk" in status_resp

    # 3. Check /myrequests
    my_resp = bot.handle_my_requests("tg_resident_1")
    assert wo.work_order_id in my_resp
    assert "Parking" in my_resp

    # 4. Unknown status query
    not_found = bot.handle_status_query("WO-UNKNOWN-999")
    assert "was not found" in not_found


def test_telegram_bot_emergency_handling(temp_store: WorkOrderStore, triage_service: JevTriageService):
    bot = CondoManagementTelegramBot(
        triage_service=triage_service,
        work_order_store=temp_store,
    )

    wo, card_text = bot.process_resident_report(
        text="Passenger trapped in lift at Tower 1 lobby!",
        resident_id="tg_resident_emerg",
    )

    assert wo.priority == Priority.EMERGENCY
    assert wo.status == WorkOrderStatus.EMERGENCY_ESCALATION.value
    assert "EMERGENCY" in card_text
    assert "995" in card_text or "Guardhouse" in card_text

    # Check staff dispatch alert
    staff_alert = format_staff_dispatch_alert(wo)
    assert "URGENT EMERGENCY WORK ORDER" in staff_alert
    assert wo.work_order_id in staff_alert


def test_probable_location_detection():
    assert _is_probable_location("#05-12") is True
    assert _is_probable_location("unit 14-08") is True
    assert _is_probable_location("Block B gym") is True
    assert _is_probable_location("Tower 2") is True
    assert _is_probable_location("Hello, is anyone there?") is False


def test_telegram_session_store_persistence(tmp_path: Path):
    sess_file = tmp_path / "test_sessions.json"
    store1 = TelegramSessionStore(file_path=sess_file)

    # Set and get pending location
    store1.set_pending_location("user_101", "WO-EST-1001")
    assert store1.get_last_work_order("user_101") == "WO-EST-1001"

    # Verify reload in fresh store instance
    store2 = TelegramSessionStore(file_path=sess_file)
    assert store2.get_last_work_order("user_101") == "WO-EST-1001"
    popped = store2.pop_pending_location("user_101")
    assert popped == "WO-EST-1001"
    assert store2.pop_pending_location("user_101") is None

    # Test photo pending
    store2.set_pending_photo("user_101", "/data/attachments/img1.jpg")
    store3 = TelegramSessionStore(file_path=sess_file)
    assert store3.pop_pending_photo("user_101") == "/data/attachments/img1.jpg"
    assert store3.pop_pending_photo("user_101") is None


def test_conversational_reply_detection():
    # Casual phrases that shouldn't create new tickets
    assert _is_conversational_reply("ok") is True
    assert _is_conversational_reply("Okay!") is True
    assert _is_conversational_reply("thanks") is True
    assert _is_conversational_reply("Thank you very much") is True
    assert _is_conversational_reply("noted") is True
    assert _is_conversational_reply("cool") is True
    assert _is_conversational_reply("k") is True
    assert _is_conversational_reply("hi") is True

    # Real issue reports
    assert _is_conversational_reply("Water leaking under the sink") is False
    assert _is_conversational_reply("Gym treadmill belt is broken") is False
    assert _is_conversational_reply("Corridor lights flickering outside #08-12") is False


def test_telegram_bot_feedback_and_reopen(temp_store: WorkOrderStore, triage_service: JevTriageService):
    bot = CondoManagementTelegramBot(
        triage_service=triage_service,
        work_order_store=temp_store,
    )
    wo, _ = bot.process_resident_report(
        text="Intercom unit is buzzing loudly.",
        resident_id="tg_intercom_user",
    )
    assert wo.work_order_id.startswith("WO-EST-")

    # Resolve and close
    temp_store.resolve_work_order(wo.work_order_id, resolution_summary="Rewired buzzer terminal")
    temp_store.close_work_order(wo.work_order_id)

    # Resident submits 4-star feedback
    feedback_wo = temp_store.submit_feedback(wo.work_order_id, rating=4, comment="Good service, on time")
    assert feedback_wo is not None
    assert feedback_wo.resident_feedback_rating == 4

    # Resident reopens
    reopened = temp_store.reopen_work_order(wo.work_order_id, reason="Buzzing sound started again today")
    assert reopened is not None
    assert reopened.status == WorkOrderStatus.PENDING_REVIEW.value
    assert reopened.reopened_count == 1


def test_clean_bot_mention():
    assert _clean_bot_mention("@CondoBot the light is broken", "CondoBot") == "the light is broken"
    assert _clean_bot_mention("@condobot #08-14", "CondoBot") == "#08-14"
    assert _clean_bot_mention("@CondoBot", "CondoBot") == ""
    assert _clean_bot_mention("Water leaking in kitchen", "CondoBot") == "Water leaking in kitchen"


def test_estate_level_validation():
    from condo_jev.estate_directory import validate_estate_location

    # Valid locations
    assert validate_estate_location("#08-14").is_valid is True
    assert validate_estate_location("Block A Level 24").is_valid is True
    assert validate_estate_location("Block C Level 20").is_valid is True
    assert validate_estate_location("Block B #12-04").is_valid is True
    assert validate_estate_location("Gym").is_valid is True
    assert validate_estate_location("Lot B2-15").is_valid is True
    assert validate_estate_location("B1-05").is_valid is True

    # Invalid locations
    res_high = validate_estate_location("#28-14")
    assert res_high.is_valid is False
    assert "not a valid level" in res_high.error_message
    assert "resend again" in res_high.error_message

    res_blk_c = validate_estate_location("Block C Level 22")
    assert res_blk_c.is_valid is False
    assert "Block C only has levels 1 to 20" in res_blk_c.error_message
    assert "resend again" in res_blk_c.error_message

    res_blk_a = validate_estate_location("Block A Level 25")
    assert res_blk_a.is_valid is False
    assert "Block A only has levels 1 to 24" in res_blk_a.error_message

    res_zero = validate_estate_location("Level 0")
    assert res_zero.is_valid is False
    assert "not a valid level" in res_zero.error_message

    res_b3 = validate_estate_location("Basement 3")
    assert res_b3.is_valid is False
    assert "not a valid level" in res_b3.error_message

    res_blk_d = validate_estate_location("Block D Level 5")
    assert res_blk_d.is_valid is False
    assert "not a valid block" in res_blk_d.error_message


def test_resident_card_buttons_removed(temp_store: WorkOrderStore, triage_service: JevTriageService):
    from condo_jev.work_order import build_resident_card_markup

    req = ResidentRequest(
        text="Corridor light flickering outside #08-14.",
        channel="Telegram",
        location="#08-14",
    )
    decision = triage_service.classify(req)
    wo = temp_store.create_work_order(request=req, classification=decision, resident_id="res_btn_test")

    markup = build_resident_card_markup(wo)
    # Contains official receipt download and status tracking buttons
    buttons = markup.get("inline_keyboard", [])[0]
    callbacks = [b["callback_data"] for b in buttons]
    assert f"receipt:{wo.work_order_id}" in callbacks
    assert f"status:{wo.work_order_id}" in callbacks


def test_ai_response_clarifies_levels_when_missing_location(triage_service: JevTriageService):
    from condo_jev.ai_response import generate_ai_resident_reply

    req = ResidentRequest(
        text="The corridor light is flickering.",
        channel="Telegram",
    )
    decision = triage_service.classify(req)
    reply = generate_ai_resident_reply(req, decision)
    assert "Estate Blocks & Levels:" in reply
    assert "Block A: Levels 1 to 24" in reply
    assert "Block B: Levels 1 to 24" in reply
    assert "Block C: Levels 1 to 20" in reply


def test_media_screening_face_rejection(tmp_path: Path):
    from condo_jev.media_screening import screen_media_file
    from sklearn.datasets import fetch_olivetti_faces
    import cv2
    import numpy as np

    # Fetch a human face and save as image
    data = fetch_olivetti_faces()
    face_img = cv2.resize((data.images[0] * 255).astype(np.uint8), (200, 200))
    face_path = tmp_path / "resident_face.jpg"
    cv2.imwrite(str(face_path), face_img)

    result = screen_media_file(face_path)
    assert result.is_acceptable is False
    assert result.has_face is True
    assert "person's face" in result.rejection_reason


def test_media_screening_maintenance_acceptance(tmp_path: Path):
    from condo_jev.media_screening import screen_media_file
    import cv2
    import numpy as np

    # Synthetic image of ceiling water leak spot (no human face)
    leak_img = np.ones((300, 300, 3), dtype=np.uint8) * 210
    cv2.circle(leak_img, (150, 150), 60, (120, 100, 80), -1)
    cv2.circle(leak_img, (150, 150), 30, (80, 60, 40), -1)
    leak_path = tmp_path / "water_leak.jpg"
    cv2.imwrite(str(leak_path), leak_img)

    result = screen_media_file(leak_path)
    assert result.is_acceptable is True
    assert result.has_face is False
    assert result.rejection_reason is None


def test_maintenance_only_classification_check(triage_service: JevTriageService):
    # Maintenance issue (accepted)
    req_maint = ResidentRequest(text="Water is dripping from the corridor pipe at #08-14.", channel="Telegram")
    dec_maint = triage_service.classify(req_maint)
    assert dec_maint.category == Category.MAINTENANCE

    # Non-maintenance issue: noise (rejected)
    req_noise = ResidentRequest(text="Loud music party from neighbours at #11-04.", channel="Telegram")
    dec_noise = triage_service.classify(req_noise)
    assert dec_noise.category != Category.MAINTENANCE

    # Non-maintenance issue: parking (rejected)
    req_park = ResidentRequest(text="Unauthorized car parked in lot B1-15.", channel="Telegram")
    dec_park = triage_service.classify(req_park)
    assert dec_park.category != Category.MAINTENANCE


def test_telegram_bot_delete_work_order(temp_store: WorkOrderStore, triage_service: JevTriageService):
    bot = CondoManagementTelegramBot(
        triage_service=triage_service,
        work_order_store=temp_store,
    )
    wo, _ = bot.process_resident_report(
        text="Corridor lamp broken outside #05-02.",
        resident_id="tg_resident_del",
    )
    assert temp_store.get(wo.work_order_id) is not None

    # Delete via store
    assert bot.store.delete_work_order(wo.work_order_id) is True
    assert bot.store.get(wo.work_order_id) is None
    assert bot.store.delete_work_order(wo.work_order_id) is False


def test_noise_complaint_detection_and_extraction():
    from condo_jev.estate_directory import (
        is_noise_complaint,
        extract_noise_details,
        format_noise_clarification_prompt,
    )

    # 1. Detection
    assert is_noise_complaint("someone is playing loud music at night") is True
    assert is_noise_complaint("loud drilling noise at block a") is True
    assert is_noise_complaint("dog barking non stop") is True
    assert is_noise_complaint("heavy banging sound at night") is True
    assert is_noise_complaint("water pipe leaking under sink") is False
    assert is_noise_complaint("aircon blowing warm air") is False

    # 2. Timing extraction - missing timing
    vague_noise = extract_noise_details("someone is playing loud music at night")
    assert vague_noise.is_noise is True
    assert vague_noise.has_timing_details is False
    assert vague_noise.start_time is None
    assert vague_noise.duration is None

    # 3. Timing extraction - full details
    full_noise = extract_noise_details(
        "There's loud music playing at Block A Level 12 #12-04 that started at 11pm and has been going on for 2 hours"
    )
    assert full_noise.is_noise is True
    assert full_noise.has_timing_details is True
    assert full_noise.start_time == "11 PM"
    assert "2 hours" in full_noise.duration

    # 4. Partial timing
    partial_st = extract_noise_details("loud drilling started around 10:30pm")
    assert partial_st.start_time == "10:30 PM"
    assert partial_st.has_timing_details is False

    partial_dur = extract_noise_details("dog barking for 45 minutes")
    assert "45 minutes" in str(partial_dur.duration)

    # 5. Clarification prompt
    prompt = format_noise_clarification_prompt(location_str="Block A Level 12")
    assert "What time did the sound start?" in prompt
    assert "How long has it been going on?" in prompt
    assert "Block A Level 12" in prompt
    # Zero emojis check
    assert not any(ord(c) > 127900 for c in prompt)


def test_telegram_session_store_noise_persistence(tmp_path: Path):
    from condo_jev.telegram_bot import TelegramSessionStore

    sess_file = tmp_path / "test_noise_sessions.json"
    store = TelegramSessionStore(file_path=sess_file)

    # Save pending noise
    store.set_pending_noise("user_noise_1", {
        "original_text": "Loud music playing at night",
        "location_hint": "Block B",
    })

    # Retrieve
    saved = store.get_pending_noise("user_noise_1")
    assert saved is not None
    assert saved["original_text"] == "Loud music playing at night"
    assert saved["location_hint"] == "Block B"

    # Pop
    popped = store.pop_pending_noise("user_noise_1")
    assert popped is not None
    assert store.get_pending_noise("user_noise_1") is None


def test_topic_manager_routing(tmp_path: Path):
    from condo_jev.topic_manager import TopicManager, normalize_topic_key

    # Test normalization
    assert normalize_topic_key("gym") == "facilities"
    assert normalize_topic_key("pool") == "facilities"
    assert normalize_topic_key("repairs") == "maintenance"
    assert normalize_topic_key("loud music") == "noise"
    assert normalize_topic_key("trash") == "cleanliness"
    assert normalize_topic_key("gate") == "parking"

    # Test file-based topic manager
    topics_file = tmp_path / "test_topics.json"
    tm = TopicManager(file_path=topics_file)

    # Pre-configured facilities topic
    assert tm.get_thread_id_for_category(Category.FACILITIES) == 10
    assert tm.get_thread_id_for_category("facilities") == 10

    # Save custom topic thread
    tm.save_topic("maintenance", 4)
    tm.save_topic("noise", 6)

    assert tm.get_thread_id_for_category(Category.MAINTENANCE) == 4
    assert tm.get_thread_id_for_category(Category.NOISE) == 6
    assert tm.get_thread_id_for_category(Category.CLEANLINESS) is None


def test_send_telegram_direct_message_compound_chat_id(monkeypatch):
    import json
    from condo_jev.work_order import send_telegram_direct_message

    captured_payload = {}

    def mock_urlopen(req, timeout=5.0):
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        class MockResp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({"ok": True, "result": {"message_id": 999}}).encode("utf-8")
        return MockResp()

    monkeypatch.setattr("condo_jev.work_order.urllib.request.urlopen", mock_urlopen)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_token_123")
    monkeypatch.setenv("CONDO_FORCE_LIVE_TELEGRAM", "1")

    # Send with compound chat_id_thread_id format (e.g. copied from web URL)
    res = send_telegram_direct_message(
        chat_id="-1003561028399_10",
        text="Test alert to gym and pool topic",
    )
    assert res["success"] is True
    assert captured_payload["chat_id"] == "-1003561028399"
    assert captured_payload["message_thread_id"] == 10


def test_is_concrete_estate_location():
    from condo_jev.estate_directory import is_concrete_estate_location

    # No location specified
    assert not is_concrete_estate_location(None, "there's a water leak")
    assert not is_concrete_estate_location("", "pipe is dripping in bathroom")
    assert not is_concrete_estate_location("Pending location confirmation", "water leak")
    assert not is_concrete_estate_location("Pending confirmation", "noise")

    # Concrete unit / block / level specified
    assert is_concrete_estate_location("#08-14", "water leak")
    assert is_concrete_estate_location(None, "corridor light flickering outside #08-14")
    assert is_concrete_estate_location(None, "water leak at Block A #09-12")
    assert is_concrete_estate_location("Block B Level 4", "flickering bulb")

    # Common facilities specified
    assert is_concrete_estate_location(None, "gym aircon leaking water")
    assert is_concrete_estate_location(None, "swimming pool dirty")
    assert is_concrete_estate_location(None, "bbq pit 2 light not working")
    assert is_concrete_estate_location(None, "carpark basement 1 flooded")
    assert is_concrete_estate_location("Clubhouse Level 2", "broken door handle")


def test_process_resident_report_location_gating(temp_store: WorkOrderStore, triage_service: JevTriageService):
    from condo_jev.telegram_bot import CondoManagementTelegramBot

    bot = CondoManagementTelegramBot(
        triage_service=triage_service,
        work_order_store=temp_store,
    )

    # 1. Report without location -> pending location set
    wo, card = bot.process_resident_report(
        text="there's a water leak",
        resident_id="user_loc_1",
        resident_username="testuser",
    )
    assert wo.work_order_id
    assert bot.session_store.get_session("user_loc_1").get("pending_location_wo_id") == wo.work_order_id

    # 2. Update location with resident reply
    updated = bot.store.update_location(wo.work_order_id, "Block A, #09-12")
    assert updated is not None
    assert updated.location == "Block A, #09-12"
    assert updated.block == "Block A"
    assert updated.unit == "#09-12"

    # 3. Report with concrete location -> not pending location
    wo2, card2 = bot.process_resident_report(
        text="gym treadmill belt snapped",
        resident_id="user_loc_2",
        resident_username="testuser2",
    )
    assert bot.session_store.get_session("user_loc_2").get("pending_location_wo_id") is None


def test_topic_manager_enquiries_routing(tmp_path: Path):
    from condo_jev.topic_manager import TopicManager, normalize_topic_key

    assert normalize_topic_key("condo enquiries") == "enquiries"
    assert normalize_topic_key("enquiry") == "enquiries"
    assert normalize_topic_key("enquiries") == "enquiries"
    assert normalize_topic_key("faq") == "enquiries"

    topics_file = tmp_path / "test_enquiries_topics.json"
    tm = TopicManager(file_path=topics_file)
    tm.save_topic("enquiries", 27)

    assert tm.get_thread_id_for_category("enquiries") == 27
    assert tm.get_thread_id_for_category(Category.GENERAL) == 27
    assert tm.get_thread_id_for_category(Category.BILLING) == 27


def test_get_resident_reply_keyboard(tmp_path: Path):
    from telegram import ReplyKeyboardRemove
    from condo_jev.telegram_bot import CondoManagementTelegramBot
    from condo_jev.work_order import WorkOrderStore
    bot = CondoManagementTelegramBot(work_order_store=WorkOrderStore(file_path=tmp_path / "orders.json"))
    kb = bot.get_resident_reply_keyboard()
    assert kb is not None
    assert isinstance(kb, ReplyKeyboardRemove)


def test_get_resident_welcome_markup(tmp_path: Path):
    from telegram import ReplyKeyboardRemove
    from condo_jev.telegram_bot import CondoManagementTelegramBot
    from condo_jev.work_order import WorkOrderStore
    bot = CondoManagementTelegramBot(work_order_store=WorkOrderStore(file_path=tmp_path / "orders.json"))
    markup = bot.get_resident_welcome_markup()
    assert markup is not None
    assert isinstance(markup, ReplyKeyboardRemove)


def test_resident_service_review_and_status_display(temp_store: WorkOrderStore, triage_service: JevTriageService):
    bot = CondoManagementTelegramBot(
        triage_service=triage_service,
        work_order_store=temp_store,
    )
    req = ResidentRequest(
        text="Corridor light flickering outside #08-14.",
        channel="Telegram",
        location="#08-14",
    )
    dec = triage_service.classify(req)
    wo = temp_store.create_work_order(
        request=req,
        classification=dec,
        resident_id="tg_review_user",
    )
    assert wo.resident_feedback_rating is None

    # Record feedback via record_feedback
    updated = temp_store.record_feedback(wo.work_order_id, rating=5)
    assert updated is not None
    assert updated.resident_feedback_rating == 5
    assert any("5/5" in n for n in updated.notes)

    # Verify status query reflects the rating
    status_text = bot.handle_status_query(wo.work_order_id)
    assert "*Resident Rating:* 5 / 5 Stars" in status_text

    # Verify numeric id also works
    ticket_num = wo.ticket_id.replace("EST-", "")
    updated_num = temp_store.record_feedback(ticket_num, rating=4)
    assert updated_num is not None
    assert updated_num.resident_feedback_rating == 4


def test_is_ambiguous_carpark_location():
    from condo_jev.estate_directory import is_ambiguous_carpark_location, is_concrete_estate_location

    # Ambiguous carpark mentions (requires B1/B2 clarification)
    assert is_ambiguous_carpark_location(None, "there is a water leak in the carpark") is True
    assert is_ambiguous_carpark_location("carpark", "oil spill on the floor") is True
    assert is_ambiguous_carpark_location(None, "car park light broken") is True
    assert is_concrete_estate_location(None, "there is a water leak in the carpark") is False

    # Concrete carpark mentions with level, lot, or pillar
    assert is_ambiguous_carpark_location(None, "water leak at carpark B1") is False
    assert is_ambiguous_carpark_location("B2", "oil spill on the floor") is False
    assert is_ambiguous_carpark_location(None, "carpark basement 1 flooded") is False
    assert is_ambiguous_carpark_location(None, "car parked at Lot 42") is False
    assert is_ambiguous_carpark_location(None, "carpark near Pillar BB3") is False
    assert is_concrete_estate_location(None, "water leak at carpark B1") is True


def test_estate_carpark_map_asset_exists():
    from pathlib import Path
    map_path = Path(__file__).resolve().parent.parent / "output" / "estate_carpark_map.jpg"
    assert map_path.exists(), f"Expected carpark map at {map_path}"
    assert map_path.stat().st_size > 5000, "Carpark map file should be a valid non-empty image"


def test_carpark_ambiguity_two_step_clarification(temp_store: WorkOrderStore, triage_service: JevTriageService):
    from condo_jev.telegram_bot import CondoManagementTelegramBot
    from condo_jev.estate_directory import is_ambiguous_carpark_location

    bot = CondoManagementTelegramBot(
        triage_service=triage_service,
        work_order_store=temp_store,
    )

    # Step 1: Resident says "there's a water leak" (generic, no location)
    wo, card = bot.process_resident_report(
        text="there's a water leak",
        resident_id="user_cp_step",
        resident_username="tester",
    )
    assert bot.session_store.get_session("user_cp_step").get("pending_location_wo_id") == wo.work_order_id

    # Step 2: Resident replies "carpark" -> Still ambiguous!
    reply_step2 = "carpark"
    assert is_ambiguous_carpark_location(reply_step2) is True

    # Step 3: Resident specifies "Basement 1 near Lot 42" -> Concrete!
    reply_step3 = "Basement 1 near Lot 42"
    assert is_ambiguous_carpark_location(reply_step3) is False
    updated = bot.store.update_location(wo.work_order_id, reply_step3)
    assert updated is not None
    assert updated.location == "Basement 1 near Lot 42"


def test_emergency_triage_classification(triage_service: JevTriageService):
    emergency_texts = [
        "Fire in the trash chute at Tower B level 3",
        "Someone is stuck inside lift 2 at Block A, please help immediately!",
        "Someone collapsed near the swimming pool and is unconscious",
        "Severe gas leak smell detected outside #05-02",
        "Burglary in progress, intruder broke into unit #10-04",
    ]
    for text in emergency_texts:
        req = ResidentRequest(text=text, channel="Telegram")
        decision = triage_service.classify(req)
        assert decision.priority == Priority.EMERGENCY, f"Expected emergency priority for: {text}"
        assert decision.nouls.get("life_safety_hazard", 0.0) >= 0.50


def test_emergency_resident_card_includes_hotlines(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(
        text="Fire broke out at Block B corridor ceiling!",
        channel="Telegram",
        location="Block B Level 4",
    )
    decision = triage_service.classify(req)
    wo = temp_store.create_work_order(
        request=req,
        classification=decision,
        resident_id="user_emerg",
    )
    assert wo.priority == Priority.EMERGENCY
    card = format_telegram_resident_card(wo)
    assert "URGENT EMERGENCY REPORT" in card
    assert "995" in card  # SCDF
    assert "999" in card  # Police


def test_audio_evidence_attachment(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(
        text="Strange loud drilling sounds from unit above #07-12",
        channel="Telegram",
        location="#07-12",
    )
    decision = triage_service.classify(req)
    wo = temp_store.create_work_order(
        request=req,
        classification=decision,
        resident_id="user_audio",
    )

    audio_file_path = "data/attachments/audio_user_audio_20261019_130000.ogg"
    updated = temp_store.attach_photo(wo.work_order_id, audio_file_path, filename="audio_user_audio_20261019_130000.ogg")
    assert updated is not None
    assert updated.photo_path == audio_file_path
    assert updated.attachment_filename == "audio_user_audio_20261019_130000.ogg"


def test_language_detection_and_translation():
    from condo_jev.translator import detect_language, translate_text

    # Language detection
    assert detect_language("泳池水质很脏")[0] == "zh"
    assert detect_language("Lif rosak di Blok B")[0] == "ms"
    assert detect_language("Water leaking outside #08-14")[0] == "en"
    assert detect_language("தண்ணீர் கசிகிறது")[0] == "ta"

    # Translation to English
    res_zh = translate_text("B2水管爆裂", target_lang="en")
    assert res_zh.is_translated is True
    assert "water" in res_zh.translated_text.lower() or "pipe" in res_zh.translated_text.lower() or "burst" in res_zh.translated_text.lower()

    # English text unchanged
    res_en = translate_text("Water leaking outside #08-14", target_lang="en")
    assert res_en.is_translated is False
    assert res_en.translated_text == "Water leaking outside #08-14"


def test_telegram_session_store_language_preference(tmp_path: Path):
    from condo_jev.telegram_bot import TelegramSessionStore

    sess_file = tmp_path / "sessions.json"
    store = TelegramSessionStore(file_path=sess_file)

    # Default is en
    assert store.get_language("user_123") == "en"

    # Set preferred language
    store.set_language("user_123", "zh")
    assert store.get_language("user_123") == "zh"

    # Set another user
    store.set_language("user_456", "ms")
    assert store.get_language("user_456") == "ms"
    assert store.get_language("user_123") == "zh"


def test_format_resident_receipt(temp_store: WorkOrderStore, triage_service: JevTriageService):
    from condo_jev.work_order import format_resident_receipt, build_resident_card_markup

    req = ResidentRequest(
        text="Major ceiling leak in Master Bedroom from upper unit",
        channel="Telegram",
        location="#14-02",
    )
    decision = triage_service.classify(req)
    wo = temp_store.create_work_order(
        request=req,
        classification=decision,
        resident_id="tg_9999",
        resident_username="alice_tan",
        resident_name="Alice Tan",
    )

    receipt = format_resident_receipt(wo)
    assert "OFFICIAL INCIDENT & REPORT RECEIPT" in receipt
    assert "Emerald Bay Residences (MCST 4521)" in receipt
    assert wo.work_order_id in receipt
    assert "Alice Tan" in receipt
    assert "#14-02" in receipt
    assert "Major ceiling leak" in receipt
    assert "Live Status Tracker" in receipt

    # Verify resident card markup includes receipt download button
    markup = build_resident_card_markup(wo)
    buttons = markup["inline_keyboard"][0]
    button_callbacks = [b["callback_data"] for b in buttons]
    assert f"receipt:{wo.work_order_id}" in button_callbacks
    assert f"status:{wo.work_order_id}" in button_callbacks


def test_audio_pending_and_text_self_transcription(tmp_path: Path):
    from condo_jev.telegram_bot import TelegramSessionStore

    sess_file = tmp_path / "sessions.json"
    session_store = TelegramSessionStore(file_path=sess_file)

    res_id = "user_audio_test"
    audio_path = "data/attachments/audio_user_audio_test_20261019_130000.ogg"

    # User uploads audio -> bot stores as pending photo/attachment without auto-transcribing
    session_store.set_pending_photo(res_id, audio_path)

    # User transcribes the text themselves in their next message
    popped = session_store.pop_pending_photo(res_id)
    assert popped == audio_path
    assert session_store.pop_pending_photo(res_id) is None


def test_child_and_infant_emergency_life_safety(temp_store: WorkOrderStore, triage_service: JevTriageService):
    scenarios = [
        "my baby's head got stuck in the toilet",
        "i dropped my baby on the balcony by accident",
    ]
    for text in scenarios:
        req = ResidentRequest(text=text, channel="Telegram")
        dec = triage_service.classify(req)
        assert dec.priority == Priority.EMERGENCY
        assert dec.nouls.get("life_safety_hazard", 0.0) >= 0.50

        wo = temp_store.create_work_order(req, dec, resident_id="user_emergency_test")
        card = format_telegram_resident_card(wo)
        assert "URGENT EMERGENCY REPORT" in card
        assert "CRITICAL LIFE-SAFETY EMERGENCY" in card
        assert "995" in card  # SCDF ambulance
        assert "999" in card  # Police
        assert "+65 6789 0123" in card  # 24/7 Guardhouse
        assert text in card


def test_semantic_intent_domain_recognition(triage_service: JevTriageService):
    cases = [
        ("Someone scratched my car in the visitor carpark", Category.PARKING, Priority.NORMAL),
        ("The gym treadmill belt snapped while I was jogging", Category.FACILITIES, Priority.NORMAL),
        ("Security guard was rude and refused entry to my mother", Category.SECURITY, Priority.NORMAL),
        ("Neighbor throwing cigarette butts down into my balcony", Category.NOISE, Priority.NORMAL),
        ("Uncut grass has created a mosquito nest", Category.CLEANLINESS, Priority.NORMAL),
        ("Need to book function room for my daughter's birthday next month", Category.FACILITIES, Priority.LOW),
        ("Why was I charged twice for the sinking fund levy on my invoice", Category.BILLING, Priority.NORMAL),
        ("Heavy hacking drilling on Sunday morning from unit above", Category.RENOVATION, Priority.NORMAL),
        ("The water heater is sparking and emitting smoke in #12-04", Category.MAINTENANCE, Priority.EMERGENCY),
        ("Corridor light flickering outside #08-14", Category.MAINTENANCE, Priority.NORMAL),
    ]
    for text, exp_cat, exp_prio in cases:
        dec = triage_service.classify(ResidentRequest(text=text, channel="Telegram"))
        assert dec.category == exp_cat, f"Expected {exp_cat} for '{text}', got {dec.category}"
        assert dec.priority == exp_prio, f"Expected {exp_prio} for '{text}', got {dec.priority}"
