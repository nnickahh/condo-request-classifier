import json
import urllib.request
from pathlib import Path

import pytest

from condo_jev.backends import LocalJevSimulatorBackend
from condo_jev.schema import Category, Priority, ResidentRequest
from condo_jev.work_order import (
    ESTATE_TECHNICIANS,
    WorkOrderStatus,
    WorkOrderStore,
    dispatch_outbound_webhook,
    format_staff_dispatch_alert,
    format_telegram_resident_card,
)
from condo_jev.work_order_api import CMMSApiServer
from condo_jev.workflow import JevTriageService


@pytest.fixture
def temp_store(tmp_path: Path) -> WorkOrderStore:
    test_json = tmp_path / "test_work_orders.json"
    return WorkOrderStore(file_path=test_json)


@pytest.fixture
def triage_service() -> JevTriageService:
    return JevTriageService(backend=LocalJevSimulatorBackend())


def test_work_order_full_cmms_lifecycle(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(
        text="Corridor light outside #08-14 is flickering and dim.",
        channel="Telegram",
        location="#08-14",
    )
    decision = triage_service.classify(req)
    wo = temp_store.create_work_order(
        request=req,
        classification=decision,
        resident_id="tg_123456",
        resident_username="test_user",
        resident_name="Test Resident",
    )

    assert wo.work_order_id.startswith("WO-EST-")
    assert wo.assigned_technician is None
    assert wo.status in (WorkOrderStatus.AUTO_DISPATCHED.value, WorkOrderStatus.PENDING_REVIEW.value)

    # 1. Assign technician
    assigned = temp_store.assign_technician(
        wo.work_order_id,
        technician_name="Kumar S.",
        phone="+65 9345 6789",
        vendor="BrightSpark Electrical",
        note="Dispatched to replace LED tube and inspect driver",
    )
    assert assigned is not None
    assert assigned.assigned_technician == "Kumar S."
    assert assigned.technician_phone == "+65 9345 6789"
    assert assigned.status == WorkOrderStatus.ASSIGNED.value
    assert any("Technician assigned" in n for n in assigned.notes)

    # 2. Advance to In Progress
    in_prog = temp_store.update_status(
        wo.work_order_id,
        WorkOrderStatus.IN_PROGRESS.value,
        note="Technician Kumar is on site at level 8.",
    )
    assert in_prog is not None
    assert in_prog.status == WorkOrderStatus.IN_PROGRESS.value

    # 3. Resolve work order
    resolved = temp_store.resolve_work_order(
        wo.work_order_id,
        resolution_summary="Replaced faulty LED starter and ballast. Light confirmed working.",
        labor_hours=1.2,
        cost_sgd=45.0,
        parts=["LED Starter 10W", "Ballast 240V"],
        note="Tested with light meter: 150 lux.",
    )
    assert resolved is not None
    assert resolved.status == WorkOrderStatus.RESOLVED.value
    assert resolved.labor_hours == 1.2
    assert resolved.cost_sgd == 45.0
    assert "LED Starter 10W" in resolved.parts_or_materials
    assert resolved.resolved_at is not None

    # 4. Close work order
    closed = temp_store.close_work_order(wo.work_order_id, note="Resident acknowledged sign-off.")
    assert closed is not None
    assert closed.status == WorkOrderStatus.CLOSED.value
    assert closed.closed_at is not None

    # Check stats
    stats = temp_store.get_cmms_stats()
    assert stats["total_work_orders"] == 1
    assert stats["resolved"] == 1
    assert stats["closed"] == 1

    # Check CSV export
    csv_text = temp_store.export_csv()
    assert "Work Order ID" in csv_text
    assert wo.work_order_id in csv_text
    assert "Kumar S." in csv_text


def test_telegram_formatting_with_technician(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(
        text="Water pooling near swimming pool pump room.",
        channel="Resident App",
        location="Poolside pump room",
    )
    dec = triage_service.classify(req)
    wo = temp_store.create_work_order(req, dec, resident_id="app_user_99")
    temp_store.assign_technician(wo.work_order_id, "Ah Seng", phone="+65 9234 5678")

    fetched = temp_store.get(wo.work_order_id)
    assert fetched is not None
    card = format_telegram_resident_card(fetched)
    assert "Ah Seng" in card

    alert = format_staff_dispatch_alert(fetched)
    assert "Ah Seng" in alert


def test_cmms_rest_api_server(temp_store: WorkOrderStore, triage_service: JevTriageService):
    # Seed 1 ticket
    req = ResidentRequest(text="Elevator button for 7th floor is stuck.", channel="Telegram", location="Lift 1")
    dec = triage_service.classify(req)
    wo = temp_store.create_work_order(req, dec, resident_id="user_elevator")

    # Start API server on ephemeral high port
    server = CMMSApiServer(host="127.0.0.1", port=18088, store=temp_store, triage_service=triage_service)
    server.start_background()

    try:
        base_url = "http://127.0.0.1:18088"

        # 1. GET /api/health
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=3) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "healthy"
            assert data["total_work_orders"] >= 1

        # 2. GET /api/stats
        with urllib.request.urlopen(f"{base_url}/api/stats", timeout=3) as resp:
            assert resp.status == 200
            stats = json.loads(resp.read().decode())
            assert "total_work_orders" in stats

        # 3. GET /api/technicians
        with urllib.request.urlopen(f"{base_url}/api/technicians", timeout=3) as resp:
            assert resp.status == 200
            techs = json.loads(resp.read().decode())
            assert len(techs["technicians"]) > 0

        # 4. GET /api/work-orders
        with urllib.request.urlopen(f"{base_url}/api/work-orders", timeout=3) as resp:
            assert resp.status == 200
            orders = json.loads(resp.read().decode())
            assert orders["count"] >= 1

        # 5. POST /api/work-orders
        post_data = json.dumps({
            "text": "The gym AC is leaking water on the mat.",
            "channel": "API Client",
            "location": "Gym Room",
            "resident_name": "API Reporter",
        }).encode("utf-8")
        req_post = urllib.request.Request(
            f"{base_url}/api/work-orders",
            data=post_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_post, timeout=5) as resp:
            assert resp.status == 201
            created = json.loads(resp.read().decode())
            new_wo_id = created["work_order"]["work_order_id"]
            assert new_wo_id.startswith("WO-EST-")

        # 6. PATCH /api/work-orders/{id}
        patch_data = json.dumps({
            "status": "In Progress",
            "assigned_technician": "John Tan",
            "technician_phone": "+65 9123 4567",
            "note": "John Tan dispatched via CMMS API",
        }).encode("utf-8")
        req_patch = urllib.request.Request(
            f"{base_url}/api/work-orders/{new_wo_id}",
            data=patch_data,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        with urllib.request.urlopen(req_patch, timeout=3) as resp:
            assert resp.status == 200
            patched = json.loads(resp.read().decode())
            assert patched["work_order"]["assigned_technician"] == "John Tan"
            assert patched["work_order"]["status"] == "In Progress"

    finally:
        server.shutdown()


def test_work_order_photo_attachment_and_feedback(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(
        text="Gym treadmill 3 belt is torn.",
        channel="Telegram",
        location="Gym Room",
    )
    dec = triage_service.classify(req)
    wo = temp_store.create_work_order(req, dec, resident_id="user_gym")

    # 1. Attach photo
    attached = temp_store.attach_photo(
        wo.work_order_id,
        photo_path="data/attachments/gym_belt.jpg",
        filename="gym_belt.jpg",
    )
    assert attached is not None
    assert attached.photo_path == "data/attachments/gym_belt.jpg"
    assert attached.attachment_filename == "gym_belt.jpg"
    assert any("Photo evidence attached" in n for n in attached.notes)

    # 2. Resolve and Close
    temp_store.resolve_work_order(wo.work_order_id, resolution_summary="Belt replaced")
    temp_store.close_work_order(wo.work_order_id)

    # 3. Submit feedback
    with_feedback = temp_store.submit_feedback(
        wo.work_order_id,
        rating=5,
        comment="Fast turnaround, thank you!",
    )
    assert with_feedback is not None
    assert with_feedback.resident_feedback_rating == 5
    assert with_feedback.resident_feedback_comment == "Fast turnaround, thank you!"
    assert any("(5/5)" in n for n in with_feedback.notes)

    # 4. Reopen work order
    reopened = temp_store.reopen_work_order(
        wo.work_order_id,
        reason="Treadmill belt makes squeaking noise when running.",
    )
    assert reopened is not None
    assert reopened.status == WorkOrderStatus.PENDING_REVIEW.value
    assert reopened.reopened_count == 1
    assert any("TICKET REOPENED BY RESIDENT" in n for n in reopened.notes)


def test_cmms_rest_api_telegram_and_reopen_endpoints(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req = ResidentRequest(text="Swimming pool shower head is leaking.", channel="Telegram", location="Poolside")
    dec = triage_service.classify(req)
    wo = temp_store.create_work_order(req, dec, resident_id="tg_pool_guest")

    server = CMMSApiServer(host="127.0.0.1", port=18089, store=temp_store, triage_service=triage_service)
    server.start_background()

    try:
        base_url = "http://127.0.0.1:18089"

        # 1. GET /api/telegram/status
        with urllib.request.urlopen(f"{base_url}/api/telegram/status", timeout=5) as resp:
            assert resp.status == 200
            t_status = json.loads(resp.read().decode())
            assert "telegram_configured" in t_status
            assert "staff_chat_configured" in t_status

        # 2. POST /api/work-orders/{id}/reopen
        reopen_data = json.dumps({"reason": "Shower handle fell off again"}).encode("utf-8")
        req_reopen = urllib.request.Request(
            f"{base_url}/api/work-orders/{wo.work_order_id}/reopen",
            data=reopen_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_reopen, timeout=5) as resp:
            assert resp.status == 200
            reopen_resp = json.loads(resp.read().decode())
            assert reopen_resp["work_order"]["reopened_count"] == 1
            assert reopen_resp["work_order"]["status"] == WorkOrderStatus.PENDING_REVIEW.value

        # 3. POST /api/work-orders/{id}/notify (with resident without valid chat id should fail gracefully)
        notify_data = json.dumps({"message": "Technician will arrive at 3 PM"}).encode("utf-8")
        req_notify = urllib.request.Request(
            f"{base_url}/api/work-orders/{wo.work_order_id}/notify",
            data=notify_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req_notify, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

    finally:
        server.shutdown()


def test_delete_work_order_and_bulk_delete(temp_store: WorkOrderStore, triage_service: JevTriageService):
    req1 = ResidentRequest(text="Pipe leak in #02-01", channel="Telegram", location="#02-01")
    req2 = ResidentRequest(text="Light flicker in #03-02", channel="Telegram", location="#03-02")
    req3 = ResidentRequest(text="Aircon noise in #04-03", channel="Telegram", location="#04-03")

    wo1 = temp_store.create_work_order(req1, triage_service.classify(req1), resident_id="111")
    wo2 = temp_store.create_work_order(req2, triage_service.classify(req2), resident_id="222")
    wo3 = temp_store.create_work_order(req3, triage_service.classify(req3), resident_id="333")

    assert len(temp_store.list_all()) == 3

    # Delete single work order
    assert temp_store.delete_work_order(wo1.work_order_id) is True
    assert temp_store.get(wo1.work_order_id) is None
    assert len(temp_store.list_all()) == 2

    # Attempt to delete non-existent ID
    assert temp_store.delete_work_order("WO-NON-EXISTENT") is False

    # Bulk delete remaining
    count = temp_store.bulk_delete_work_orders([wo2.work_order_id, wo3.work_order_id])
    assert count == 2
    assert len(temp_store.list_all()) == 0
