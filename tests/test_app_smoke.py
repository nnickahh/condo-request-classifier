import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_streamlit_app_renders_and_classifies_ticket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_json = tmp_path / "smoke_work_orders.json"
    prod_json = Path(__file__).resolve().parent.parent / "data" / "work_orders.json"
    if prod_json.exists():
        test_json.write_text(prod_json.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        test_json.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CONDO_WORK_ORDERS_FILE", str(test_json))

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception
    assert "#14-08" in at.text_area(key="request_text").value

    # Click the "Evaluate with Jev System-One" form submit button
    submit_buttons = [b for b in at.button if "Evaluate with Jev" in b.label]
    assert submit_buttons
    submit_buttons[0].click().run()
    assert not at.exception
    assert "single_result" in at.session_state

    # Click a Quick Scenario pill button and verify auto-evaluation
    lift_buttons = [b for b in at.button if "Lift Emergency" in b.label]
    assert lift_buttons
    lift_buttons[0].click().run()
    assert not at.exception
    assert at.session_state["single_result"].priority.value == "emergency"

    # Click the One-Click Sample Inbox button in Batch Review tab
    batch_buttons = [b for b in at.button if "Load & classify sample inbox" in b.label]
    assert batch_buttons
    batch_buttons[0].click().run()
    assert not at.exception
    assert len(at.session_state["batch_result"][1]) == 4

    # Verify Live Ticket Queue is populated and Admin Approve & Dispatch works
    assert "live_ticket_queue" in at.session_state
    assert len(at.session_state["live_ticket_queue"]) >= 2
    approve_buttons = [b for b in at.button if "Approve & Dispatch" in b.label]
    assert approve_buttons
    approve_buttons[0].click().run()
    assert not at.exception
    assert any("Approved" in item["admin_status"] for item in at.session_state["live_ticket_queue"])

    # Switch to Resident Portal View and verify clean render
    at.radio(key="triage_role_perspective").set_value("🏠 Resident Portal View").run()
    assert not at.exception


