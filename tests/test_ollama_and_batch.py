import json
from types import SimpleNamespace

import pytest

from condo_jev.backends import OllamaJevBackend, parse_and_verify_llm_jev_payload
from condo_jev.batch import batch_results_to_csv, classify_csv_bytes
from condo_jev.evaluation import run_evaluation, verify_no_train_eval_leakage
from condo_jev.schema import Category, Priority, ResidentRequest
from condo_jev.workflow import JevTriageService


def test_ollama_payload_parser_accepts_outer_quotes_on_verbatim_evidence() -> None:
    req = ResidentRequest(text="The kitchen sink pipe in #09-04 is leaking water onto the floor.")
    raw = json.dumps(
        {
            "category": "maintenance",
            "priority": "normal",
            "choice_confidence": 0.86,
            "urgency_score": 0.42,
            "noul_life_safety_hazard": 0.02,
            "noul_multi_intent_request": 0.05,
            "noul_adversarial_or_sensitive": 0.01,
            "noul_requires_human_triage": 0.15,
            "evidence_quotes": ['"leaking water"', "'#09-04'"],
            "rationale_note": "Resident reports a leaking kitchen sink pipe in unit #09-04.",
        }
    )
    parsed = parse_and_verify_llm_jev_payload(f"```json\n{raw}\n```", req)
    assert parsed.category == Category.MAINTENANCE
    assert parsed.evidence_quotes == ["leaking water", "#09-04"]


def test_ollama_payload_parser_rejects_fabricated_evidence_quote() -> None:
    req = ResidentRequest(text="The kitchen sink pipe in #09-04 is leaking water onto the floor.")
    raw = json.dumps(
        {
            "category": "maintenance",
            "priority": "normal",
            "choice_confidence": 0.86,
            "urgency_score": 0.42,
            "noul_life_safety_hazard": 0.02,
            "noul_multi_intent_request": 0.05,
            "noul_adversarial_or_sensitive": 0.01,
            "noul_requires_human_triage": 0.15,
            "evidence_quotes": ["hallucinated phrase not in text"],
            "rationale_note": "Fabricated evidence quote test.",
        }
    )
    with pytest.raises(ValueError, match="verbatim span"):
        parse_and_verify_llm_jev_payload(raw, req)


def test_ollama_backend_retries_and_always_flags_uncalibrated_for_review() -> None:
    calls = []

    class FakeChatClient:
        def invoke(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return SimpleNamespace(content="not valid json")
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "category": "parking",
                        "priority": "high",
                        "choice_confidence": 0.91,
                        "urgency_score": 0.72,
                        "noul_life_safety_hazard": 0.01,
                        "noul_multi_intent_request": 0.04,
                        "noul_adversarial_or_sensitive": 0.01,
                        "noul_requires_human_triage": 0.12,
                        "evidence_quotes": ["parked in my designated lot"],
                        "rationale_note": "Unauthorized vehicle occupying resident lot.",
                    }
                )
            )

    backend = OllamaJevBackend(model="gemma3:4b", max_retries=2, chat_client=FakeChatClient())
    service = JevTriageService(backend=backend)
    req = ResidentRequest(text="An unknown car is parked in my designated lot B1-19 right now.")
    decision = service.classify(req)

    assert len(calls) == 2
    assert decision.category == Category.PARKING
    assert decision.priority == Priority.HIGH
    assert decision.calibration_status == "uncalibrated_llm_self_report"
    assert decision.needs_human_review is True


def test_batch_csv_processing_and_serialization(jev_service: JevTriageService) -> None:
    csv_content = (
        "channel,location,text\n"
        'Resident App,#07-11,"The bathroom tap in #07-11 is leaking water."\n'
        'Web,,"???"\n'
    ).encode("utf-8")

    records = classify_csv_bytes(csv_content, jev_service)
    assert len(records) == 2
    assert records[0]["status"] == "ok"
    assert records[0]["category"] == "maintenance"
    assert records[1]["status"] == "validation_error"
    assert records[1]["needs_human_review"] is True

    csv_out = batch_results_to_csv(records)
    assert "noul_life_safety_hazard" in csv_out
    assert "maintenance" in csv_out


def test_dataset_leakage_and_evaluation_benchmark(jev_service: JevTriageService) -> None:
    verify_no_train_eval_leakage()
    report = run_evaluation(service=jev_service)
    assert report["dataset"]["train_rows"] == 160
    assert report["dataset"]["eval_rows"] == 60
    assert report["metrics"]["category_accuracy"] >= 0.95
    assert report["metrics"]["emergency_recall"] == 1.0
    assert report["metrics"]["multi_intent_noul_recall"] == 1.0
    assert report["metrics"]["adversarial_noul_recall"] == 1.0
