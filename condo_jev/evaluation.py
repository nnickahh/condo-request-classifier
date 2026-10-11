import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from condo_jev.backends import DEFAULT_TRAIN_CSV, LocalJevSimulatorBackend, ROOT_DIR
from condo_jev.schema import Category, Priority, ResidentRequest
from condo_jev.workflow import JevTriageService

DEFAULT_EVAL_CSV = ROOT_DIR / "data" / "evaluation.csv"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "jev-evaluation-report.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def verify_no_train_eval_leakage(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    eval_csv: Path = DEFAULT_EVAL_CSV,
) -> None:
    with train_csv.open(encoding="utf-8-sig", newline="") as f_train:
        train_keys = {normalize_text_key(row["text"]) for row in csv.DictReader(f_train)}
    with eval_csv.open(encoding="utf-8-sig", newline="") as f_eval:
        eval_keys = {normalize_text_key(row["text"]) for row in csv.DictReader(f_eval)}
    overlap = train_keys & eval_keys
    if overlap:
        raise ValueError(f"Detected {len(overlap)} exact leaked examples between train and evaluation splits.")


def compute_brier_and_ece(
    outcomes: list[tuple[dict[str, float], str, float]],
    num_bins: int = 5,
) -> tuple[float, float]:
    """
    Compute multi-class Brier score and top-label Expected Calibration Error (ECE)
    for Jev Choice probability distributions.
    """
    if not outcomes:
        return 0.0, 0.0

    brier_terms: list[float] = []
    bin_buckets: list[list[tuple[float, int]]] = [[] for _ in range(num_bins)]

    for probs, gold_cat, _ in outcomes:
        if not probs:
            continue
        sq_err = sum(
            (float(p) - (1.0 if cat_key == gold_cat else 0.0)) ** 2
            for cat_key, p in probs.items()
        ) / len(probs)
        brier_terms.append(sq_err)

        pred_cat = max(probs, key=probs.get)
        top_p = float(probs[pred_cat])
        correct = 1 if pred_cat == gold_cat else 0
        idx = min(num_bins - 1, max(0, int(top_p * num_bins)))
        bin_buckets[idx].append((top_p, correct))

    brier = round(mean(brier_terms), 4) if brier_terms else 0.0
    total = sum(len(b) for b in bin_buckets)
    ece = 0.0
    if total > 0:
        for bucket in bin_buckets:
            if not bucket:
                continue
            avg_conf = mean(item[0] for item in bucket)
            avg_acc = mean(item[1] for item in bucket)
            ece += (len(bucket) / total) * abs(avg_conf - avg_acc)

    return brier, round(ece, 4)


def run_evaluation(
    service: JevTriageService | None = None,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    eval_csv: Path = DEFAULT_EVAL_CSV,
) -> dict[str, Any]:
    """Run the full Jev System-One evaluation benchmark on `data/evaluation.csv`."""
    verify_no_train_eval_leakage(train_csv, eval_csv)

    if service is None:
        service = JevTriageService(backend=LocalJevSimulatorBackend(train_csv=train_csv))

    with eval_csv.open(encoding="utf-8-sig", newline="") as handle:
        eval_rows = list(csv.DictReader(handle))

    case_records: list[dict[str, Any]] = []
    calibration_inputs: list[tuple[dict[str, float], str, float]] = []

    for idx, row in enumerate(eval_rows, start=1):
        req = ResidentRequest(
            text=row["text"],
            channel=row.get("channel", "Resident App") or "Resident App",  # type: ignore[arg-type]
            location=row.get("location") or None,
        )
        gold_category = Category(row["category"]).value
        gold_priority = Priority(row["priority"]).value
        gold_review = row["expected_review"].strip().lower() in {"true", "1", "yes"}
        case_Slice = row.get("slice", "standard").strip()

        decision = service.classify(req)
        pred_category = decision.category.value
        pred_priority = decision.priority.value

        calibration_inputs.append(
            (decision.category_probabilities, gold_category, decision.choice_confidence)
        )

        case_records.append(
            {
                "id": idx,
                "slice": case_Slice,
                "text": row["text"],
                "gold_category": gold_category,
                "predicted_category": pred_category,
                "category_correct": pred_category == gold_category,
                "gold_priority": gold_priority,
                "predicted_priority": pred_priority,
                "priority_correct": pred_priority == gold_priority,
                "gold_review": gold_review,
                "predicted_review": decision.needs_human_review,
                "review_correct": decision.needs_human_review == gold_review,
                "choice_confidence": decision.choice_confidence,
                "urgency_score": decision.urgency_score,
                "nouls": decision.nouls,
                "review_reasons": decision.review_reasons,
                "backend": decision.backend,
                "model": decision.model,
                "latency_ms": decision.latency_ms,
            }
        )

    total = len(case_records)
    cat_hits = sum(1 for r in case_records if r["category_correct"])
    prio_hits = sum(1 for r in case_records if r["priority_correct"])
    rev_hits = sum(1 for r in case_records if r["review_correct"])

    emergency_cases = [r for r in case_records if r["gold_priority"] == Priority.EMERGENCY.value]
    emergency_hits = sum(
        1
        for r in emergency_cases
        if r["predicted_priority"] == Priority.EMERGENCY.value and r["predicted_review"]
    )

    multi_cases = [r for r in case_records if r["slice"] == "multi_intent"]
    multi_noul_hits = sum(
        1 for r in multi_cases if r["nouls"].get("multi_intent_request", 0.0) >= 0.50
    )

    adversarial_cases = [r for r in case_records if r["slice"] == "adversarial"]
    adversarial_noul_hits = sum(
        1 for r in adversarial_cases if r["nouls"].get("adversarial_or_sensitive", 0.0) >= 0.50
    )

    brier_score, ece_score = compute_brier_and_ece(calibration_inputs)

    per_category: list[dict[str, Any]] = []
    for cat in Category:
        c_val = cat.value
        tp = sum(1 for r in case_records if r["gold_category"] == c_val and r["predicted_category"] == c_val)
        fp = sum(1 for r in case_records if r["gold_category"] != c_val and r["predicted_category"] == c_val)
        fn = sum(1 for r in case_records if r["gold_category"] == c_val and r["predicted_category"] != c_val)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_category.append(
            {
                "category": c_val,
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    correct_confs = [r["choice_confidence"] for r in case_records if r["category_correct"]]
    incorrect_confs = [r["choice_confidence"] for r in case_records if not r["category_correct"]]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "train_rows": sum(1 for _ in csv.DictReader(train_csv.open(encoding="utf-8-sig"))),
            "eval_rows": total,
            "train_sha256": file_sha256(train_csv),
            "eval_sha256": file_sha256(eval_csv),
        },
        "metrics": {
            "category_accuracy": round(cat_hits / total, 4) if total else 0.0,
            "priority_accuracy": round(prio_hits / total, 4) if total else 0.0,
            "review_policy_accuracy": round(rev_hits / total, 4) if total else 0.0,
            "emergency_recall": round(emergency_hits / len(emergency_cases), 4) if emergency_cases else 1.0,
            "multi_intent_noul_recall": round(multi_noul_hits / len(multi_cases), 4) if multi_cases else 1.0,
            "adversarial_noul_recall": round(adversarial_noul_hits / len(adversarial_cases), 4) if adversarial_cases else 1.0,
            "brier_score": brier_score,
            "expected_calibration_error": ece_score,
            "mean_confidence_correct": round(mean(correct_confs), 4) if correct_confs else 0.0,
            "mean_confidence_incorrect": round(mean(incorrect_confs), 4) if incorrect_confs else 0.0,
            "mean_latency_ms": round(mean(r["latency_ms"] for r in case_records), 2) if case_records else 0.0,
        },
        "per_category": per_category,
        "cases": case_records,
    }


def save_evaluation_report(
    report: dict[str, Any],
    output_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
