import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from condo_jev.backends import JevCloudBackend, LocalJevSimulatorBackend, OllamaJevBackend
from condo_jev.batch import batch_results_to_json, classify_csv_bytes
from condo_jev.evaluation import DEFAULT_REPORT_PATH, run_evaluation, save_evaluation_report
from condo_jev.schema import ResidentRequest
from condo_jev.workflow import JevTriageService


def build_backend(backend_name: str, model: str | None = None):
    if backend_name == "jev_local":
        return LocalJevSimulatorBackend()
    if backend_name == "jev_cloud":
        return JevCloudBackend(model=model)
    if backend_name == "ollama_jev":
        return OllamaJevBackend(model=model)
    raise ValueError(f"Unknown backend: {backend_name}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m condo_jev",
        description="Condo Jev Triage CLI — evaluate resident requests using Jev System-One primitives (Choice, Score, Noul).",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Single resident request message to classify.",
    )
    parser.add_argument(
        "--backend",
        choices=["jev_local", "jev_cloud", "ollama_jev"],
        default="jev_local",
        help="Decision engine backend (default: jev_local).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override (e.g. jev-latest or gemma3:4b).",
    )
    parser.add_argument(
        "--channel",
        choices=["Resident App", "WhatsApp", "Email", "Intercom / Call Log", "Web"],
        default="Resident App",
        help="Input communication channel.",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional unit/block/facility location.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Minimum Jev Choice concentration confidence before staff review is triggered.",
    )
    parser.add_argument(
        "--noul-threshold",
        type=float,
        default=0.50,
        help="Jev Noul probability threshold for triggering human review.",
    )
    parser.add_argument(
        "--batch-csv",
        type=Path,
        default=None,
        help="Path to a CSV file with a 'text' column for batch classification.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the full benchmark evaluation on data/evaluation.csv and write reports/jev-evaluation-report.json.",
    )
    args = parser.parse_args(argv)

    backend = build_backend(args.backend, args.model)
    service = JevTriageService(
        backend=backend,
        confidence_threshold=args.confidence_threshold,
        noul_review_threshold=args.noul_threshold,
    )

    if args.evaluate:
        report = run_evaluation(service=service)
        saved_path = save_evaluation_report(report, DEFAULT_REPORT_PATH)
        print(
            json.dumps(
                {
                    "report_saved_to": str(saved_path),
                    "dataset": report["dataset"],
                    "metrics": report["metrics"],
                },
                indent=2,
            )
        )
        return 0

    if args.batch_csv is not None:
        raw_bytes = args.batch_csv.read_bytes()
        records = classify_csv_bytes(raw_bytes, service)
        print(batch_results_to_json(records))
        return 0

    if not args.text:
        parser.error("Provide a resident request message, --batch-csv PATH, or --evaluate.")

    req = ResidentRequest(text=args.text, channel=args.channel, location=args.location)
    result = service.classify(req)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
