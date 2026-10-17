from pathlib import Path
import pytest

from condo_jev.backends import LocalJevSimulatorBackend
from condo_jev.workflow import JevTriageService


@pytest.fixture(autouse=True)
def isolate_work_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_json = tmp_path / "isolated_work_orders.json"
    prod_json = Path(__file__).resolve().parent.parent / "data" / "work_orders.json"
    if prod_json.exists():
        test_json.write_text(prod_json.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        test_json.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CONDO_WORK_ORDERS_FILE", str(test_json))


@pytest.fixture(scope="session")
def local_jev_backend() -> LocalJevSimulatorBackend:
    return LocalJevSimulatorBackend()


@pytest.fixture(scope="session")
def jev_service(local_jev_backend: LocalJevSimulatorBackend) -> JevTriageService:
    return JevTriageService(
        backend=local_jev_backend,
        confidence_threshold=0.55,
        noul_review_threshold=0.50,
    )
