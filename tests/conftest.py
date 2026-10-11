import pytest

from condo_jev.backends import LocalJevSimulatorBackend
from condo_jev.workflow import JevTriageService


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
