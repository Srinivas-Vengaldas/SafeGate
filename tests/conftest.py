from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = "https://upstream.test/v1"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        upstream_base_url=UPSTREAM,
        upstream_api_key="server-key",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        policy_dir=str(ROOT / "policies"),
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c
