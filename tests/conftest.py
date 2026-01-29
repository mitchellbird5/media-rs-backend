import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="session")
def api_client():
    return TestClient(app)
