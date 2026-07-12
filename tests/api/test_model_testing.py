import pytest
from fastapi.testclient import TestClient
from tests.api.support import create_test_app

app = create_test_app()

def test_model_testing_routes_success():
    client = TestClient(app, client=("127.0.0.1", 50000))
    
    # 1. Get models list
    response = client.get("/admin/api/test-models/models")
    assert response.status_code == 200
    data = response.json()
    assert "grouped" in data

    # 2. Run model tests
    run_response = client.post("/admin/api/test-models/run", json={"models": ["deepseek/deepseek-chat"]})
    assert run_response.status_code == 200
    run_data = run_response.json()
    assert run_data["status"] == "started"
    assert run_data["total"] == 1

    # 3. Check status
    status_response = client.get("/admin/api/test-models/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert "is_running" in status_data
    assert "results" in status_data
