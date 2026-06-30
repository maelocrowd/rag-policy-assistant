"""
tests/test_smoke.py

Post-deployment smoke test targeting the live production infrastructure.
"""

import os
import requests
import pytest

# Extract the live URL from the pipeline environment
LIVE_BACKEND_URL = os.getenv("PRODUCTION_BACKEND_URL")


def test_production_backend_url_exists():
    """Verify that the production target URL is provided to the script."""
    assert LIVE_BACKEND_URL is not None, "PRODUCTION_BACKEND_URL environment variable is missing"
    assert LIVE_BACKEND_URL.startswith("http"), f"Invalid production URL format: {LIVE_BACKEND_URL}"


def test_live_production_health():
    """Test 1: Invoke the /health route on the live production backend server."""
    endpoint = f"{LIVE_BACKEND_URL.rstrip('/')}/health"
    try:
        response = requests.get(endpoint, timeout=15)
        assert response.status_code == 200, f"Production returned server error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Smoke Test Failed: Unable to reach production endpoint {endpoint}. Error: {e}")


def test_live_production_chat():
    """Test 2: Ensure the core /chat RAG pipeline works over the live production endpoint."""
    endpoint = f"{LIVE_BACKEND_URL.rstrip('/')}/chat"
    payload = {"question": "Smoke test validation query"}
    try:
        response = requests.post(endpoint, json=payload, timeout=20)
        assert response.status_code == 200, f"Production /chat failed with status code {response.status_code}"
        
        data = response.json()
        assert "answer" in data, "Production response payload missing 'answer' string"
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Smoke Test Failed: Production /chat communication dropped. Error: {e}")
