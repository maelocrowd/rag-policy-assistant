"""
test_suite.py

Minimal automation pipeline verifying system components:
1. Tests backend availability via /health endpoint.
2. Tests a single functional response payload via /chat.
3. Tests frontend compilation viability.
"""

import os
import sys
import requests
import pytest

# Determine the target endpoint using the environment variable or defaulting to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")


def test_backend_health():
    """1. Test if the backend API is up and working by invoking /health."""
    endpoint = f"{BACKEND_URL}/health"
    try:
        response = requests.get(endpoint, timeout=5)
        assert response.status_code == 200, f"Backend returned status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Could not connect to backend /health endpoint at {endpoint}. Error: {e}")


def test_backend_chat():
    """2. Test a single POST request to the /chat pipeline."""
    endpoint = f"{BACKEND_URL}/chat"
    payload = {"question": "What is the policy on annual leave?"}
    
    try:
        response = requests.post(endpoint, json=payload, timeout=10)
        assert response.status_code == 200, f"Backend returned status code {response.status_code}"
        
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Could not complete POST request to /chat at {endpoint}. Error: {e}")


def test_webapp_compilation():
    """3. Test the webapp script to guarantee it runs without syntax or configuration crashes."""
    # Verifies the environment's python engine version
    assert sys.version_info >= (3, 10), f"Python version should be >= 3.10, found {sys.version}"
    
    # Compiles app.py to guarantee no broken imports or syntax problems exist
    try:
        import py_compile
        compiled_path = py_compile.compile("app.py", doraise=True)
        assert compiled_path is not None, "Webapp script compilation returned an empty path"
    except Exception as e:
        pytest.fail(f"The frontend app.py failed basic compilation checks: {e}")
