import httpx
import pytest
from PIL import Image
from io import BytesIO

# --- Test Configuration ---
BACKEND_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"

# --- Fixtures and Helper Functions ---
def is_service_running(url):
    """Checks if a service is running at the given URL using httpx."""
    try:
        # httpx uses a client model
        with httpx.Client() as client:
            response = client.get(url, timeout=5)
            # Check for any successful status code (2xx)
            return response.is_success
    except httpx.ConnectError:
        return False

# --- Test Cases ---
def test_ollama_server_is_running():
    """
    [Test 1] Checks if the Ollama server is running and responsive.
    """
    print("\n[TEST 1/3] Checking if the Ollama server is running...")
    assert is_service_running(OLLAMA_URL), \
        f"❌ FAILURE: Ollama is not responding at {OLLAMA_URL}. Please ensure it is running."
    print("  > ✅ SUCCESS: Ollama is running.")


def test_backend_service_is_running():
    """
    [Test 2] Checks if our FastAPI backend service is running.
    """
    print("\n[TEST 2/3] Checking if our FastAPI backend service is running...")
    assert is_service_running(f"{BACKEND_URL}/"), \
        f"❌ FAILURE: Backend is not responding at {BACKEND_URL}. Run 'make start' first."
    print("  > ✅ SUCCESS: Backend service is running.")


def test_image_analysis_workflow():
    """
    [Test 3] Tests the full end-to-end analysis workflow.
    """
    print("\n[TEST 3/3] Testing the full analysis workflow...")

    # 1. Create a dummy image in memory
    print("  > [SETUP] Creating a dummy red image in memory...")
    img_byte_arr = BytesIO()
    image = Image.new('RGB', (60, 30), color='red')
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # 2. Prepare the request data
    files = {'image': ('test_image.png', img_byte_arr, 'image/png')}
    payload = {'prompt': 'In one single word, what is the dominant color of this image?'}

    # 3. Send the request to the backend using httpx
    print("  > [ACTION] Sending image and prompt to the backend...")
    with httpx.Client() as client:
        response = client.post(f"{BACKEND_URL}/analyze-image/", files=files, data=payload)

    # 4. Assert the response
    assert response.status_code == 200, \
        f"❌ FAILURE: API returned status {response.status_code}. Expected 200."

    response_data = response.json()
    print(f"  > [RESULT] Received analysis: {response_data}")

    assert "analysis" in response_data, \
        "❌ FAILURE: The key 'analysis' was not found in the response."

    assert "red" in response_data["analysis"].lower(), \
        "❌ FAILURE: The analysis did not correctly identify the color 'red'."

    print("  > ✅ SUCCESS: The response correctly identified the color. Workflow confirmed.")

