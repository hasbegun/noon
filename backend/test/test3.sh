#!/bin/bash

# This script tests the full end-to-end workflow:
# 1. Downloads a test image.
# 2. Sends it to the backend for analysis.
# 3. Verifies that the response from Ollama is correct.

echo -e "\n[TEST 3/3] Sending image to backend to test the full analysis workflow..."

# --- Variables ---
TEST_IMAGE_URL="https://placehold.co/100x100/ff0000/ffffff.png?text=RED"
TEST_IMAGE_FILE="test_red_square.png"
OUTPUT_FILE="test_output.json"
PROMPT="In one single word, what is the dominant color of this image?"
API_URL="http://127.0.0.1:8000/analyze-image/"

# --- Test Execution ---
# 1. Download the image
echo "  > [SETUP] Downloading test image..."
curl -s -L "$TEST_IMAGE_URL" -o "$TEST_IMAGE_FILE"

# 2. Send the request to the backend
echo "  > [ACTION] Sending image and prompt to the backend..."
curl -s -X POST \
    -F "prompt=$PROMPT" \
    -F "image=@$TEST_IMAGE_FILE" \
    "$API_URL" > "$OUTPUT_FILE"

echo "  > [RESULT] Received the following analysis from the server:"
python -m json.tool "$OUTPUT_FILE"

# 3. Verify the result
if grep -qi "red" "$OUTPUT_FILE"; then
    echo "  > ✅ SUCCESS: The response correctly identified the color. Workflow confirmed."
    rm "$TEST_IMAGE_FILE" "$OUTPUT_FILE"
    exit 0
else
    echo "  > ❌ FAILURE: The response did not contain the expected analysis."
    rm "$TEST_IMAGE_FILE" "$OUTPUT_FILE"
    exit 1
fi
