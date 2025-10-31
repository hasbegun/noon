#!/bin/bash

# This script checks if our FastAPI backend service is running.

echo -e "\n[TEST 2/3] Checking if our FastAPI backend service is running..."

if curl -s --head http://127.0.0.1:8000/ 2>&1 | grep "\"status\":\"ok\"" > /dev/null; then
    echo "  > ✅ SUCCESS: Backend service is running and responding on port 8000."
    exit 0
else
    echo "  > ❌ FAILURE: Backend service is not responding. Please run 'make start' in a separate terminal."
    exit 1
fi
