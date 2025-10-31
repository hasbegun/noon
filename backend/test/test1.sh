#!/bin/bash

# This script checks if the Ollama server is running and responsive.

echo -e "\n[TEST 1/3] Checking if the Ollama server is running..."

if curl -s --head http://localhost:11434/ 2>&1 | grep "200 OK" > /dev/null; then
    echo "  > ✅ SUCCESS: Ollama is running and responding on port 11434."
    exit 0
else
    echo "  > ❌ FAILURE: Ollama is not responding. Please ensure it is installed and running."
    exit 1
fi
