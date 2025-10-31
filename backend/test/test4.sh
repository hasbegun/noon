#!/bin/bash

# This script sends a POST request to the /analyze-image/ endpoint.
#
# It simulates a form submission with three fields:
# 1. prompt: A text question about the image.
# 2. image: The image file to be analyzed ('@test.jpg' tells curl to upload this file).
# 3. engine: Specifies which inference engine to use (ollama or llamacpp).
#
# Ensure 'test.jpg' is in the same directory as this script before running.
# To run this script, make it executable with 'chmod +x test_endpoint.sh'
# and then execute it with './test_endpoint.sh'.

curl -X POST \
  -F "prompt=What food is this and what are its nutritional values?" \
  -F "image=@test_dish1.jpg" \
  -F "engine=ollama" \
  http://localhost:8000/analyze-image/

# Add a newline at the end for cleaner terminal output
echo

