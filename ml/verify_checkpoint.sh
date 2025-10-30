#!/bin/bash
# Wrapper script to verify checkpoint functionality
# Usage: ./verify_checkpoint.sh

cd "$(dirname "$0")"
python src/utils/verify_checkpoint.py "$@"
