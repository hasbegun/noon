#!/bin/bash
# Wrapper script to run mask debugging utility
# Usage: ./debug_masks.sh

cd "$(dirname "$0")"
python src/utils/debug_masks.py "$@"
