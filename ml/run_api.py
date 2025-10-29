#!/usr/bin/env python
"""
Entry point for running the API server
This ensures proper module paths for relative imports
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from src.api.main import run_server
    run_server()
