#!/usr/bin/env python3
"""
Run the FastAPI server
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import run_server
from src.config import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Food Detection API server")

    parser.add_argument("--host", type=str, default=config.api_host, help="Server host")
    parser.add_argument("--port", type=int, default=config.api_port, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
