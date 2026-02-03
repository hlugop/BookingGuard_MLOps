#!/usr/bin/env python
"""
API Runner Script
=================

Convenience script to start the FastAPI server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080 --reload
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Start the FastAPI server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Start the prediction API")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="MLflow model URI",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/models",
        help="Artifacts directory",
    )

    args = parser.parse_args()

    # Set environment variables
    if args.model_uri:
        os.environ["MODEL_URI"] = args.model_uri
    os.environ["ARTIFACTS_DIR"] = args.artifacts_dir

    print(f"\nStarting API server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Artifacts: {args.artifacts_dir}")
    print(f"\nAPI docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
