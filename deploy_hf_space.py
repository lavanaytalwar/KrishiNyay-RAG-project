#!/usr/bin/env python3
"""
Create/update the KrishiNyay Hugging Face Space.

This script intentionally never prints secret values. It expects a cached
Hugging Face token from `huggingface-cli login` or HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent

PUBLIC_VARIABLES = {
    "DEMO_PUBLIC": "true",
    "ENABLE_LIVE_INGEST": "false",
    "CHROMA_PATH": "demo_chroma_db",
    "CHUNKS_DIR": "demo_data/chunks",
    "LLM_PROVIDER": "gemini",
    "GEMINI_MODEL": "gemini-1.5-flash",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

SECRET_KEYS = [
    "GEMINI_API_KEY",
    "DATA_GOV_IN_API_KEY",
    "AGMARKNET_API_KEY",
]

IGNORE_PATTERNS = [
    ".git/**",
    ".DS_Store",
    ".env",
    ".env.*",
    "__pycache__/**",
    "*.pyc",
    ".venv/**",
    "venv/**",
    "env/**",
    "krishinyay-env/**",
    "chroma_db/**",
    "data/**",
    "logs/**",
    "web/media/*.mp4",
    "web/media/*.mov",
    "web/media/*.webm",
]


def require_artifacts() -> None:
    required = [
        ROOT / "Dockerfile",
        ROOT / "README.md",
        ROOT / "demo_chroma_db" / "chroma.sqlite3",
        ROOT / "demo_data" / "chunks" / "all_chunks.jsonl",
        ROOT / "demo_data" / "chunks" / "embed_meta.json",
    ]
    missing = [path.relative_to(ROOT) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required demo artifacts: " + ", ".join(map(str, missing)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Space repo id, for example lavanay1/krishinyay-ai",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the Space as private instead of public.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only create/configure the Space; do not upload files.",
    )
    parser.add_argument(
        "--require-gemini-key",
        action="store_true",
        help="Fail if GEMINI_API_KEY is not present locally.",
    )
    args = parser.parse_args()

    try:
        require_artifacts()
        if args.require_gemini_key and not os.environ.get("GEMINI_API_KEY", "").strip():
            raise RuntimeError("GEMINI_API_KEY is required for launch-ready public deployment.")

        api = HfApi()
        repo_url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
            space_variables=[{"key": key, "value": value} for key, value in PUBLIC_VARIABLES.items()],
        )
        print(f"Space ready: {repo_url}")

        for key in SECRET_KEYS:
            value = os.environ.get(key, "").strip()
            if not value:
                print(f"Secret skipped: {key}=missing")
                continue
            api.add_space_secret(args.repo_id, key=key, value=value)
            print(f"Secret configured: {key}=set")

        if not args.skip_upload:
            commit = api.upload_folder(
                repo_id=args.repo_id,
                repo_type="space",
                folder_path=ROOT,
                commit_message="Deploy KrishiNyay public demo",
                ignore_patterns=IGNORE_PATTERNS,
            )
            print(f"Uploaded commit: {commit.oid}")

        print(f"URL: https://huggingface.co/spaces/{args.repo_id}")
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            print("WARNING: GEMINI_API_KEY is missing; set it as a Space secret before launch.")
        return 0
    except Exception as exc:
        print(f"Deployment failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
