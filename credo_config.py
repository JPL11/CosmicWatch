#!/usr/bin/env python3
"""
Shared local configuration for CREDO Elasticsearch scripts.

Copy .env.example to .env and fill in CREDO_USER / CREDO_PASS, or export those
variables in your shell before running the analysis scripts.
"""
import os
from pathlib import Path

DEFAULT_ES_URL = "https://credo-es.nrp-nautilus.io"
DEFAULT_INDEX = "credo-detections"
FALSE_VALUES = {"0", "false", "no", "off"}


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def es_settings():
    load_dotenv()
    return (
        os.environ.get("CREDO_ES_URL", DEFAULT_ES_URL).rstrip("/"),
        os.environ.get("CREDO_INDEX", DEFAULT_INDEX),
    )


def es_auth():
    load_dotenv()
    username = os.environ.get("CREDO_USER")
    password = os.environ.get("CREDO_PASS")
    if not username or not password:
        raise SystemExit(
            "Missing CREDO credentials. Copy .env.example to .env and set "
            "CREDO_USER/CREDO_PASS, or export them in your shell."
        )
    return username, password


def verify_certs():
    load_dotenv()
    value = os.environ.get("CREDO_VERIFY_CERTS", "false").strip().lower()
    return value not in FALSE_VALUES

