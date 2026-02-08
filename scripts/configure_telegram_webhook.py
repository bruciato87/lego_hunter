from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests


TELEGRAM_API_BASE = "https://api.telegram.org"


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _normalize_base_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("Empty base URL")
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    return f"https://{value.rstrip('/')}"


def _telegram_call(token: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    response = requests.post(url, json=payload, timeout=30) if payload is not None else requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error on {method}: {data}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure Telegram webhook for Lego Hunter on Vercel")
    parser.add_argument(
        "--base-url",
        default=os.getenv("WEBHOOK_BASE_URL") or os.getenv("VERCEL_URL") or "",
        help="Public base URL (es. https://lego-hunter.vercel.app). Can also come from WEBHOOK_BASE_URL/VERCEL_URL.",
    )
    parser.add_argument(
        "--drop-pending-updates",
        action="store_true",
        help="If set, Telegram drops pending updates when registering webhook.",
    )
    args = parser.parse_args()

    try:
        token = _required_env("TELEGRAM_TOKEN")
        secret = _required_env("TELEGRAM_WEBHOOK_SECRET")
        base_url = _normalize_base_url(args.base_url)
    except Exception as exc:  # noqa: BLE001
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    webhook_url = f"{base_url}/api/telegram_webhook"

    payload = {
        "url": webhook_url,
        "secret_token": secret,
        "allowed_updates": ["message"],
        "drop_pending_updates": bool(args.drop_pending_updates),
    }

    try:
        set_result = _telegram_call(token, "setWebhook", payload)
        info_result = _telegram_call(token, "getWebhookInfo")
    except Exception as exc:  # noqa: BLE001
        print(f"Telegram webhook setup failed: {exc}", file=sys.stderr)
        return 1

    info = info_result.get("result") or {}
    print("Webhook configured successfully.")
    print(f"setWebhook result: {set_result.get('description')}")
    print(f"Webhook URL: {info.get('url')}")
    print(f"Pending updates: {info.get('pending_update_count')}")
    print(f"Last error date: {info.get('last_error_date')}")
    print(f"Last error message: {info.get('last_error_message')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
