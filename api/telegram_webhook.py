from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any, Optional

from telegram import Update

from bot import LegoHunterTelegramBot, build_application
from fiscal import FiscalGuardian
from models import LegoHunterRepository
from oracle import DiscoveryOracle

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("lego_hunter.webhook")

WEBHOOK_PATHS = {"/api/telegram_webhook", "/telegram/webhook"}
HEALTH_PATHS = {"/", "/healthz", "/api/telegram_webhook/healthz"}

_MANAGER: Optional[LegoHunterTelegramBot] = None


def _normalize_path(path: str) -> str:
    raw = (path or "").split("?", 1)[0].strip()
    if not raw:
        return "/"
    normalized = raw.rstrip("/")
    return normalized or "/"


def _is_webhook_path(path: str) -> bool:
    normalized = _normalize_path(path)
    if normalized in WEBHOOK_PATHS:
        return True
    return normalized.endswith("/telegram_webhook")


def _is_health_path(path: str) -> bool:
    normalized = _normalize_path(path)
    return normalized in HEALTH_PATHS


def _expected_webhook_secret() -> str:
    return (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()


def _has_valid_secret(received_secret: Optional[str]) -> bool:
    expected = _expected_webhook_secret()
    if not expected:
        return False
    if not received_secret:
        return False
    return hmac.compare_digest(expected, str(received_secret))


def _build_manager_from_env() -> LegoHunterTelegramBot:
    repository = LegoHunterRepository.from_env()
    fiscal_guardian = FiscalGuardian(repository)
    return LegoHunterTelegramBot(
        repository=repository,
        oracle=None,
        oracle_factory=lambda: DiscoveryOracle(repository=repository),
        fiscal_guardian=fiscal_guardian,
        allowed_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    )


def _get_manager() -> LegoHunterTelegramBot:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = _build_manager_from_env()
    return _MANAGER


async def _process_update_payload(payload: dict[str, Any]) -> None:
    token = (os.getenv("TELEGRAM_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_TOKEN")

    manager = _get_manager()
    app = build_application(manager, token, register_commands_on_init=False)

    initialized = False
    try:
        await app.initialize()
        initialized = True
        update = Update.de_json(payload, app.bot)
        if update is None:
            raise ValueError("Invalid Telegram update payload")
        await app.process_update(update)
    finally:
        if initialized:
            try:
                await app.shutdown()
            except Exception:  # noqa: BLE001
                LOGGER.warning("Application shutdown warning in webhook flow", exc_info=True)


class handler(BaseHTTPRequestHandler):
    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if _is_health_path(self.path):
            self._write_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "lego_hunter_telegram_webhook",
                    "path": _normalize_path(self.path),
                },
            )
            return

        if _is_webhook_path(self.path):
            # Endpoint exists, but Telegram sends updates via POST.
            self._write_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {
                    "ok": False,
                    "error": "use POST for Telegram updates",
                },
            )
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not _is_webhook_path(self.path):
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        expected_secret = _expected_webhook_secret()
        if not expected_secret:
            LOGGER.error("Webhook misconfigured: TELEGRAM_WEBHOOK_SECRET missing")
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "server misconfigured"},
            )
            return

        received_secret = self.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not _has_valid_secret(received_secret):
            LOGGER.warning("Webhook rejected: invalid secret header")
            self._write_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "forbidden"})
            return

        content_length_header = self.headers.get("Content-Length") or "0"
        try:
            content_length = int(content_length_header)
        except ValueError:
            content_length = 0

        if content_length <= 0:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "empty body"})
            return

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json"})
            return

        try:
            asyncio.run(_process_update_payload(payload))
        except ValueError as exc:
            LOGGER.warning("Invalid Telegram update payload: %s", exc)
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid update payload"})
            return
        except Exception:  # noqa: BLE001
            LOGGER.exception("Webhook processing failed")
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "internal error"})
            return

        self._write_json(HTTPStatus.OK, {"ok": True})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        LOGGER.info("HTTP %s", format % args)
