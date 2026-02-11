from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any, Optional

from telegram import Bot, Update

from bot import (
    LegoHunterTelegramBot,
    _dispatch_scova_workflow,
    _dispatch_single_set_analysis_workflow,
    _normalize_set_id_token,
    build_application,
)
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
CLOUD_DISPATCH_COMMANDS = {"/scova", "/hunt", "/analizza", "/analisi"}


def _is_truthy(raw_value: Optional[str], default: bool = False) -> bool:
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _webhook_light_mode_enabled() -> bool:
    return _is_truthy(os.getenv("WEBHOOK_LIGHT_MODE"), default=True)


def _blocked_webhook_commands() -> set[str]:
    raw = str(os.getenv("WEBHOOK_BLOCKED_COMMANDS") or "").strip()
    if not raw:
        return {"/scova", "/hunt"}
    blocked: set[str] = set()
    for item in raw.split(","):
        cmd = str(item or "").strip().lower()
        if not cmd:
            continue
        if not cmd.startswith("/"):
            cmd = f"/{cmd}"
        blocked.add(cmd)
    return blocked or {"/scova", "/hunt"}


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


def _extract_command_from_payload(payload: dict[str, Any]) -> str:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        message = payload.get(key)
        if not isinstance(message, dict):
            continue
        raw_text = str(message.get("text") or message.get("caption") or "").strip()
        if not raw_text.startswith("/"):
            continue
        command = raw_text.split(None, 1)[0].strip().lower()
        # Telegram supports commands like /help@my_bot
        command = command.split("@", 1)[0]
        return command
    return ""


def _extract_text_from_payload(payload: dict[str, Any]) -> str:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        message = payload.get(key)
        if not isinstance(message, dict):
            continue
        raw_text = str(message.get("text") or message.get("caption") or "").strip()
        if raw_text:
            return raw_text
    return ""


def _extract_command_args_from_payload(payload: dict[str, Any]) -> list[str]:
    raw_text = _extract_text_from_payload(payload)
    if not raw_text.startswith("/"):
        return []
    parts = raw_text.split()
    if len(parts) <= 1:
        return []
    return [str(item).strip() for item in parts[1:] if str(item).strip()]


def _extract_chat_and_message_id(payload: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
        message = payload.get(key)
        if not isinstance(message, dict):
            continue
        chat = message.get("chat")
        if not isinstance(chat, dict):
            continue
        chat_id_raw = chat.get("id")
        message_id_raw = message.get("message_id")
        try:
            chat_id = int(chat_id_raw)
        except (TypeError, ValueError):
            continue
        try:
            message_id = int(message_id_raw) if message_id_raw is not None else None
        except (TypeError, ValueError):
            message_id = None
        return chat_id, message_id
    return None, None


async def _send_webhook_message(
    *,
    token: str,
    chat_id: int,
    reply_to_message_id: Optional[int],
    text: str,
) -> None:
    bot = Bot(token=token)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_to_message_id=reply_to_message_id,
            disable_web_page_preview=True,
            connect_timeout=8,
            read_timeout=12,
            write_timeout=12,
            pool_timeout=8,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to send webhook message | chat_id=%s error=%s", chat_id, exc)
    finally:
        try:
            await bot.shutdown()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Bot shutdown warning after lightweight notice", exc_info=True)


async def _send_webhook_light_notice(
    *,
    token: str,
    payload: dict[str, Any],
    command: str,
) -> None:
    chat_id, reply_to_message_id = _extract_chat_and_message_id(payload)
    if chat_id is None:
        return
    notice = (
        "Comando non disponibile nel webhook leggero.\n"
        "Usa /acquista, /venduto, /analizza, /radar, /cerca, /collezione, /vendi, /help.\n"
        "La discovery completa (/scova) gira nei cicli schedulati cloud."
    )
    await _send_webhook_message(
        token=token,
        chat_id=chat_id,
        reply_to_message_id=reply_to_message_id,
        text=notice,
    )
    LOGGER.info("Webhook lightweight notice sent | command=%s chat_id=%s", command, chat_id)


async def _maybe_dispatch_cloud_command_from_webhook(
    *,
    token: str,
    payload: dict[str, Any],
    command: str,
) -> bool:
    if command not in CLOUD_DISPATCH_COMMANDS:
        return False

    chat_id, reply_to_message_id = _extract_chat_and_message_id(payload)
    if chat_id is None:
        return True

    allowed_chat_id = str(os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if allowed_chat_id and str(chat_id) != allowed_chat_id:
        await _send_webhook_message(
            token=token,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
            text="Chat non autorizzata.",
        )
        LOGGER.warning("Webhook cloud dispatch rejected for unauthorized chat_id=%s", chat_id)
        return True

    if command in {"/scova", "/hunt"}:
        ok, detail = await asyncio.to_thread(
            _dispatch_scova_workflow,
            chat_id=str(chat_id),
        )
        message = (
            detail
            if ok
            else (
                "Impossibile avviare /scova su GitHub.\n"
                f"Dettaglio: {detail}\n"
                "Config richiesta: GITHUB_ACTIONS_DISPATCH_TOKEN, GITHUB_REPO, "
                "GITHUB_WORKFLOW_FILE (opzionale), GITHUB_WORKFLOW_REF (opzionale)."
            )
        )
        await _send_webhook_message(
            token=token,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
            text=message,
        )
        LOGGER.info("Webhook cloud dispatch /scova handled | chat_id=%s ok=%s", chat_id, ok)
        return True

    if command in {"/analizza", "/analisi"}:
        args = _extract_command_args_from_payload(payload)
        set_id = _normalize_set_id_token(args[0] if args else "")
        if not set_id:
            await _send_webhook_message(
                token=token,
                chat_id=chat_id,
                reply_to_message_id=reply_to_message_id,
                text="Uso: /analizza <set_id>  (esempio: /analizza 76441)",
            )
            return True

        ok, detail = await asyncio.to_thread(
            _dispatch_single_set_analysis_workflow,
            set_id=set_id,
            chat_id=str(chat_id),
        )
        message = (
            detail
            if ok
            else (
                "Impossibile avviare l'analisi su GitHub.\n"
                f"Dettaglio: {detail}\n"
                "Config richiesta: GITHUB_ACTIONS_DISPATCH_TOKEN, GITHUB_REPO, "
                "GITHUB_WORKFLOW_FILE (opzionale), GITHUB_WORKFLOW_REF (opzionale)."
            )
        )
        await _send_webhook_message(
            token=token,
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
            text=message,
        )
        LOGGER.info("Webhook cloud dispatch /analizza handled | chat_id=%s set_id=%s ok=%s", chat_id, set_id, ok)
        return True

    return False


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

    command = _extract_command_from_payload(payload)
    if _webhook_light_mode_enabled() and command:
        if await _maybe_dispatch_cloud_command_from_webhook(
            token=token,
            payload=payload,
            command=command,
        ):
            return

    if _webhook_light_mode_enabled() and command and command in _blocked_webhook_commands():
        LOGGER.info("Webhook lightweight command block | command=%s", command)
        await _send_webhook_light_notice(token=token, payload=payload, command=command)
        return

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
