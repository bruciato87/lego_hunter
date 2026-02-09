from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from api.telegram_webhook import (
    _blocked_webhook_commands,
    _extract_command_from_payload,
    _extract_chat_and_message_id,
    _has_valid_secret,
    _is_health_path,
    _is_webhook_path,
    _normalize_path,
    _webhook_light_mode_enabled,
)


class WebhookEndpointTests(unittest.TestCase):
    def test_normalize_path_handles_query_and_trailing_slash(self) -> None:
        self.assertEqual(_normalize_path("/api/telegram_webhook/?a=1"), "/api/telegram_webhook")
        self.assertEqual(_normalize_path(""), "/")

    def test_webhook_path_matching(self) -> None:
        self.assertTrue(_is_webhook_path("/api/telegram_webhook"))
        self.assertTrue(_is_webhook_path("/telegram/webhook"))
        self.assertTrue(_is_webhook_path("/foo/bar/telegram_webhook"))
        self.assertFalse(_is_webhook_path("/api/other"))

    def test_health_path_matching(self) -> None:
        self.assertTrue(_is_health_path("/"))
        self.assertTrue(_is_health_path("/healthz"))
        self.assertTrue(_is_health_path("/api/telegram_webhook/healthz"))
        self.assertFalse(_is_health_path("/api/telegram_webhook"))

    def test_secret_validation_requires_match(self) -> None:
        with patch.dict(os.environ, {"TELEGRAM_WEBHOOK_SECRET": "abc123"}, clear=False):
            self.assertTrue(_has_valid_secret("abc123"))
            self.assertFalse(_has_valid_secret("wrong"))
            self.assertFalse(_has_valid_secret(None))

    def test_secret_validation_fails_when_missing_expected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_has_valid_secret("anything"))

    def test_extract_command_from_payload_supports_mentions(self) -> None:
        payload = {
            "message": {
                "text": "/scova@lego_hunter_bot adesso",
                "chat": {"id": 12345},
                "message_id": 88,
            }
        }
        self.assertEqual(_extract_command_from_payload(payload), "/scova")

    def test_extract_chat_and_message_id(self) -> None:
        payload = {
            "message": {
                "text": "/help",
                "chat": {"id": 12345},
                "message_id": 88,
            }
        }
        self.assertEqual(_extract_chat_and_message_id(payload), (12345, 88))

    def test_webhook_light_mode_default_enabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_webhook_light_mode_enabled())

    def test_blocked_webhook_commands_default_and_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_blocked_webhook_commands(), {"/scova", "/hunt"})
        with patch.dict(os.environ, {"WEBHOOK_BLOCKED_COMMANDS": "scova, hunt,offerte"}, clear=True):
            self.assertEqual(_blocked_webhook_commands(), {"/scova", "/hunt", "/offerte"})


if __name__ == "__main__":
    unittest.main()
