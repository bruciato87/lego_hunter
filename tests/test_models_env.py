from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from models import LegoHunterRepository


class ModelsEnvTests(unittest.TestCase):
    @staticmethod
    def _make_jwt(role: str) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {"role": role}

        def _b64(data: dict) -> str:
            raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
            return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

        return f"{_b64(header)}.{_b64(payload)}.signature"

    def test_looks_like_anon_jwt(self) -> None:
        anon_token = self._make_jwt("anon")
        auth_token = self._make_jwt("authenticated")
        service_token = self._make_jwt("service_role")

        self.assertTrue(LegoHunterRepository._looks_like_anon_jwt(anon_token))
        self.assertTrue(LegoHunterRepository._looks_like_anon_jwt(auth_token))
        self.assertFalse(LegoHunterRepository._looks_like_anon_jwt(service_token))
        self.assertFalse(LegoHunterRepository._looks_like_anon_jwt("not-a-jwt"))

    def test_from_env_prefers_service_role_key(self) -> None:
        with patch("models.create_client", return_value=object()) as mocked_client:
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://example.supabase.co",
                    "SUPABASE_KEY": self._make_jwt("anon"),
                    "SUPABASE_SERVICE_ROLE_KEY": "srv-role-key",
                },
                clear=True,
            ):
                LegoHunterRepository.from_env()

        mocked_client.assert_called_once_with("https://example.supabase.co", "srv-role-key")

    def test_from_env_warns_for_anon_key_without_service_role(self) -> None:
        with patch("models.create_client", return_value=object()) as mocked_client:
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://example.supabase.co",
                    "SUPABASE_KEY": self._make_jwt("anon"),
                },
                clear=True,
            ):
                with self.assertLogs("models", level="WARNING") as captured:
                    LegoHunterRepository.from_env()

        mocked_client.assert_called_once_with(
            "https://example.supabase.co",
            self._make_jwt("anon"),
        )
        full_log = "\n".join(captured.output)
        self.assertIn("appears to be anon/publishable key", full_log)

    def test_from_env_raises_without_required_variables(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                LegoHunterRepository.from_env()


if __name__ == "__main__":
    unittest.main()
