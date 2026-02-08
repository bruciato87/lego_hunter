from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from telegram.error import TimedOut

from bot import LegoHunterTelegramBot, run_scheduled_cycle


class DummyMessage:
    def __init__(self) -> None:
        self.replies = []

    async def reply_text(self, text, **kwargs):  # noqa: ANN001
        self.replies.append(text)


class DummyChat:
    def __init__(self, chat_id: str = "111") -> None:
        self.id = chat_id


class DummyUpdate:
    def __init__(self, chat_id: str = "111") -> None:
        self.effective_chat = DummyChat(chat_id)
        self.message = DummyMessage()


class DummyContext:
    def __init__(self, args=None):  # noqa: ANN001
        self.args = args or []


class FakeRepo:
    def get_top_opportunities(self, limit=8, min_score=50):  # noqa: ANN001
        return [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "ai_investment_score": 88,
                "market_demand_score": 79,
            }
        ]

    def search_opportunities(self, query_text, limit=10):  # noqa: ANN001
        return [
            {
                "set_id": "10316",
                "set_name": "Rivendell",
                "theme": "Icons",
                "ai_investment_score": 85,
            }
        ]

    def get_portfolio(self, status="holding"):  # noqa: ANN001
        return [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "quantity": 1,
                "purchase_price": 90.0,
                "shipping_in_cost": 5.0,
            }
        ]

    def get_best_secondary_price(self, set_id):  # noqa: ANN001
        return {"platform": "vinted", "price": 130.0}

    def get_latest_price(self, set_id, platform=None):  # noqa: ANN001
        return {"platform": platform or "vinted", "price": 140.0}


class FakeOracle:
    async def discover_opportunities(self, persist=True, top_limit=20):  # noqa: ANN001
        return [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "ai_investment_score": 90,
                "market_demand_score": 81,
                "current_price": 129.99,
                "eol_date_prediction": "2026-06-01",
            }
        ]

    async def validate_secondary_deals(self, opportunities):  # noqa: ANN001
        return [
            {
                **opportunities[0],
                "secondary_best_price": 110.0,
                "secondary_platform": "vinted",
                "discount_vs_primary_pct": 15.0,
            }
        ]

    async def discover_with_diagnostics(self, persist=True, top_limit=12, fallback_limit=3):  # noqa: ANN001
        return {
            "selected": [],
            "diagnostics": {
                "fallback_used": True,
                "source_raw_counts": {
                    "lego_proxy_reader": 0,
                    "amazon_proxy_reader": 0,
                    "lego_retiring": 0,
                    "amazon_bestsellers": 0,
                    "lego_http_fallback": 0,
                    "amazon_http_fallback": 0,
                },
                "dedup_candidates": 0,
                "threshold": 60,
                "above_threshold_count": 0,
                "max_ai_score": 0,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }


class FakeFiscal:
    def __init__(self, status):  # noqa: ANN001
        self._status = status

    def check_safety_status(self):
        return self._status


class BotTests(unittest.IsolatedAsyncioTestCase):
    async def test_help_lists_new_lego_commands(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        await manager.cmd_help(update, DummyContext())

        self.assertTrue(update.message.replies)
        text = update.message.replies[-1]
        self.assertIn("/scova", text)
        self.assertIn("/offerte", text)
        self.assertIn("/collezione", text)
        self.assertIn("Alias compatibilita'", text)
        self.assertIn("/hunt -> /scova", text)
        self.assertIn("Esempi rapidi", text)

    async def test_cerca_returns_results(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        await manager.cmd_cerca(update, DummyContext(args=["Rivendell"]))

        self.assertTrue(update.message.replies)
        self.assertIn("Rivendell", update.message.replies[-1])

    async def test_vendi_blocked_when_fiscal_not_safe(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal(
                {
                    "allow_sell_signals": False,
                    "status": "YELLOW",
                    "message": "WARNING DAC7",
                }
            ),
        )
        update = DummyUpdate()

        await manager.cmd_vendi(update, DummyContext())

        self.assertTrue(update.message.replies)
        self.assertIn("sospesi", update.message.replies[-1].lower())

    def test_format_discovery_report_shows_source_pipeline(self) -> None:
        report = {
            "selected": [],
            "diagnostics": {
                "fallback_used": True,
                "source_raw_counts": {
                    "lego_proxy_reader": 3,
                    "amazon_proxy_reader": 0,
                    "lego_retiring": 0,
                    "amazon_bestsellers": 0,
                    "lego_http_fallback": 0,
                    "amazon_http_fallback": 0,
                },
                "dedup_candidates": 3,
                "threshold": 60,
                "above_threshold_count": 0,
                "max_ai_score": 58,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "gemini", "model": "gemini-1.5-flash", "mode": "api"},
            },
        }
        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("Pipeline fonti", joined)
        self.assertIn("external_first", joined)
        self.assertIn("external_proxy", joined)

    def test_format_discovery_report_adds_clickable_lego_link(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "10332",
                    "set_name": "Piazza della citta medievale",
                    "source": "lego_proxy_reader",
                    "ai_investment_score": 82,
                    "market_demand_score": 76,
                    "current_price": 229.99,
                    "eol_date_prediction": "2026-06-01",
                    "signal_strength": "HIGH_CONFIDENCE",
                    "metadata": {
                        "listing_url": "https://www.lego.com/it-it/product/medieval-town-square-10332",
                        "success_pattern_score": 84,
                        "success_pattern_summary": "Completismo di serie, Continuita' collezione",
                    },
                }
            ],
            "diagnostics": {
                "fallback_used": False,
                "source_raw_counts": {},
                "dedup_candidates": 1,
                "threshold": 60,
                "above_threshold_count": 1,
                "max_ai_score": 82,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }
        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("Apri su LEGO", joined)
        self.assertIn('href="https://www.lego.com/it-it/product/medieval-town-square-10332"', joined)
        self.assertIn("EOL 01/06/2026", joined)
        self.assertIn("Pattern 84/100", joined)
        self.assertIn("Completismo di serie", joined)

    def test_format_eol_date_parses_iso_datetime(self) -> None:
        self.assertEqual(LegoHunterTelegramBot._format_eol_date("2026-06-01T08:15:30Z"), "01/06/2026")
        self.assertEqual(LegoHunterTelegramBot._format_eol_date(""), "n/d")
        self.assertEqual(LegoHunterTelegramBot._format_eol_date("non valida"), "non valida")

    def test_format_discovery_report_adds_lego_search_link_when_listing_missing(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "75367",
                    "set_name": "Venator",
                    "source": "amazon_proxy_reader",
                    "ai_investment_score": 70,
                    "market_demand_score": 61,
                    "current_price": 649.99,
                    "eol_date_prediction": "2026-12-01",
                    "signal_strength": "HIGH_CONFIDENCE",
                }
            ],
            "diagnostics": {
                "fallback_used": False,
                "source_raw_counts": {},
                "dedup_candidates": 1,
                "threshold": 60,
                "above_threshold_count": 1,
                "max_ai_score": 70,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }
        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("Cerca su LEGO", joined)

    def test_format_discovery_report_explains_when_only_low_confidence_above_threshold(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "76281",
                    "set_name": "X-Jet di X-Men",
                    "source": "lego_proxy_reader",
                    "signal_strength": "LOW_CONFIDENCE",
                    "ai_investment_score": 64,
                }
            ],
            "diagnostics": {
                "fallback_used": True,
                "above_threshold_count": 9,
                "above_threshold_high_confidence_count": 0,
                "source_raw_counts": {},
                "dedup_candidates": 41,
                "threshold": 60,
                "max_ai_score": 79,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }

        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)

        self.assertIn("Nessun set <b>HIGH_CONFIDENCE</b>", joined)
        self.assertIn("LOW_CONFIDENCE", joined)
        self.assertIn('href="https://www.lego.com/it-it/search?q=76281"', joined)

    def test_format_discovery_report_bootstrap_high_conf_note_is_html_safe(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "42182",
                    "set_name": "Rover lunare NASA Apollo - LRV",
                    "source": "lego_proxy_reader",
                    "signal_strength": "HIGH_CONFIDENCE",
                    "ai_investment_score": 78,
                    "market_demand_score": 96,
                    "forecast_probability_upside_12m": 64.4,
                    "confidence_score": 56,
                    "metadata": {
                        "forecast_data_points": 12,
                    },
                }
            ],
            "diagnostics": {
                "fallback_used": False,
                "above_threshold_count": 16,
                "above_threshold_high_confidence_count": 3,
                "source_raw_counts": {},
                "dedup_candidates": 41,
                "threshold": 60,
                "max_ai_score": 82,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "bootstrap_thresholds_enabled": True,
                "bootstrap_min_history_points": 45,
                "bootstrap_min_probability_high_confidence": 0.52,
                "bootstrap_min_confidence_score_high_confidence": 50,
                "bootstrap_rows_count": 8,
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }

        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        note_line = next((line for line in lines if line.startswith("Nota: HIGH_CONFIDENCE in bootstrap")), "")
        self.assertTrue(note_line)
        self.assertIn("inferiori a 45", note_line)
        self.assertNotIn("<", note_line)
        self.assertNotIn(">", note_line)

    async def test_scheduled_cycle_continues_when_command_sync_times_out(self) -> None:
        bot_mock = AsyncMock()
        bot_mock.set_my_commands = AsyncMock(side_effect=TimedOut("timed out"))
        bot_mock.send_message = AsyncMock(return_value={"ok": True})
        bot_mock.shutdown = AsyncMock(return_value=None)

        with patch("bot.Bot", return_value=bot_mock), patch("bot.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            await run_scheduled_cycle(
                token="token",
                chat_id="123",
                oracle=FakeOracle(),
                repository=FakeRepo(),
                fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
            )

        self.assertGreaterEqual(bot_mock.set_my_commands.await_count, 3)
        self.assertEqual(bot_mock.send_message.await_count, 1)
        self.assertEqual(bot_mock.shutdown.await_count, 1)
        self.assertGreaterEqual(sleep_mock.await_count, 2)

    async def test_scheduled_cycle_does_not_retry_send_message_on_timeout(self) -> None:
        bot_mock = AsyncMock()
        bot_mock.set_my_commands = AsyncMock(return_value=True)
        bot_mock.send_message = AsyncMock(side_effect=TimedOut("timed out"))
        bot_mock.shutdown = AsyncMock(return_value=None)

        with patch("bot.Bot", return_value=bot_mock), patch("bot.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            await run_scheduled_cycle(
                token="token",
                chat_id="123",
                oracle=FakeOracle(),
                repository=FakeRepo(),
                fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
            )

        self.assertEqual(bot_mock.send_message.await_count, 1)
        self.assertEqual(bot_mock.shutdown.await_count, 1)
        self.assertEqual(sleep_mock.await_count, 0)


if __name__ == "__main__":
    unittest.main()
