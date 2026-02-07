from __future__ import annotations

import unittest

from bot import LegoHunterTelegramBot


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


if __name__ == "__main__":
    unittest.main()
