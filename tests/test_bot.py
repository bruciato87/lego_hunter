from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from telegram.error import BadRequest, TimedOut

from bot import (
    LegoHunterTelegramBot,
    _dispatch_single_set_analysis_workflow,
    _parse_eur_amount,
    _parse_positive_quantity,
    build_application,
    run_scheduled_cycle,
)


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
    def __init__(self) -> None:
        self.last_purchase_payload = None
        self.last_sale_payload = None

    def get_top_opportunities(self, limit=8, min_score=50):  # noqa: ANN001
        return [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "ai_investment_score": 88,
                "market_demand_score": 79,
                "current_price": 129.99,
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

    def get_best_recent_secondary_price(self, set_id, max_age_hours=72.0):  # noqa: ANN001
        return {
            "platform": "vinted",
            "price": 110.0,
            "listing_url": "https://example.com/listing/75367",
        }

    def get_latest_price(self, set_id, platform=None):  # noqa: ANN001
        return {"platform": platform or "vinted", "price": 140.0}

    def resolve_set_identity(self, set_id):  # noqa: ANN001
        if str(set_id) == "76441":
            return {"set_id": "76441", "set_name": "Castello di Hogwarts™: Club dei Duellanti", "theme": "Harry Potter"}
        return {"set_id": str(set_id), "set_name": None, "theme": None}

    def add_portfolio_purchase(self, **kwargs):  # noqa: ANN003
        self.last_purchase_payload = dict(kwargs)
        quantity = int(kwargs.get("quantity") or 1)
        return {
            "set_id": kwargs.get("set_id"),
            "set_name": kwargs.get("set_name"),
            "theme": kwargs.get("theme"),
            "purchase_price": kwargs.get("purchase_price"),
            "quantity": quantity,
            "purchase_platform": kwargs.get("purchase_platform"),
            "status": "holding",
        }

    def register_portfolio_sale(self, **kwargs):  # noqa: ANN003
        self.last_sale_payload = dict(kwargs)
        set_id = str(kwargs.get("set_id") or "").strip()
        if set_id != "76441":
            raise ValueError("Set non presente in collezione.")

        sold_units = int(kwargs.get("quantity") or 1)
        sale_price = float(kwargs.get("sale_price") or 0.0)
        remaining = max(0, 3 - sold_units)
        return {
            "set_id": set_id,
            "set_name": "Castello di Hogwarts™: Club dei Duellanti",
            "sold_units": sold_units,
            "sale_unit_price": sale_price,
            "gross_amount": round(sale_price * sold_units, 2),
            "remaining_quantity": remaining,
            "sold_all": remaining == 0,
            "platform": str(kwargs.get("platform") or "telegram_manual"),
        }


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

    async def discover_with_diagnostics(
        self,
        persist=True,
        top_limit=12,
        fallback_limit=3,
        exclude_owned=True,
    ):  # noqa: ANN001
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
    def test_parse_eur_amount_supports_common_formats(self) -> None:
        self.assertEqual(_parse_eur_amount("34,99"), 34.99)
        self.assertEqual(_parse_eur_amount("34.99"), 34.99)
        self.assertEqual(_parse_eur_amount("1.234,56"), 1234.56)
        self.assertEqual(_parse_eur_amount("1,234.56"), 1234.56)

    def test_parse_positive_quantity_requires_positive_int(self) -> None:
        self.assertEqual(_parse_positive_quantity("2"), 2)
        with self.assertRaises(ValueError):
            _parse_positive_quantity("0")

    def test_dispatch_single_set_analysis_requires_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            ok, detail = _dispatch_single_set_analysis_workflow(set_id="76441", chat_id="123")
        self.assertFalse(ok)
        self.assertIn("Configurazione incompleta", detail)

    def test_dispatch_single_set_analysis_success(self) -> None:
        fake_response = Mock()
        fake_response.status = 204
        fake_cm = Mock()
        fake_cm.__enter__ = Mock(return_value=fake_response)
        fake_cm.__exit__ = Mock(return_value=None)

        with (
            patch.dict(
                os.environ,
                {
                    "GITHUB_ACTIONS_DISPATCH_TOKEN": "token",
                    "GITHUB_REPO": "bruciato87/lego_hunter",
                    "GITHUB_WORKFLOW_FILE": "main.yml",
                    "GITHUB_WORKFLOW_REF": "main",
                },
                clear=False,
            ),
            patch("bot.urllib.request.urlopen", return_value=fake_cm) as urlopen_mock,
        ):
            ok, detail = _dispatch_single_set_analysis_workflow(set_id="76441", chat_id="123")

        self.assertTrue(ok)
        self.assertIn("Analisi approfondita avviata", detail)
        self.assertEqual(urlopen_mock.call_count, 1)

    def test_build_application_can_disable_post_init_command_sync(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )

        app = build_application(manager, "telegram-token", register_commands_on_init=False)

        self.assertIsNone(app.post_init)

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
        self.assertIn("/acquista <set_id> <prezzo>", text)
        self.assertIn("/venduto <set_id> <prezzo_vendita>", text)
        self.assertIn("/analizza <set_id>", text)
        self.assertIn("/offerte", text)
        self.assertIn("/collezione", text)
        self.assertIn("Alias compatibilita'", text)
        self.assertIn("/hunt -> /scova", text)
        self.assertIn("Esempi rapidi", text)

    async def test_help_does_not_initialize_oracle(self) -> None:
        oracle_factory = Mock(return_value=FakeOracle())
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=None,
            oracle_factory=oracle_factory,
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        await manager.cmd_help(update, DummyContext())

        oracle_factory.assert_not_called()

    async def test_scova_initializes_oracle_on_demand(self) -> None:
        oracle_factory = Mock(return_value=FakeOracle())
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=None,
            oracle_factory=oracle_factory,
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        await manager.cmd_scova(update, DummyContext())

        oracle_factory.assert_called_once()

    async def test_analizza_requires_set_id_argument(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        await manager.cmd_analizza(update, DummyContext(args=[]))

        self.assertTrue(update.message.replies)
        self.assertIn("Uso: /analizza <set_id>", update.message.replies[-1])

    async def test_analizza_dispatches_github_workflow(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        with patch(
            "bot._dispatch_single_set_analysis_workflow",
            return_value=(True, "dispatch ok"),
        ) as dispatch_mock:
            await manager.cmd_analizza(update, DummyContext(args=["76441"]))

        dispatch_mock.assert_called_once_with(set_id="76441", chat_id="98765")
        self.assertGreaterEqual(len(update.message.replies), 2)
        self.assertIn("Avvio analisi approfondita", update.message.replies[0])
        self.assertIn("dispatch ok", update.message.replies[-1])

    async def test_analizza_reports_dispatch_configuration_error(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        with patch(
            "bot._dispatch_single_set_analysis_workflow",
            return_value=(False, "missing token"),
        ):
            await manager.cmd_analizza(update, DummyContext(args=["76441"]))

        self.assertTrue(update.message.replies)
        self.assertIn("Impossibile avviare l'analisi su GitHub", update.message.replies[-1])
        self.assertIn("missing token", update.message.replies[-1])

    async def test_acquista_requires_required_args(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_acquista(update, DummyContext(args=[]))

        self.assertTrue(update.message.replies)
        self.assertIn("Uso: /acquista", update.message.replies[-1])

    async def test_acquista_registers_purchase_with_auto_set_name(self) -> None:
        repo = FakeRepo()
        manager = LegoHunterTelegramBot(
            repository=repo,
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_acquista(update, DummyContext(args=["76441", "34,99", "2"]))

        self.assertIsNotNone(repo.last_purchase_payload)
        self.assertEqual(repo.last_purchase_payload.get("set_id"), "76441")
        self.assertEqual(float(repo.last_purchase_payload.get("purchase_price") or 0.0), 34.99)
        self.assertEqual(int(repo.last_purchase_payload.get("quantity") or 0), 2)
        self.assertEqual(repo.last_purchase_payload.get("set_name"), "Castello di Hogwarts™: Club dei Duellanti")
        self.assertTrue(update.message.replies)
        self.assertIn("Acquisto registrato in collezione", update.message.replies[-1])
        self.assertIn("Club dei Duellanti", update.message.replies[-1])
        self.assertIn("Quantita aggiunta: 2", update.message.replies[-1])

    async def test_acquista_uses_set_id_fallback_when_name_not_found(self) -> None:
        repo = FakeRepo()
        manager = LegoHunterTelegramBot(
            repository=repo,
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_acquista(update, DummyContext(args=["99999", "10"]))

        self.assertIsNotNone(repo.last_purchase_payload)
        self.assertEqual(repo.last_purchase_payload.get("set_name"), "LEGO 99999")
        self.assertTrue(update.message.replies)
        self.assertIn("Nome set non trovato nello storico", update.message.replies[-1])

    async def test_venduto_requires_required_args(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_venduto(update, DummyContext(args=[]))

        self.assertTrue(update.message.replies)
        self.assertIn("Uso: /venduto", update.message.replies[-1])

    async def test_venduto_registers_sale_and_removes_position(self) -> None:
        repo = FakeRepo()
        manager = LegoHunterTelegramBot(
            repository=repo,
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_venduto(update, DummyContext(args=["76441", "89,90", "3", "ebay"]))

        self.assertIsNotNone(repo.last_sale_payload)
        self.assertEqual(repo.last_sale_payload.get("set_id"), "76441")
        self.assertEqual(float(repo.last_sale_payload.get("sale_price") or 0.0), 89.9)
        self.assertEqual(int(repo.last_sale_payload.get("quantity") or 0), 3)
        self.assertEqual(repo.last_sale_payload.get("platform"), "ebay")
        self.assertTrue(update.message.replies)
        reply = update.message.replies[-1]
        self.assertIn("Vendita registrata", reply)
        self.assertIn("Posizione rimossa dalla collezione", reply)
        self.assertIn("Fiscal Guard: GREEN", reply)

    async def test_venduto_partial_sale_updates_remaining(self) -> None:
        repo = FakeRepo()
        manager = LegoHunterTelegramBot(
            repository=repo,
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_venduto(update, DummyContext(args=["76441", "89,90", "2"]))

        self.assertIsNotNone(repo.last_sale_payload)
        self.assertEqual(int(repo.last_sale_payload.get("quantity") or 0), 2)
        self.assertTrue(update.message.replies)
        self.assertIn("Residuo in collezione: 1", update.message.replies[-1])

    async def test_venduto_rejects_negative_quantity(self) -> None:
        repo = FakeRepo()
        manager = LegoHunterTelegramBot(
            repository=repo,
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_venduto(update, DummyContext(args=["76441", "89,90", "-1"]))

        self.assertTrue(update.message.replies)
        self.assertIn("Quantita non valida", update.message.replies[-1])
        self.assertIsNone(repo.last_sale_payload)

    async def test_venduto_reports_when_set_not_in_collection(self) -> None:
        repo = FakeRepo()
        manager = LegoHunterTelegramBot(
            repository=repo,
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate(chat_id="98765")

        await manager.cmd_venduto(update, DummyContext(args=["99999", "50"]))

        self.assertTrue(update.message.replies)
        self.assertIn("Vendita non registrata", update.message.replies[-1])

    async def test_offerte_uses_cloud_cache_in_light_runtime_without_oracle(self) -> None:
        oracle_factory = Mock(return_value=FakeOracle())
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=None,
            oracle_factory=oracle_factory,
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        with patch("bot.PLAYWRIGHT_AVAILABLE", False):
            await manager.cmd_offerte(update, DummyContext())

        oracle_factory.assert_not_called()
        self.assertTrue(update.message.replies)
        self.assertIn("cache cloud", update.message.replies[-1].lower())
        self.assertIn("LEGO Star Wars", update.message.replies[-1])

    async def test_light_commands_do_not_initialize_oracle(self) -> None:
        oracle_factory = Mock(return_value=FakeOracle())
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=None,
            oracle_factory=oracle_factory,
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )

        await manager.cmd_start(DummyUpdate(), DummyContext())
        await manager.cmd_help(DummyUpdate(), DummyContext())
        await manager.cmd_acquista(DummyUpdate(), DummyContext(args=["76441", "34,99", "1"]))
        await manager.cmd_venduto(DummyUpdate(), DummyContext(args=["76441", "49,99"]))
        await manager.cmd_radar(DummyUpdate(), DummyContext())
        await manager.cmd_cerca(DummyUpdate(), DummyContext(args=["Rivendell"]))
        await manager.cmd_collezione(DummyUpdate(), DummyContext())
        await manager.cmd_vendi(DummyUpdate(), DummyContext())

        oracle_factory.assert_not_called()

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

    async def test_vendi_prefers_ebay_it_when_best_quote(self) -> None:
        class EbayFirstRepo(FakeRepo):
            def get_latest_price(self, set_id, platform=None):  # noqa: ANN001
                if platform == "ebay":
                    return {"platform": "ebay", "price": 180.0}
                if platform == "vinted":
                    return {"platform": "vinted", "price": 140.0}
                if platform == "subito":
                    return {"platform": "subito", "price": 135.0}
                return None

        manager = LegoHunterTelegramBot(
            repository=EbayFirstRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        await manager.cmd_vendi(update, DummyContext())

        self.assertTrue(update.message.replies)
        text = update.message.replies[-1]
        self.assertIn("Segnali uscita", text)
        self.assertIn("ebay.it", text.lower())

    async def test_scova_runs_silent_sell_scan_and_shows_only_when_profitable(self) -> None:
        manager = LegoHunterTelegramBot(
            repository=FakeRepo(),
            oracle=FakeOracle(),
            fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
        )
        update = DummyUpdate()

        await manager.cmd_scova(update, DummyContext())

        self.assertTrue(update.message.replies)
        text = update.message.replies[-1]
        self.assertIn("Occasioni vendita rilevate", text)
        self.assertIn("LEGO Star Wars", text)

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

    def test_format_discovery_report_splits_top_picks_and_quantitative_radar(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "76281",
                    "set_name": "X-Jet di X-Men",
                    "source": "lego_proxy_reader",
                    "ai_shortlisted": True,
                    "signal_strength": "LOW_CONFIDENCE",
                    "ai_investment_score": 72,
                    "market_demand_score": 95,
                    "forecast_score": 59,
                    "forecast_probability_upside_12m": 66.2,
                    "confidence_score": 58,
                    "eol_date_prediction": "2026-05-16",
                    "risk_note": "Confidenza dati sotto soglia.",
                },
                {
                    "set_id": "71486",
                    "set_name": "Castello di Nocturnia",
                    "source": "lego_proxy_reader",
                    "ai_shortlisted": False,
                    "signal_strength": "LOW_CONFIDENCE",
                    "ai_investment_score": 67,
                    "market_demand_score": 95,
                    "forecast_score": 61,
                    "forecast_probability_upside_12m": 69.9,
                    "confidence_score": 53,
                    "eol_date_prediction": "2026-05-30",
                    "risk_note": "AI non eseguita: pre-filter rank #6 oltre top 3.",
                },
            ],
            "diagnostics": {
                "fallback_used": True,
                "above_threshold_count": 14,
                "above_threshold_high_confidence_count": 0,
                "source_raw_counts": {},
                "dedup_candidates": 41,
                "threshold": 60,
                "max_ai_score": 82,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }

        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("<b>Top Picks</b>", joined)
        self.assertIn("<b>Radar quantitativo (AI non eseguita nel ciclo)</b>", joined)
        top_section = joined.split("<b>Top Picks</b>", 1)[1].split("<b>Radar quantitativo (AI non eseguita nel ciclo)</b>", 1)[0]
        self.assertIn("X-Jet di X-Men", top_section)
        self.assertNotIn("Castello di Nocturnia", top_section)
        radar_section = joined.split("<b>Radar quantitativo (AI non eseguita nel ciclo)</b>", 1)[1]
        self.assertIn("Castello di Nocturnia", radar_section)
        self.assertIn("AI non eseguita: pre-filter rank #6 oltre top 3.", radar_section)

    def test_format_discovery_report_only_quantitative_radar_when_no_ai_shortlisted(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "71486",
                    "set_name": "Castello di Nocturnia",
                    "source": "lego_proxy_reader",
                    "ai_shortlisted": False,
                    "signal_strength": "LOW_CONFIDENCE",
                    "ai_investment_score": 67,
                    "market_demand_score": 95,
                    "forecast_score": 61,
                    "forecast_probability_upside_12m": 69.9,
                    "confidence_score": 53,
                    "eol_date_prediction": "2026-05-30",
                    "risk_note": "AI non eseguita: pre-filter rank #6 oltre top 3.",
                }
            ],
            "diagnostics": {
                "fallback_used": True,
                "above_threshold_count": 14,
                "above_threshold_high_confidence_count": 0,
                "source_raw_counts": {},
                "dedup_candidates": 41,
                "threshold": 60,
                "max_ai_score": 82,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }

        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("🟠 Nessun Top Pick AI nel ciclo corrente.", joined)
        self.assertIn("<b>Radar quantitativo (AI non eseguita nel ciclo)</b>", joined)
        self.assertIn("Castello di Nocturnia", joined)

    def test_format_discovery_report_includes_historical_metrics_and_gate_thresholds(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "76281",
                    "set_name": "X-Jet di X-Men",
                    "source": "lego_proxy_reader",
                    "signal_strength": "LOW_CONFIDENCE",
                    "ai_investment_score": 72,
                    "market_demand_score": 95,
                    "forecast_probability_upside_12m": 66.2,
                    "confidence_score": 58,
                    "current_price": 79.99,
                    "eol_date_prediction": "2026-05-16",
                    "historical_sample_size": 33,
                    "historical_win_rate_12m_pct": 62.4,
                    "historical_prior_score": 71,
                    "historical_support_confidence": 59,
                }
            ],
            "diagnostics": {
                "fallback_used": True,
                "above_threshold_count": 9,
                "above_threshold_high_confidence_count": 0,
                "source_raw_counts": {},
                "dedup_candidates": 41,
                "threshold": 60,
                "max_ai_score": 86,
                "max_composite_score": 72,
                "max_probability_upside_12m": 66.2,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "historical_high_conf_required": True,
                "historical_high_conf_effective_min_samples": 18,
                "historical_high_conf_effective_min_win_rate_pct": 50.0,
                "historical_high_conf_effective_min_support_confidence": 50,
                "historical_high_conf_effective_min_prior_score": 60,
                "adaptive_historical_thresholds_active": True,
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }

        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("Storico: 33 campioni | Win-rate 12m 62.4% | Prior 71/100 | Supporto 59/100", joined)
        self.assertIn("Gate storico (adattive): campioni>=18", joined)

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

    def test_format_discovery_report_marks_bootstrap_as_preliminary_and_shows_guardrail_diagnostics(self) -> None:
        report = {
            "selected": [
                {
                    "set_id": "76281",
                    "set_name": "X-Jet di X-Men",
                    "source": "lego_proxy_reader",
                    "signal_strength": "HIGH_CONFIDENCE_BOOTSTRAP",
                    "ai_investment_score": 72,
                    "ai_raw_score": 78,
                    "market_demand_score": 95,
                    "forecast_probability_upside_12m": 66.2,
                    "confidence_score": 58,
                    "metadata": {
                        "forecast_data_points": 40,
                    },
                }
            ],
            "diagnostics": {
                "fallback_used": False,
                "above_threshold_count": 12,
                "above_threshold_high_confidence_count": 1,
                "source_raw_counts": {},
                "dedup_candidates": 41,
                "threshold": 60,
                "max_ai_score": 78,
                "max_ai_model_raw_score": 99,
                "ai_guardrail_applied_count": 2,
                "strict_pass_rate_shortlist": 0.14,
                "non_json_rate_shortlist": 0.21,
                "fallback_rate_shortlist": 0.33,
                "non_json_rate_total": 0.05,
                "fallback_rate_total": 0.71,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "bootstrap_thresholds_enabled": True,
                "bootstrap_min_history_points": 45,
                "bootstrap_min_probability_high_confidence": 0.56,
                "bootstrap_min_confidence_score_high_confidence": 55,
                "bootstrap_rows_count": 8,
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }

        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("solo <b>HIGH_CONFIDENCE_BOOTSTRAP</b>", joined)
        self.assertIn("Segnale: HIGH_CONFIDENCE_BOOTSTRAP (preliminare)", joined)
        self.assertIn("🧪 Qualita AI (shortlist): strict-pass 14% | non-JSON 21% | fallback 33%", joined)
        self.assertIn("📊 Copertura ranking totale: non-JSON 5% | fallback 71%", joined)
        self.assertIn("🛡️ AI guardrail: 2 score normalizzati | Max AI raw 99", joined)

    def test_format_discovery_report_no_signal_line_is_html_safe(self) -> None:
        report = {
            "selected": [],
            "diagnostics": {
                "fallback_used": True,
                "no_signal_due_to_low_strict_pass": True,
                "no_signal_strict_pass_rate_shortlist": 0.2,
                "no_signal_trust_pass_rate_shortlist": 0.3,
                "no_signal_strict_pass_min_rate": 0.5,
                "source_raw_counts": {},
                "dedup_candidates": 0,
                "threshold": 60,
                "above_threshold_count": 0,
                "max_ai_score": 0,
                "source_strategy": "external_first",
                "selected_source": "external_proxy",
                "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
            },
        }

        lines = LegoHunterTelegramBot._format_discovery_report(report, top_limit=3)
        joined = "\n".join(lines)
        self.assertIn("affidabilità shortlist 30% &lt; 50%", joined)
        self.assertIn("strict-pass 20%", joined)
        self.assertNotIn("affidabilità shortlist 30% < 50%", joined)

    async def test_scheduled_cycle_continues_when_command_sync_times_out(self) -> None:
        bot_mock = AsyncMock()
        bot_mock.set_my_commands = AsyncMock(side_effect=TimedOut("timed out"))
        bot_mock.send_message = AsyncMock(return_value={"ok": True})
        bot_mock.shutdown = AsyncMock(return_value=None)

        with (
            patch("bot.Bot", return_value=bot_mock),
            patch("bot.PLAYWRIGHT_AVAILABLE", False),
            patch("bot.asyncio.sleep", new=AsyncMock()) as sleep_mock,
        ):
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

        with (
            patch("bot.Bot", return_value=bot_mock),
            patch("bot.PLAYWRIGHT_AVAILABLE", False),
            patch("bot.asyncio.sleep", new=AsyncMock()) as sleep_mock,
        ):
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

    async def test_scheduled_cycle_falls_back_to_plain_text_on_html_parse_error(self) -> None:
        bot_mock = AsyncMock()
        bot_mock.set_my_commands = AsyncMock(return_value=True)
        bot_mock.send_message = AsyncMock(
            side_effect=[
                BadRequest('Can\'t parse entities: unsupported start tag "" at byte offset 158'),
                {"ok": True},
            ]
        )
        bot_mock.shutdown = AsyncMock(return_value=None)

        with (
            patch("bot.Bot", return_value=bot_mock),
            patch("bot.PLAYWRIGHT_AVAILABLE", False),
            patch("bot.asyncio.sleep", new=AsyncMock()),
        ):
            await run_scheduled_cycle(
                token="token",
                chat_id="123",
                oracle=FakeOracle(),
                repository=FakeRepo(),
                fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
            )

        self.assertEqual(bot_mock.send_message.await_count, 2)
        first_kwargs = bot_mock.send_message.await_args_list[0].kwargs
        second_kwargs = bot_mock.send_message.await_args_list[1].kwargs
        self.assertEqual(first_kwargs.get("parse_mode"), "HTML")
        self.assertNotIn("parse_mode", second_kwargs)
        self.assertIn("LEGO HUNTER", str(second_kwargs.get("text") or ""))

    async def test_scheduled_cycle_can_send_single_set_analysis_report(self) -> None:
        class SingleSetOracle(FakeOracle):
            async def discover_with_diagnostics(
                self,
                persist=True,
                top_limit=12,
                fallback_limit=3,
                exclude_owned=True,
            ):  # noqa: ANN001
                return {
                    "selected": [],
                    "above_threshold": [],
                    "ranked": [
                        {
                            "set_id": "76441",
                            "set_name": "Castello di Hogwarts",
                            "source": "lego_proxy_reader",
                            "ai_investment_score": 82,
                            "ai_strict_pass": True,
                            "market_demand_score": 90,
                            "forecast_score": 61,
                            "forecast_probability_upside_12m": 64.1,
                            "confidence_score": 61,
                            "composite_score": 72,
                            "current_price": 24.99,
                            "eol_date_prediction": "2026-05-11",
                        }
                    ],
                    "diagnostics": {
                        "threshold": 60,
                        "min_probability_high_confidence": 0.60,
                        "min_confidence_score_high_confidence": 68,
                        "source_raw_counts": {},
                        "dedup_candidates": 1,
                        "above_threshold_count": 1,
                        "max_ai_score": 82,
                        "max_composite_score": 72,
                        "max_probability_upside_12m": 64.1,
                        "fallback_used": True,
                        "source_strategy": "external_first",
                        "selected_source": "external_proxy",
                        "ai_runtime": {"engine": "openrouter", "model": "x/y:free", "mode": "api"},
                    },
                }

            def _high_confidence_signal_strength(self, row):  # noqa: ANN001
                return "LOW_CONFIDENCE"

            def _build_low_confidence_note(self, row):  # noqa: ANN001
                return "nota test"

        bot_mock = AsyncMock()
        bot_mock.set_my_commands = AsyncMock(return_value=True)
        bot_mock.send_message = AsyncMock(return_value={"ok": True})
        bot_mock.shutdown = AsyncMock(return_value=None)

        with (
            patch("bot.Bot", return_value=bot_mock),
            patch("bot.PLAYWRIGHT_AVAILABLE", False),
            patch.dict("os.environ", {"SINGLE_SET_ANALYSIS_ID": "76441"}, clear=False),
        ):
            await run_scheduled_cycle(
                token="token",
                chat_id="123",
                oracle=SingleSetOracle(),
                repository=FakeRepo(),
                fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
            )

        self.assertEqual(bot_mock.send_message.await_count, 1)
        text = str(bot_mock.send_message.await_args.kwargs.get("text") or "")
        self.assertIn("Analisi approfondita singolo set", text)
        self.assertIn("Set richiesto: <b>76441</b>", text)
        self.assertIn("Set richiesto trovato ma <b>LOW_CONFIDENCE</b>", text)
        self.assertIn("Castello di Hogwarts", text)

    async def test_scheduled_cycle_writes_diagnostic_pack_when_enabled(self) -> None:
        bot_mock = AsyncMock()
        bot_mock.set_my_commands = AsyncMock(return_value=True)
        bot_mock.send_message = AsyncMock(return_value={"ok": True})
        bot_mock.shutdown = AsyncMock(return_value=None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch("bot.Bot", return_value=bot_mock),
                patch("bot.PLAYWRIGHT_AVAILABLE", False),
                patch.dict("os.environ", {"LH_DIAGNOSTIC_DIR": tmp_dir}, clear=False),
            ):
                await run_scheduled_cycle(
                    token="token",
                    chat_id="123",
                    oracle=FakeOracle(),
                    repository=FakeRepo(),
                    fiscal_guardian=FakeFiscal({"allow_sell_signals": True, "status": "GREEN", "message": "ok"}),
                )

            snapshot_path = os.path.join(tmp_dir, "discovery_snapshot.json")
            preview_html_path = os.path.join(tmp_dir, "telegram_message_preview.html")
            preview_txt_path = os.path.join(tmp_dir, "telegram_message_preview.txt")
            kpi_path = os.path.join(tmp_dir, "kpi_summary.txt")

            self.assertTrue(os.path.exists(snapshot_path))
            self.assertTrue(os.path.exists(preview_html_path))
            self.assertTrue(os.path.exists(preview_txt_path))
            self.assertTrue(os.path.exists(kpi_path))


if __name__ == "__main__":
    unittest.main()
