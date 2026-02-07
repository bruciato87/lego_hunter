from __future__ import annotations

import asyncio
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from oracle import AIInsight, DiscoveryOracle, PatternEvaluation
from scrapers import MarketListing


class FakeRepo:
    def __init__(self) -> None:
        self.upserted = []
        self.snapshots = []
        self.recent = {}
        self.theme_baselines = {}
        self.recent_ai_insights = {}

    def upsert_opportunity(self, record):  # noqa: ANN001
        self.upserted.append(record)
        return {"ok": True}

    def insert_market_snapshot(self, record):  # noqa: ANN001
        self.snapshots.append(record)
        return {"ok": True}

    def get_recent_market_prices(self, set_id, days=30, platform=None):  # noqa: ANN001
        return self.recent.get(set_id, [])

    def get_market_history_for_sets(self, set_ids, days=540):  # noqa: ANN001
        rows = []
        for set_id in set_ids:
            for row in self.recent.get(set_id, []):
                enriched = dict(row)
                enriched.setdefault("set_id", set_id)
                rows.append(enriched)
        return rows

    def get_theme_radar_baseline(self, theme, days=180, limit=120):  # noqa: ANN001
        return self.theme_baselines.get(
            theme,
            {
                "sample_size": 0.0,
                "avg_ai_score": 0.0,
                "avg_market_demand": 0.0,
                "std_ai_score": 0.0,
            },
        )

    def get_recent_ai_insights(self, set_ids, max_age_hours=36.0):  # noqa: ANN001
        _ = max_age_hours
        return {
            str(set_id): self.recent_ai_insights[str(set_id)]
            for set_id in set_ids
            if str(set_id) in self.recent_ai_insights
        }


class DummyOracle(DiscoveryOracle):
    def __init__(self, repository, candidates):  # noqa: ANN001
        super().__init__(repository, gemini_api_key=None, min_ai_score=60)
        self._candidates = candidates
        self.ai_calls = 0

    async def _collect_source_candidates(self):
        return self._candidates

    async def _get_ai_insight(self, candidate):  # noqa: ANN001
        self.ai_calls += 1
        mock_fallback = bool(candidate.get("mock_fallback", False))
        return AIInsight(
            score=int(candidate.get("mock_score", 50)),
            summary="mock summary",
            predicted_eol_date=candidate.get("eol_date_prediction"),
            fallback_used=mock_fallback,
            confidence="LOW_CONFIDENCE" if mock_fallback else "HIGH_CONFIDENCE",
            risk_note="mock fallback risk" if mock_fallback else None,
        )


class OracleTests(unittest.IsolatedAsyncioTestCase):
    def test_validate_ai_payload_requires_json_contract(self) -> None:
        valid = DiscoveryOracle._validate_ai_payload(
            {
                "score": 73,
                "summary": "domanda buona",
                "predicted_eol_date": "2026-09-01",
            },
            candidate=None,
        )
        self.assertEqual(valid["score"], 73)

        with self.assertRaises(ValueError):
            DiscoveryOracle._validate_ai_payload({"summary": "x"}, candidate=None)
        with self.assertRaises(ValueError):
            DiscoveryOracle._validate_ai_payload({"score": 101, "summary": "x"}, candidate=None)
        with self.assertRaises(ValueError):
            DiscoveryOracle._validate_ai_payload(
                {"score": 70, "summary": "", "predicted_eol_date": "bad-date"},
                candidate=None,
            )

    def test_success_patterns_rule_exclusive_cult_license(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        candidate = {
            "set_id": "76917",
            "set_name": "LEGO Fast & Furious Skyline con minifigure esclusiva Paul Walker",
            "theme": "Speed Champions",
            "source": "lego_proxy_reader",
            "current_price": 24.99,
            "eol_date_prediction": "2026-04-30",
        }

        pattern = oracle._evaluate_success_patterns(candidate)
        codes = {str(row.get("code")) for row in pattern.signals}
        self.assertIn("exclusive_cult_license", codes)
        self.assertGreaterEqual(pattern.score, 90)

    def test_success_patterns_rule_series_completism(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        candidate = {
            "set_id": "75343",
            "set_name": "Casco Dark Trooper - Star Wars Helmet Collection",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 69.99,
            "eol_date_prediction": "2026-05-30",
        }

        pattern = oracle._evaluate_success_patterns(candidate)
        codes = {str(row.get("code")) for row in pattern.signals}
        self.assertIn("series_completism", codes)
        self.assertGreaterEqual(pattern.score, 82)

    def test_success_patterns_rule_adult_display_value(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        candidate = {
            "set_id": "31215",
            "set_name": "LEGO Art Van Gogh 18+ display model for adults",
            "theme": "Art",
            "source": "lego_proxy_reader",
            "current_price": 79.99,
            "eol_date_prediction": "2026-07-15",
        }

        pattern = oracle._evaluate_success_patterns(candidate)
        codes = {str(row.get("code")) for row in pattern.signals}
        self.assertIn("adult_display_value", codes)
        self.assertGreaterEqual(pattern.score, 88)

    def test_composite_score_increases_with_pattern_score(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        low_pattern = oracle._calculate_composite_score(
            ai_score=78,
            demand_score=74,
            forecast_score=69,
            pattern_score=45,
            ai_fallback_used=False,
        )
        high_pattern = oracle._calculate_composite_score(
            ai_score=78,
            demand_score=74,
            forecast_score=69,
            pattern_score=92,
            ai_fallback_used=False,
        )
        self.assertGreater(high_pattern, low_pattern)

    def test_effective_pattern_score_penalizes_fallback_retiring_only(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        pattern_eval = PatternEvaluation(
            score=83,
            confidence_score=74,
            summary="Catalizzatore EOL",
            signals=[{"code": "retiring_window", "label": "Catalizzatore EOL", "score": 83, "confidence": 0.74}],
            features={},
        )
        effective = oracle._effective_pattern_score(pattern_eval=pattern_eval, ai_fallback_used=True)
        self.assertLess(effective, 83)
        self.assertEqual(effective, 60)

    def test_effective_ai_shortlist_limit_scales_with_openrouter_inventory(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_rank_max_candidates = 10
        oracle.ai_dynamic_shortlist_enabled = True
        oracle.ai_dynamic_shortlist_floor = 4
        oracle.ai_dynamic_shortlist_per_model = 2
        oracle.ai_dynamic_shortlist_bonus = 1
        oracle.ai_runtime["engine"] = "openrouter"
        oracle.ai_runtime["inventory_available"] = 2

        limit = oracle._effective_ai_shortlist_limit(41)
        self.assertEqual(limit, 5)

    def test_effective_ai_shortlist_limit_keeps_floor_with_single_inventory(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_rank_max_candidates = 10
        oracle.ai_dynamic_shortlist_enabled = True
        oracle.ai_dynamic_shortlist_floor = 4
        oracle.ai_dynamic_shortlist_multi_model_floor = 5
        oracle.ai_dynamic_shortlist_per_model = 2
        oracle.ai_dynamic_shortlist_bonus = 1
        oracle.ai_runtime["engine"] = "openrouter"
        oracle.ai_runtime["inventory_available"] = 1

        limit = oracle._effective_ai_shortlist_limit(41)
        self.assertEqual(limit, 4)

    def test_success_patterns_summary_uses_top_two_signals(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        candidate = {
            "set_id": "76281",
            "set_name": "X-Jet di X-Men",
            "theme": "Marvel",
            "source": "lego_proxy_reader",
            "current_price": 74.99,
            "eol_date_prediction": "2026-05-16",
        }

        pattern = oracle._evaluate_success_patterns(candidate)
        self.assertGreaterEqual(len(pattern.signals), 2)
        self.assertIn(" + ", pattern.summary)

    def test_rank_candidate_models_skips_temporarily_banned(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)

        oracle._record_model_failure("openrouter", "m1", "timeout 524", phase="probe")
        ranked = oracle._rank_candidate_models("openrouter", ["m1", "m2"], allow_forced_retry=False)
        self.assertEqual(ranked, ["m2"])

    def test_record_model_failure_sets_temporary_ban(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_model_ban_sec = 60.0
        oracle.ai_model_ban_failures = 2

        banned = oracle._record_model_failure("gemini", "models/gemini-2.0-flash", "timeout", phase="scoring")
        self.assertTrue(banned)
        self.assertTrue(oracle._is_model_temporarily_banned("gemini", "models/gemini-2.0-flash"))

    def test_openrouter_strict_probe_enables_best_effort_activation(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_gemini_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key="fake", openrouter_api_key="fake-or-key")

        oracle.strict_ai_probe_validation = True
        oracle._openrouter_inventory_loaded = False
        oracle._openrouter_model_id = None
        oracle._openrouter_candidates = []
        oracle._openrouter_available_candidates = []
        oracle._openrouter_probe_report = []

        with patch.object(
            oracle,
            "_fetch_openrouter_model_payloads",
            return_value=[
                {
                    "id": "vendor/model-a:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text->text"},
                    "context_length": 32000,
                }
            ],
        ), patch.object(
            oracle,
            "_probe_all_openrouter_candidates",
            return_value=[
                {
                    "model": "vendor/model-a:free",
                    "available": False,
                    "status": "not_probed_budget_exhausted",
                    "reason": "budget",
                }
            ],
        ):
            oracle._initialize_openrouter_runtime()

        self.assertEqual(oracle._openrouter_model_id, "vendor/model-a:free")
        self.assertEqual(oracle.ai_runtime.get("mode"), "api_openrouter_best_effort")

    async def test_discover_opportunities_filters_by_min_score(self) -> None:
        repo = FakeRepo()
        now = datetime.now(timezone.utc)
        repo.recent["75367"] = [
            {
                "price": 110 + idx,
                "platform": "vinted" if idx % 2 == 0 else "subito",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(10)
        ]
        repo.theme_baselines["Star Wars"] = {
            "sample_size": 32.0,
            "avg_ai_score": 76.0,
            "avg_market_demand": 82.0,
            "std_ai_score": 6.0,
        }

        candidates = [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "theme": "Star Wars",
                "source": "lego_retiring",
                "current_price": 129.99,
                "eol_date_prediction": "2026-05-01",
                "metadata": {},
                "mock_score": 82,
            },
            {
                "set_id": "60316",
                "set_name": "LEGO City",
                "theme": "City",
                "source": "amazon_bestsellers",
                "current_price": None,
                "eol_date_prediction": None,
                "metadata": {},
                "mock_score": 45,
            },
        ]

        oracle = DummyOracle(repo, candidates)
        rows = await oracle.discover_opportunities(persist=True, top_limit=10)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["set_id"], "75367")
        self.assertEqual(len(repo.upserted), 2)
        self.assertEqual(len(repo.snapshots), 1)

    async def test_validate_secondary_deals_computes_discount(self) -> None:
        repo = FakeRepo()
        oracle = DummyOracle(repo, [])

        opportunities = [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "current_price": 150.0,
                "ai_investment_score": 88,
            }
        ]

        mocked = {
            "75367": [
                MarketListing(
                    platform="vinted",
                    set_id="75367",
                    set_name="LEGO Star Wars",
                    price=120.0,
                    listing_url="https://vinted.example/item",
                    condition="new",
                )
            ]
        }

        with patch(
            "oracle.SecondaryMarketValidator.compare_secondary_prices",
            new=AsyncMock(return_value=mocked),
        ):
            rows = await oracle.validate_secondary_deals(opportunities)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["secondary_platform"], "vinted")
        self.assertAlmostEqual(rows[0]["discount_vs_primary_pct"], 20.0, places=2)
        self.assertEqual(len(repo.snapshots), 1)

    async def test_ranking_payload_contains_composite_and_forecast_metrics(self) -> None:
        repo = FakeRepo()
        now = datetime.now(timezone.utc)
        repo.recent["75367"] = [
            {
                "price": 120 + (idx % 4),
                "platform": "vinted" if idx % 2 == 0 else "subito",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(25)
        ]
        repo.theme_baselines["Star Wars"] = {
            "sample_size": 28.0,
            "avg_ai_score": 74.0,
            "avg_market_demand": 80.0,
            "std_ai_score": 7.0,
        }

        candidates = [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "theme": "Star Wars",
                "source": "lego_proxy_reader",
                "current_price": 129.99,
                "eol_date_prediction": "2026-05-01",
                "metadata": {},
                "mock_score": 82,
            }
        ]
        oracle = DummyOracle(repo, candidates)

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)
        row = report["ranked"][0]
        diagnostics = report["diagnostics"]

        self.assertIn("composite_score", row)
        self.assertIn("forecast_score", row)
        self.assertIn("forecast_probability_upside_12m", row)
        self.assertIn("confidence_score", row)
        self.assertIn("expected_roi_12m_pct", row)
        self.assertGreaterEqual(int(row["composite_score"]), 1)
        self.assertLessEqual(int(row["composite_score"]), 100)
        self.assertIn("threshold_profile", diagnostics)
        self.assertIn("backtest_runtime", diagnostics)

    async def test_non_fallback_can_be_low_confidence_with_weak_quant_data(self) -> None:
        repo = FakeRepo()
        candidates = [
            {
                "set_id": "60316",
                "set_name": "LEGO City",
                "theme": "City",
                "source": "amazon_bestsellers",
                "current_price": 35.0,
                "eol_date_prediction": None,
                "metadata": {},
                "mock_score": 75,
                "mock_fallback": False,
            }
        ]
        oracle = DummyOracle(repo, candidates)

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)
        selected = report["selected"]

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["signal_strength"], "LOW_CONFIDENCE")
        self.assertFalse(bool(selected[0].get("ai_fallback_used")))

    async def test_non_fallback_low_confidence_explains_quant_reason(self) -> None:
        repo = FakeRepo()
        candidates = [
            {
                "set_id": "13000",
                "set_name": "Gru cingolata Liebherr LR 13000",
                "theme": "Technic",
                "source": "lego_proxy_reader",
                "current_price": 129.99,
                "eol_date_prediction": "2026-08-01",
                "metadata": {},
                "mock_score": 95,
                "mock_fallback": False,
            }
        ]
        oracle = DummyOracle(repo, candidates)

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)
        selected = report["selected"]
        note = str(selected[0].get("risk_note") or "")

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["signal_strength"], "LOW_CONFIDENCE")
        self.assertNotIn("Score AI non affidabile", note)
        self.assertTrue(("Probabilita" in note) or ("Confidenza" in note))

    async def test_discovery_excludes_fallback_scores_from_high_confidence_picks(self) -> None:
        repo = FakeRepo()
        now = datetime.now(timezone.utc)
        repo.recent["10332"] = [
            {
                "price": 210.0 + (idx % 5),
                "platform": "vinted" if idx % 2 == 0 else "subito",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(35)
        ]
        repo.recent["40747"] = [
            {
                "price": 16.5 + ((idx % 3) * 0.2),
                "platform": "vinted",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(28)
        ]
        repo.theme_baselines["Icons"] = {
            "sample_size": 40.0,
            "avg_ai_score": 78.0,
            "avg_market_demand": 84.0,
            "std_ai_score": 5.0,
        }
        candidates = [
            {
                "set_id": "40747",
                "set_name": "Narcisi",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 14.99,
                "eol_date_prediction": "2026-04-23",
                "metadata": {},
                "mock_score": 73,
                "mock_fallback": True,
            },
            {
                "set_id": "10332",
                "set_name": "Piazza Medievale",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 229.99,
                "eol_date_prediction": "2026-06-01",
                "metadata": {},
                "mock_score": 72,
                "mock_fallback": False,
            },
        ]
        oracle = DummyOracle(repo, candidates)

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)
        selected = report["selected"]
        diagnostics = report["diagnostics"]

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["set_id"], "10332")
        self.assertEqual(selected[0]["signal_strength"], "HIGH_CONFIDENCE")
        self.assertEqual(diagnostics["above_threshold_count"], 2)
        self.assertEqual(diagnostics["above_threshold_high_confidence_count"], 1)
        self.assertEqual(diagnostics["above_threshold_low_confidence_count"], 1)
        self.assertGreaterEqual(float(selected[0].get("forecast_probability_upside_12m") or 0.0), 60.0)
        self.assertGreaterEqual(int(selected[0].get("confidence_score") or 0), 68)

    async def test_discovery_marks_only_fallback_scores_as_low_confidence(self) -> None:
        repo = FakeRepo()
        now = datetime.now(timezone.utc)
        repo.recent["40747"] = [
            {
                "price": 14.0 + (idx % 3) * 0.5,
                "platform": "vinted",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(12)
        ]
        candidates = [
            {
                "set_id": "40747",
                "set_name": "Narcisi",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 14.99,
                "eol_date_prediction": "2026-04-23",
                "metadata": {},
                "mock_score": 90,
                "mock_fallback": True,
            }
        ]
        oracle = DummyOracle(repo, candidates)

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)
        selected = report["selected"]
        diagnostics = report["diagnostics"]

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["signal_strength"], "LOW_CONFIDENCE")
        self.assertIn("fallback", str(selected[0].get("risk_note", "")).lower())
        self.assertTrue(diagnostics["fallback_used"])
        self.assertEqual(diagnostics["above_threshold_count"], 1)
        self.assertEqual(diagnostics["above_threshold_high_confidence_count"], 0)
        self.assertEqual(diagnostics["above_threshold_low_confidence_count"], 1)

    async def test_source_collection_prefers_external_proxy(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None)
        external_rows = [
            {
                "set_id": "10332",
                "set_name": "Piazza della citta medievale",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 229.99,
                "eol_date_prediction": "2026-05-01",
                "listing_url": "https://www.lego.com/it-it/product/medieval-town-square-10332",
            }
        ]

        with patch.object(
            oracle,
            "_collect_external_proxy_candidates",
            return_value=(
                external_rows,
                {
                    "source_raw_counts": {"lego_proxy_reader": 1, "amazon_proxy_reader": 0},
                    "errors": [],
                    "signals": {"lego_proxy_status_ok": True},
                },
            ),
        ), patch.object(
            oracle,
            "_collect_playwright_candidates",
            new=AsyncMock(
                return_value=(
                    [],
                    {
                        "source_raw_counts": {"lego_retiring": 0, "amazon_bestsellers": 0},
                        "errors": [],
                        "signals": {},
                    },
                )
            ),
        ), patch.object(
            oracle,
            "_collect_http_fallback_candidates",
            return_value=(
                [],
                {
                    "source_raw_counts": {"lego_http_fallback": 0, "amazon_http_fallback": 0},
                    "errors": [],
                    "signals": {},
                },
            ),
        ):
            rows, diagnostics = await oracle._collect_source_candidates_with_diagnostics()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "lego_proxy_reader")
        self.assertEqual(diagnostics.get("selected_source"), "external_proxy")
        self.assertEqual(diagnostics.get("source_strategy"), "external_first")
        self.assertFalse(diagnostics.get("fallback_source_used"))

    async def test_source_collection_falls_back_to_playwright(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None)
        playwright_rows = [
            {
                "set_id": "75367",
                "set_name": "Venator-Class Republic Attack Cruiser",
                "theme": "Star Wars",
                "source": "lego_retiring",
                "current_price": 649.99,
                "eol_date_prediction": "2026-12-01",
                "listing_url": "https://www.lego.com/it-it/product/venator-class-republic-attack-cruiser-75367",
            }
        ]

        with patch.object(
            oracle,
            "_collect_external_proxy_candidates",
            return_value=(
                [],
                {
                    "source_raw_counts": {"lego_proxy_reader": 0, "amazon_proxy_reader": 0},
                    "errors": [],
                    "signals": {"lego_proxy_blocked": True},
                },
            ),
        ), patch.object(
            oracle,
            "_collect_playwright_candidates",
            new=AsyncMock(
                return_value=(
                    playwright_rows,
                    {
                        "source_raw_counts": {"lego_retiring": 1, "amazon_bestsellers": 0},
                        "errors": [],
                        "signals": {},
                    },
                )
            ),
        ), patch.object(
            oracle,
            "_collect_http_fallback_candidates",
            return_value=(
                [],
                {
                    "source_raw_counts": {"lego_http_fallback": 0, "amazon_http_fallback": 0},
                    "errors": [],
                    "signals": {},
                },
            ),
        ):
            rows, diagnostics = await oracle._collect_source_candidates_with_diagnostics()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "lego_retiring")
        self.assertTrue(diagnostics.get("fallback_source_used"))
        self.assertEqual(diagnostics.get("selected_source"), "playwright")

    def test_parse_lego_proxy_markdown(self) -> None:
        markdown = """
### [Piazza della citta medievale](https://www.lego.com/it-it/product/medieval-town-square-10332)

229,99€

### [Castello di Hogwarts](https://www.lego.com/it-it/product/hogwarts-castle-dueling-club-76441)

24,99€
"""
        rows = DiscoveryOracle._parse_lego_proxy_markdown(markdown, limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["set_id"], "10332")
        self.assertEqual(rows[1]["set_id"], "76441")
        self.assertEqual(rows[0]["current_price"], 229.99)
        self.assertNotEqual(rows[0]["eol_date_prediction"], rows[1]["eol_date_prediction"])

    def test_parse_amazon_proxy_markdown_from_context_links(self) -> None:
        markdown = """
[![Image 4: LEGO Super Mario Game Boy Building Set for Adults - 72046](https://m.media-amazon.com/images/I/81rGqE218zL._AC_UL320_.jpg)](https://www.amazon.it/-/en/LEGO-Super-Mario-Building-Adults/dp/B0DWDGVHM6/ref=sr_1_1)

LEGO
----
[Super Mario Game Boy Building Set for Adults - Nintendo Display Model - 72046 -----------------------------------------------------------------------]
4.6[_4.6 out of 5 stars_](javascript:void(0))[(2.1K)](https://www.amazon.it/-/en/LEGO-Super-Mario-Building-Adults/dp/B0DWDGVHM6/ref=sr_1_1?dib=abc)
Price, product page[€47,51€47,51](https://www.amazon.it/-/en/LEGO-Super-Mario-Building-Adults/dp/B0DWDGVHM6/ref=sr_1_1_so_TOY_BUILDING_BLOCK)
"""
        rows = DiscoveryOracle._parse_amazon_proxy_markdown(markdown, limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["set_id"], "72046")
        self.assertEqual(rows[0]["listing_url"], "https://www.amazon.it/dp/B0DWDGVHM6")
        self.assertAlmostEqual(rows[0]["current_price"], 47.51, places=2)

    def test_estimate_market_demand_is_not_flat_or_always_100(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        forecast = oracle.forecaster.forecast(
            candidate={
                "set_id": "10332",
                "set_name": "Piazza della citta medievale",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 229.99,
                "eol_date_prediction": "2026-06-01",
            },
            history_rows=[],
            theme_baseline={},
        )
        recent_rows = [
            {
                "price": 220.0,
                "platform": "vinted",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

        score_mid = oracle._estimate_market_demand(
            {
                "set_id": "10332",
                "source": "lego_proxy_reader",
                "current_price": 229.99,
            },
            85,
            forecast=forecast,
            recent_prices=recent_rows,
        )
        score_high = oracle._estimate_market_demand(
            {
                "set_id": "13000",
                "source": "lego_proxy_reader",
                "current_price": 699.99,
            },
            85,
            forecast=forecast,
            recent_prices=recent_rows,
        )

        self.assertLess(score_mid, 100)
        self.assertNotEqual(score_mid, score_high)

    def test_sort_gemini_candidates_prefers_more_capable_latest(self) -> None:
        ranked = DiscoveryOracle._sort_gemini_model_candidates(
            [
                "models/gemini-1.5-flash",
                "models/gemini-2.0-flash",
                "models/gemini-2.5-pro",
                "models/gemini-2.0-flash-lite",
            ]
        )
        self.assertEqual(ranked[0], "models/gemini-2.5-pro")
        self.assertIn("models/gemini-2.0-flash", ranked)

    def test_sort_gemini_candidates_honors_preferred_when_present(self) -> None:
        ranked = DiscoveryOracle._sort_gemini_model_candidates(
            [
                "models/gemini-2.5-pro",
                "models/gemini-2.0-flash",
            ],
            preferred_model="gemini-2.0-flash",
        )
        self.assertEqual(ranked[0], "models/gemini-2.0-flash")

    def test_should_rotate_gemini_model_for_quota_or_404(self) -> None:
        self.assertTrue(
            DiscoveryOracle._should_rotate_gemini_model(Exception("404 model not found"))
        )
        self.assertTrue(
            DiscoveryOracle._should_rotate_gemini_model(Exception("Quota exceeded"))
        )
        self.assertFalse(
            DiscoveryOracle._should_rotate_gemini_model(Exception("invalid json response"))
        )

    def test_detect_global_quota_exhausted(self) -> None:
        self.assertTrue(
            DiscoveryOracle._is_global_quota_exhausted(
                "Quota exceeded for metric ... limit: 0, model: gemini-2.0-flash"
            )
        )
        self.assertFalse(
            DiscoveryOracle._is_global_quota_exhausted(
                "Quota exceeded for metric ... limit: 50, model: gemini-2.0-flash"
            )
        )

    def test_openrouter_free_model_detection(self) -> None:
        free_payload = {
            "id": "openai/gpt-oss-20b:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"modality": "text->text"},
        }
        paid_payload = {
            "id": "openai/gpt-4.1",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "architecture": {"modality": "text->text"},
        }
        self.assertTrue(DiscoveryOracle._is_openrouter_free_model(free_payload))
        self.assertFalse(DiscoveryOracle._is_openrouter_free_model(paid_payload))

    def test_openrouter_candidate_ranking(self) -> None:
        payloads = [
            {
                "id": "vendor/model-mini:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 32000,
                "architecture": {"modality": "text->text"},
            },
            {
                "id": "vendor/model-pro:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 128000,
                "architecture": {"modality": "text->text"},
            },
        ]
        ranked = DiscoveryOracle._sort_openrouter_model_candidates(payloads)
        self.assertEqual(ranked[0], "vendor/model-pro:free")

    def test_classify_openrouter_probe_failure(self) -> None:
        self.assertEqual(
            DiscoveryOracle._classify_openrouter_probe_failure(
                "429 quota exceeded ... limit: 0"
            ),
            "quota_exhausted_global",
        )
        self.assertEqual(
            DiscoveryOracle._classify_openrouter_probe_failure(
                "429 rate limit"
            ),
            "quota_limited",
        )
        self.assertEqual(
            DiscoveryOracle._classify_openrouter_probe_failure(
                "404 model not found"
            ),
            "unsupported_or_denied",
        )
        self.assertTrue(
            DiscoveryOracle._should_rotate_openrouter_model(
                Exception("OpenRouter response missing choices")
            )
        )

    async def test_get_ai_insight_uses_openrouter_when_available(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True) as init_or:
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        init_or.assert_called()
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle._openrouter_inventory_loaded = True

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        with patch.object(
            oracle,
            "_openrouter_generate",
            return_value='{"score": 88, "summary": "Buon potenziale", "predicted_eol_date": "2026-11-01"}',
        ):
            insight = await oracle._get_ai_insight(candidate)

        self.assertEqual(insight.score, 88)
        self.assertEqual(insight.predicted_eol_date, "2026-11-01")

    def test_openrouter_generate_retries_on_malformed_payload(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-pro:free"

        with patch.object(
            oracle,
            "_openrouter_chat_completion",
            side_effect=[
                {"id": "resp-missing-choices"},
                {"choices": [{"message": {"content": '{"score": 71, "summary": "ok"}'}}]},
            ],
        ) as mocked_call:
            text = oracle._openrouter_generate("prompt")

        self.assertIn('"score": 71', text)
        self.assertEqual(mocked_call.call_count, 2)

    async def test_openrouter_malformed_threshold_disables_provider(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle._openrouter_inventory_loaded = True
        oracle.openrouter_malformed_limit = 2

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }

        with patch.object(
            oracle,
            "_openrouter_generate",
            side_effect=RuntimeError("OpenRouter response missing text content"),
        ), patch.object(
            oracle,
            "_advance_openrouter_model_locked",
            new=AsyncMock(return_value=False),
        ):
            first = await oracle._get_ai_insight(candidate)
            second = await oracle._get_ai_insight(candidate)

        self.assertTrue(first.fallback_used)
        self.assertTrue(second.fallback_used)
        self.assertIsNone(oracle._openrouter_model_id)
        self.assertEqual(oracle.ai_runtime.get("mode"), "fallback_openrouter_malformed_payload")

    async def test_openrouter_opportunistic_recovers_with_alternate_model(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-1:free"
        oracle._openrouter_inventory_loaded = True
        oracle._openrouter_candidates = ["vendor/model-1:free", "vendor/model-2:free"]
        oracle._openrouter_available_candidates = ["vendor/model-1:free", "vendor/model-2:free"]
        oracle.openrouter_opportunistic_enabled = True
        oracle.openrouter_opportunistic_attempts = 3
        oracle.openrouter_opportunistic_timeout_sec = 5.0

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }

        async def rotate(reason: str) -> bool:  # noqa: ARG001
            oracle._openrouter_model_id = "vendor/model-2:free"
            return True

        with patch.object(
            oracle,
            "_openrouter_generate",
            side_effect=[
                RuntimeError("OpenRouter error 429: rate limit"),
                '{"score": 84, "summary": "ok", "predicted_eol_date": "2026-10-01"}',
            ],
        ) as mocked_generate, patch.object(
            oracle,
            "_advance_openrouter_model_locked",
            new=AsyncMock(side_effect=rotate),
        ):
            insight = await oracle._get_ai_insight(candidate)

        self.assertFalse(insight.fallback_used)
        self.assertEqual(insight.score, 84)
        self.assertEqual(oracle._openrouter_model_id, "vendor/model-2:free")
        self.assertEqual(mocked_generate.call_count, 2)

    async def test_openrouter_opportunistic_exhaustion_falls_back(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-1:free"
        oracle._openrouter_inventory_loaded = True
        oracle._openrouter_candidates = ["vendor/model-1:free", "vendor/model-2:free"]
        oracle._openrouter_available_candidates = ["vendor/model-1:free", "vendor/model-2:free"]
        oracle.openrouter_opportunistic_enabled = True
        oracle.openrouter_opportunistic_attempts = 2
        oracle.openrouter_opportunistic_timeout_sec = 5.0

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }

        async def rotate(reason: str) -> bool:  # noqa: ARG001
            oracle._openrouter_model_id = "vendor/model-2:free"
            return True

        with patch.object(
            oracle,
            "_openrouter_generate",
            side_effect=[
                RuntimeError("OpenRouter error 429: rate limit"),
                RuntimeError("OpenRouter error 429: rate limit"),
            ],
        ) as mocked_generate, patch.object(
            oracle,
            "_advance_openrouter_model_locked",
            new=AsyncMock(side_effect=rotate),
        ):
            insight = await oracle._get_ai_insight(candidate)

        self.assertTrue(insight.fallback_used)
        self.assertEqual(mocked_generate.call_count, 2)
        self.assertEqual(oracle._openrouter_model_id, "vendor/model-2:free")
        self.assertNotEqual(oracle.ai_runtime.get("mode"), "fallback_after_openrouter_error")

    async def test_openrouter_non_rate_error_does_not_disable_provider(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-1:free"
        oracle._openrouter_inventory_loaded = True
        oracle._openrouter_candidates = ["vendor/model-1:free"]
        oracle._openrouter_available_candidates = ["vendor/model-1:free"]
        oracle.openrouter_opportunistic_enabled = True
        oracle.openrouter_opportunistic_attempts = 2
        oracle.openrouter_opportunistic_timeout_sec = 5.0

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }

        with patch.object(
            oracle,
            "_openrouter_generate",
            side_effect=RuntimeError("OpenRouter error 500: upstream crash"),
        ), patch.object(
            oracle,
            "_advance_openrouter_model_locked",
            new=AsyncMock(return_value=False),
        ):
            insight = await oracle._get_ai_insight(candidate)

        self.assertTrue(insight.fallback_used)
        self.assertEqual(oracle._openrouter_model_id, "vendor/model-1:free")
        self.assertNotEqual(oracle.ai_runtime.get("mode"), "fallback_after_openrouter_error")

    async def test_openrouter_non_json_text_is_parsed_into_ai_insight(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle._openrouter_inventory_loaded = True

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        with patch.object(
            oracle,
            "_openrouter_generate",
            return_value=(
                "Valutazione investimento a 12 mesi. "
                "Punteggio: 77/100. "
                "Set con domanda stabile e buona rivendibilita'."
            ),
        ):
            insight = await oracle._get_ai_insight(candidate)

        self.assertFalse(insight.fallback_used)
        self.assertEqual(insight.score, 77)
        self.assertEqual(insight.confidence, "LOW_CONFIDENCE")
        self.assertIn("non json", str(insight.risk_note or "").lower())

    async def test_openrouter_non_json_uses_repair_prompt_before_text_parse(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle._openrouter_inventory_loaded = True

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        with patch.object(
            oracle,
            "_openrouter_generate",
            side_effect=[
                "Analisi descrittiva senza JSON e senza punteggio esplicito.",
                '{"score": 86, "summary": "Riparato JSON", "predicted_eol_date": "2026-12-01"}',
            ],
        ) as mocked_generate:
            insight = await oracle._get_ai_insight(candidate)

        self.assertFalse(insight.fallback_used)
        self.assertEqual(insight.score, 86)
        self.assertEqual(insight.confidence, "HIGH_CONFIDENCE")
        self.assertIsNone(insight.risk_note)
        self.assertEqual(mocked_generate.call_count, 2)

    async def test_openrouter_json_repair_can_use_secondary_model(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-a:free"
        oracle._openrouter_inventory_loaded = True

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }

        with patch.object(
            oracle,
            "_resolve_openrouter_json_repair_model",
            return_value="vendor/model-b:free",
        ), patch.object(
            oracle,
            "_openrouter_generate",
            return_value='{"score": 89, "summary": "repair ok", "predicted_eol_date": "2026-12-01"}',
        ) as mocked_generate:
            insight = await oracle._repair_openrouter_non_json_output(
                raw_text="testo non json",
                candidate=candidate,
                timeout_sec=8.0,
            )

        self.assertIsNotNone(insight)
        self.assertEqual(insight.score, 89)
        self.assertEqual(mocked_generate.call_args.kwargs.get("model_id_override"), "vendor/model-b:free")

    def test_probe_openrouter_model_accepts_non_json_scored_text(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle.strict_ai_probe_validation = True

        with patch.object(
            oracle,
            "_openrouter_chat_completion",
            return_value={"choices": [{"message": {"content": "Punteggio 74/100, outlook positivo."}}]},
        ):
            ok, reason = oracle._probe_openrouter_model("vendor/model-pro:free")

        self.assertTrue(ok)
        self.assertEqual(reason, "ok_text_non_json")

    def test_extract_openrouter_text_reads_tool_call_arguments(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "emit_result",
                                    "arguments": '{"score": 81, "summary": "ok", "predicted_eol_date": null}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        text = DiscoveryOracle._extract_openrouter_text(payload)
        self.assertIn('"score": 81', text)

    async def test_ranking_prefilter_limits_ai_scoring_calls(self) -> None:
        repo = FakeRepo()
        now = datetime.now(timezone.utc)
        candidates = []
        for idx in range(14):
            set_id = str(76000 + idx)
            repo.recent[set_id] = [
                {
                    "set_id": set_id,
                    "price": 25.0 + float((idx + d) % 9),
                    "platform": "vinted" if d % 2 == 0 else "subito",
                    "recorded_at": (now - timedelta(days=d)).isoformat(),
                }
                for d in range(8)
            ]
            candidates.append(
                {
                    "set_id": set_id,
                    "set_name": f"Set {set_id}",
                    "theme": "City" if idx % 2 == 0 else "Icons",
                    "source": "lego_proxy_reader" if idx < 10 else "amazon_proxy_reader",
                    "current_price": 39.99 + idx,
                    "eol_date_prediction": "2026-09-01",
                    "metadata": {},
                    "mock_score": max(35, 95 - idx * 4),
                    "mock_fallback": False,
                }
            )

        oracle = DummyOracle(repo, candidates)
        oracle.ai_rank_max_candidates = 5
        report = await oracle.discover_with_diagnostics(persist=False, top_limit=20, fallback_limit=3)
        ranking_diag = report["diagnostics"]["ranking"]

        self.assertEqual(oracle.ai_calls, 5)
        self.assertEqual(int(ranking_diag.get("ai_shortlist_count") or 0), 5)
        self.assertEqual(int(ranking_diag.get("ai_prefilter_skipped_count") or 0), len(candidates) - 5)
        self.assertEqual(int(ranking_diag.get("ai_cache_misses") or 0), 5)

    async def test_ai_insight_cache_reuses_previous_score(self) -> None:
        repo = FakeRepo()
        now = datetime.now(timezone.utc)
        repo.recent["75367"] = [
            {
                "set_id": "75367",
                "price": 120.0 + (idx % 3),
                "platform": "vinted",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(6)
        ]
        candidates = [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "theme": "Star Wars",
                "source": "lego_proxy_reader",
                "current_price": 129.99,
                "eol_date_prediction": "2026-05-01",
                "metadata": {},
                "mock_score": 82,
                "mock_fallback": False,
            }
        ]
        oracle = DummyOracle(repo, candidates)
        oracle.ai_rank_max_candidates = 1
        oracle.ai_cache_ttl_sec = 3600.0

        first_report = await oracle.discover_with_diagnostics(persist=False, top_limit=5, fallback_limit=1)
        second_report = await oracle.discover_with_diagnostics(persist=False, top_limit=5, fallback_limit=1)

        self.assertEqual(oracle.ai_calls, 1)
        self.assertEqual(int(first_report["diagnostics"]["ranking"].get("ai_cache_hits") or 0), 0)
        self.assertGreaterEqual(int(second_report["diagnostics"]["ranking"].get("ai_cache_hits") or 0), 1)

    async def test_ai_persisted_cache_reuses_db_score_before_external_call(self) -> None:
        repo = FakeRepo()
        now = datetime.now(timezone.utc)
        repo.recent["75367"] = [
            {
                "set_id": "75367",
                "price": 129.0,
                "platform": "vinted",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(4)
        ]
        repo.recent_ai_insights["75367"] = {
            "set_id": "75367",
            "ai_investment_score": 74,
            "ai_analysis_summary": "Cache DB recente",
            "eol_date_prediction": "2026-12-01",
            "metadata": {
                "ai_raw_score": 86,
                "ai_fallback_used": False,
                "ai_confidence": "HIGH_CONFIDENCE",
            },
        }
        candidates = [
            {
                "set_id": "75367",
                "set_name": "LEGO Star Wars",
                "theme": "Star Wars",
                "source": "lego_proxy_reader",
                "current_price": 129.99,
                "eol_date_prediction": "2026-05-01",
                "metadata": {},
                "mock_score": 35,
                "mock_fallback": False,
            }
        ]
        oracle = DummyOracle(repo, candidates)
        oracle.ai_rank_max_candidates = 1
        oracle.ai_cache_ttl_sec = 3600.0
        oracle.ai_persisted_cache_ttl_sec = 172800.0

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=5, fallback_limit=1)

        self.assertEqual(oracle.ai_calls, 0)
        self.assertGreaterEqual(int(report["diagnostics"]["ranking"].get("ai_persisted_cache_hits") or 0), 1)
        self.assertGreaterEqual(int(report["diagnostics"]["ranking"].get("ai_cache_hits") or 0), 1)

    def test_extract_json_from_wrapped_text(self) -> None:
        raw = "Risposta:\n{\"score\": 77, \"summary\": \"ok\"}\nfine"
        payload = DiscoveryOracle._extract_json(raw)

        self.assertEqual(payload["score"], 77)
        self.assertEqual(payload["summary"], "ok")

    def test_select_probe_candidates_keeps_head_and_sparse_tail(self) -> None:
        candidates = [f"m{i}" for i in range(1, 11)]
        selected = DiscoveryOracle._select_probe_candidates(candidates, 5)

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[:4], ["m1", "m2", "m3", "m4"])
        self.assertIn(selected[-1], candidates[4:])

    def test_probe_candidates_with_budget_early_stop(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_probe_max_candidates = 3
        oracle.ai_probe_batch_size = 1
        oracle.ai_probe_early_successes = 1
        oracle.ai_probe_budget_sec = 60.0

        report = oracle._probe_candidates_with_budget(
            provider="TestAI",
            candidates=["m1", "m2", "m3", "m4"],
            probe_fn=lambda model: (model == "m1", "ok" if model == "m1" else "fail"),
            classify_fn=lambda reason: "probe_error",
        )
        by_model = {row["model"]: row for row in report}
        self.assertTrue(by_model["m1"]["available"])
        self.assertEqual(by_model["m2"]["status"], "not_probed_early_stop")
        self.assertEqual(by_model["m3"]["status"], "not_probed_early_stop")
        self.assertEqual(by_model["m4"]["status"], "skipped_low_priority")

    def test_probe_candidates_with_budget_marks_budget_exhausted(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_probe_max_candidates = 3
        oracle.ai_probe_batch_size = 1
        oracle.ai_probe_early_successes = 1
        oracle.ai_probe_budget_sec = 0.0

        report = oracle._probe_candidates_with_budget(
            provider="TestAI",
            candidates=["m1", "m2", "m3", "m4"],
            probe_fn=lambda model: (False, "timeout"),
            classify_fn=lambda reason: "transient_error",
        )
        by_model = {row["model"]: row for row in report}
        self.assertEqual(by_model["m1"]["status"], "not_probed_budget_exhausted")
        self.assertEqual(by_model["m2"]["status"], "not_probed_budget_exhausted")
        self.assertEqual(by_model["m3"]["status"], "not_probed_budget_exhausted")
        self.assertEqual(by_model["m4"]["status"], "skipped_low_priority")

    def test_ai_score_collapse_detection_true(self) -> None:
        ranked = [{"ai_investment_score": score} for score in [50, 50, 50, 49, 51, 50, 50, 49]]
        self.assertTrue(DiscoveryOracle._is_ai_score_collapse(ranked))

    def test_ai_score_collapse_detection_false_with_wide_spread(self) -> None:
        ranked = [{"ai_investment_score": score} for score in [35, 42, 55, 61, 73, 67, 58, 49]]
        self.assertFalse(DiscoveryOracle._is_ai_score_collapse(ranked))

    async def test_score_ai_shortlist_retries_timeout_then_succeeds(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_scoring_item_timeout_sec = 0.05
        oracle.ai_scoring_timeout_retries = 1
        oracle.ai_scoring_retry_timeout_sec = 0.05
        oracle.ai_scoring_hard_budget_sec = 2.0

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        shortlist = [{"set_id": "75367", "candidate": candidate}]
        call_counter = {"value": 0}

        async def fake_insight(_candidate):  # noqa: ANN001
            call_counter["value"] += 1
            if call_counter["value"] == 1:
                await asyncio.sleep(0.2)
            return AIInsight(score=84, summary="ok", predicted_eol_date="2026-10-01")

        with patch.object(oracle, "_get_ai_insight", new=AsyncMock(side_effect=fake_insight)):
            results, stats = await oracle._score_ai_shortlist(shortlist)

        self.assertEqual(call_counter["value"], 2)
        self.assertFalse(results["75367"].fallback_used)
        self.assertEqual(int(stats["ai_scored_count"]), 1)
        self.assertEqual(int(stats["ai_errors"]), 0)
        self.assertEqual(int(stats["ai_timeout_count"]), 0)

    async def test_score_ai_shortlist_timeout_recovery_retries_before_fallback(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle.ai_scoring_item_timeout_sec = 0.05
        oracle.ai_scoring_timeout_retries = 0
        oracle.ai_scoring_retry_timeout_sec = 0.05
        oracle.ai_scoring_hard_budget_sec = 2.0
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 1,
        }

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        shortlist = [{"set_id": "75367", "candidate": candidate}]
        call_counter = {"value": 0}

        async def fake_insight(_candidate):  # noqa: ANN001
            call_counter["value"] += 1
            if call_counter["value"] == 1:
                await asyncio.sleep(0.2)
            return AIInsight(score=84, summary="ok", predicted_eol_date="2026-10-01")

        with patch.object(oracle, "_get_ai_insight", new=AsyncMock(side_effect=fake_insight)), patch.object(
            oracle,
            "_recover_openrouter_after_timeout",
            new=AsyncMock(return_value=True),
        ) as mocked_recovery:
            results, stats = await oracle._score_ai_shortlist(shortlist)

        self.assertEqual(call_counter["value"], 2)
        self.assertFalse(results["75367"].fallback_used)
        self.assertEqual(int(stats["ai_scored_count"]), 1)
        self.assertEqual(int(stats["ai_errors"]), 0)
        mocked_recovery.assert_awaited_once()

    async def test_top_pick_rescue_attempts_external_for_unscored_top_pick(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_top_pick_rescue_enabled = True
        oracle.ai_top_pick_rescue_count = 3
        oracle.ai_top_pick_rescue_timeout_sec = 3.0

        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        prepared = [
            {
                "candidate": candidate,
                "set_id": "75367",
                "theme": "Star Wars",
                "forecast": oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={}),
                "history_30": [],
                "prefilter_score": 85,
                "prefilter_rank": 5,
                "ai_shortlisted": False,
            }
        ]
        ranked = [
            {
                "set_id": "75367",
                "composite_score": 72,
                "ai_fallback_used": True,
            }
        ]
        ai_results = {}

        with patch.object(
            oracle,
            "_get_ai_insight",
            new=AsyncMock(
                return_value=AIInsight(
                    score=88,
                    summary="rescued",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            ),
        ) as mocked_get:
            stats = await oracle._rescue_top_pick_ai_scores(
                prepared=prepared,
                ranked=ranked,
                ai_results=ai_results,
            )

        self.assertEqual(mocked_get.await_count, 1)
        self.assertEqual(int(stats["ai_top_pick_rescue_attempts"]), 1)
        self.assertEqual(int(stats["ai_top_pick_rescue_successes"]), 1)
        self.assertIn("75367", ai_results)
        self.assertFalse(ai_results["75367"].fallback_used)

    async def test_top_pick_rescue_uses_composite_order_not_input_order(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_top_pick_rescue_enabled = True
        oracle.ai_top_pick_rescue_count = 1
        oracle.ai_top_pick_rescue_timeout_sec = 3.0

        candidate_low = {
            "set_id": "10001",
            "set_name": "Low composite",
            "theme": "City",
            "source": "lego_proxy_reader",
            "current_price": 29.99,
            "eol_date_prediction": "2026-05-01",
        }
        candidate_high = {
            "set_id": "10002",
            "set_name": "High composite",
            "theme": "Icons",
            "source": "lego_proxy_reader",
            "current_price": 119.99,
            "eol_date_prediction": "2026-06-01",
        }

        prepared = [
            {
                "candidate": candidate_low,
                "set_id": "10001",
                "theme": "City",
                "forecast": oracle.forecaster.forecast(candidate=candidate_low, history_rows=[], theme_baseline={}),
                "history_30": [],
                "prefilter_score": 95,
                "prefilter_rank": 1,
                "ai_shortlisted": False,
            },
            {
                "candidate": candidate_high,
                "set_id": "10002",
                "theme": "Icons",
                "forecast": oracle.forecaster.forecast(candidate=candidate_high, history_rows=[], theme_baseline={}),
                "history_30": [],
                "prefilter_score": 60,
                "prefilter_rank": 5,
                "ai_shortlisted": False,
            },
        ]

        # Intentionally unsorted input: first row has lower composite score.
        ranked = [
            {"set_id": "10001", "composite_score": 60, "forecast_score": 40, "market_demand_score": 60, "ai_fallback_used": True},
            {"set_id": "10002", "composite_score": 78, "forecast_score": 55, "market_demand_score": 90, "ai_fallback_used": True},
        ]
        ai_results = {}

        calls = []

        async def fake_get_ai_insight(candidate):  # noqa: ANN001
            calls.append(str(candidate.get("set_id")))
            return AIInsight(
                score=85,
                summary="rescued",
                predicted_eol_date=candidate.get("eol_date_prediction"),
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )

        with patch.object(oracle, "_get_ai_insight", new=AsyncMock(side_effect=fake_get_ai_insight)):
            stats = await oracle._rescue_top_pick_ai_scores(
                prepared=prepared,
                ranked=ranked,
                ai_results=ai_results,
            )

        self.assertEqual(calls, ["10002"])
        self.assertEqual(int(stats["ai_top_pick_rescue_attempts"]), 1)
        self.assertEqual(int(stats["ai_top_pick_rescue_successes"]), 1)
        self.assertIn("10002", ai_results)

    async def test_score_ai_shortlist_limits_openrouter_concurrency_with_single_model(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle.ai_scoring_concurrency = 4
        oracle.ai_scoring_item_timeout_sec = 1.0
        oracle.ai_scoring_timeout_retries = 0
        oracle.ai_scoring_hard_budget_sec = 5.0
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 1,
        }

        shortlist = []
        for idx in range(4):
            set_id = str(75000 + idx)
            shortlist.append(
                {
                    "set_id": set_id,
                    "candidate": {
                        "set_id": set_id,
                        "set_name": f"Set {set_id}",
                        "theme": "City",
                        "source": "lego_proxy_reader",
                        "current_price": 49.99,
                        "eol_date_prediction": "2026-10-01",
                    },
                }
            )

        counters = {"active": 0, "max_active": 0}
        lock = asyncio.Lock()

        async def fake_insight(_candidate):  # noqa: ANN001
            async with lock:
                counters["active"] += 1
                counters["max_active"] = max(counters["max_active"], counters["active"])
            await asyncio.sleep(0.05)
            async with lock:
                counters["active"] -= 1
            return AIInsight(score=75, summary="ok", predicted_eol_date="2026-10-01")

        with patch.object(oracle, "_get_ai_insight", new=AsyncMock(side_effect=fake_insight)):
            results, _stats = await oracle._score_ai_shortlist(shortlist)

        self.assertEqual(len(results), 4)
        self.assertLessEqual(counters["max_active"], 2)

    async def test_score_ai_shortlist_batch_scores_multiple_candidates_with_single_call(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 1,
        }
        oracle.ai_batch_scoring_enabled = True
        oracle.ai_batch_min_candidates = 2
        oracle.ai_batch_max_candidates = 10
        oracle.ai_scoring_hard_budget_sec = 8.0
        oracle.ai_scoring_item_timeout_sec = 2.0

        shortlist = [
            {
                "set_id": "75367",
                "candidate": {
                    "set_id": "75367",
                    "set_name": "Set 75367",
                    "theme": "Star Wars",
                    "source": "lego_proxy_reader",
                    "current_price": 99.99,
                    "eol_date_prediction": "2026-12-01",
                },
            },
            {
                "set_id": "76281",
                "candidate": {
                    "set_id": "76281",
                    "set_name": "Set 76281",
                    "theme": "Marvel",
                    "source": "lego_proxy_reader",
                    "current_price": 79.99,
                    "eol_date_prediction": "2026-10-01",
                },
            },
        ]

        batch_json = (
            '{"results": ['
            '{"set_id":"75367","score":88,"summary":"ok1","predicted_eol_date":"2026-12-01"},'
            '{"set_id":"76281","score":81,"summary":"ok2","predicted_eol_date":"2026-10-01"}'
            "]} "
        )

        with patch.object(oracle, "_openrouter_generate", return_value=batch_json) as mocked_batch, patch.object(
            oracle,
            "_get_ai_insight",
            new=AsyncMock(side_effect=AssertionError("per-candidate path should not run")),
        ):
            results, stats = await oracle._score_ai_shortlist(shortlist)

        self.assertEqual(len(results), 2)
        self.assertEqual(int(stats["ai_scored_count"]), 2)
        self.assertEqual(int(stats["ai_batch_scored_count"]), 2)
        self.assertEqual(int(stats["ai_errors"]), 0)
        self.assertEqual(mocked_batch.call_count, 1)

    async def test_score_ai_shortlist_batch_uses_json_repair_when_non_json(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 1,
        }

        entries = [
            {
                "set_id": "75367",
                "candidate": {
                    "set_id": "75367",
                    "set_name": "Set 75367",
                    "theme": "Star Wars",
                    "source": "lego_proxy_reader",
                    "current_price": 99.99,
                    "eol_date_prediction": "2026-12-01",
                },
            }
        ]

        repaired = {
            "75367": AIInsight(
                score=84,
                summary="repair ok",
                predicted_eol_date="2026-12-01",
                fallback_used=False,
            )
        }
        with patch.object(oracle, "_openrouter_generate", return_value="not-json"), patch.object(
            oracle,
            "_repair_openrouter_non_json_batch_output",
            new=AsyncMock(return_value=repaired),
        ) as mocked_repair:
            results, error = await oracle._score_ai_shortlist_batch(entries, deadline=time.monotonic() + 10.0)

        self.assertIsNone(error)
        self.assertEqual(set(results.keys()), {"75367"})
        self.assertEqual(results["75367"].score, 84)
        mocked_repair.assert_awaited_once()

    def test_compute_fast_fail_timeouts_reduces_timeout_when_budget_is_tight(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_fast_fail_enabled = True
        oracle.ai_scoring_item_timeout_sec = 18.0
        oracle.ai_scoring_retry_timeout_sec = 7.0
        oracle.ai_scoring_timeout_retries = 1

        first_timeout, retry_timeout, retries = oracle._compute_fast_fail_timeouts(
            pending_count=10,
            budget_left_sec=30.0,
        )

        self.assertLess(first_timeout, 18.0)
        self.assertLessEqual(retry_timeout, first_timeout)
        self.assertLessEqual(retries, 1)

    def test_batch_payload_to_ai_insights_accepts_results_array(self) -> None:
        candidates = [
            {"set_id": "75367", "set_name": "Set A", "theme": "Star Wars", "source": "lego_proxy_reader"},
            {"set_id": "76281", "set_name": "Set B", "theme": "Marvel", "source": "lego_proxy_reader"},
        ]
        payload = {
            "results": [
                {"set_id": "75367", "score": 90, "summary": "ok", "predicted_eol_date": "2026-11-01"},
                {"set_id": "76281", "score": 75, "summary": "ok", "predicted_eol_date": None},
            ]
        }

        insights = DiscoveryOracle._batch_payload_to_ai_insights(payload, candidates)
        self.assertEqual(set(insights.keys()), {"75367", "76281"})
        self.assertEqual(insights["75367"].score, 90)
        self.assertEqual(insights["76281"].score, 75)

    def test_bootstrap_thresholds_can_promote_high_confidence_when_history_is_short(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.bootstrap_thresholds_enabled = True
        oracle.bootstrap_min_history_points = 45
        oracle.bootstrap_min_upside_probability = 0.52
        oracle.bootstrap_min_confidence_score = 50
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60

        short_history_row = {
            "set_id": "77051",
            "ai_fallback_used": False,
            "composite_score": 71,
            "forecast_probability_upside_12m": 55.6,
            "confidence_score": 52,
            "forecast_data_points": 20,
        }
        long_history_row = {
            "set_id": "77051",
            "ai_fallback_used": False,
            "composite_score": 71,
            "forecast_probability_upside_12m": 55.6,
            "confidence_score": 52,
            "forecast_data_points": 120,
        }

        self.assertTrue(oracle._is_high_confidence_pick(short_history_row))
        self.assertFalse(oracle._is_high_confidence_pick(long_history_row))

    def test_low_confidence_note_mentions_bootstrap_when_active(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.bootstrap_thresholds_enabled = True
        oracle.bootstrap_min_history_points = 45
        oracle.bootstrap_min_upside_probability = 0.52
        oracle.bootstrap_min_confidence_score = 50
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68

        row = {
            "set_id": "77051",
            "ai_fallback_used": False,
            "forecast_probability_upside_12m": 50.0,
            "confidence_score": 45,
            "forecast_data_points": 20,
        }
        note = oracle._build_low_confidence_note(row)
        self.assertIn("Bootstrap soglie attivo", note)
        self.assertIn("50.0% < 52%", note)
        self.assertIn("45 < 50", note)

    def test_format_exception_for_log_timeout_has_message(self) -> None:
        err_type, err_message = DiscoveryOracle._format_exception_for_log(asyncio.TimeoutError())
        self.assertEqual(err_type, "TimeoutError")
        self.assertTrue(err_message)
        self.assertIn("timeout", err_message.lower())


if __name__ == "__main__":
    unittest.main()
