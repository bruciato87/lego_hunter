from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from oracle import AIInsight, DiscoveryOracle
from scrapers import MarketListing


class FakeRepo:
    def __init__(self) -> None:
        self.upserted = []
        self.snapshots = []
        self.recent = {}
        self.theme_baselines = {}

    def upsert_opportunity(self, record):  # noqa: ANN001
        self.upserted.append(record)
        return {"ok": True}

    def insert_market_snapshot(self, record):  # noqa: ANN001
        self.snapshots.append(record)
        return {"ok": True}

    def get_recent_market_prices(self, set_id, days=30, platform=None):  # noqa: ANN001
        return self.recent.get(set_id, [])

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


class DummyOracle(DiscoveryOracle):
    def __init__(self, repository, candidates):  # noqa: ANN001
        super().__init__(repository, gemini_api_key=None, min_ai_score=60)
        self._candidates = candidates

    async def _collect_source_candidates(self):
        return self._candidates

    async def _get_ai_insight(self, candidate):  # noqa: ANN001
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

        self.assertIn("composite_score", row)
        self.assertIn("forecast_score", row)
        self.assertIn("forecast_probability_upside_12m", row)
        self.assertIn("confidence_score", row)
        self.assertIn("expected_roi_12m_pct", row)
        self.assertGreaterEqual(int(row["composite_score"]), 1)
        self.assertLessEqual(int(row["composite_score"]), 100)

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
                "mock_score": 73,
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


if __name__ == "__main__":
    unittest.main()
