from __future__ import annotations

import asyncio
import csv
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from forecast import ForecastInsight
from oracle import AIInsight, DiscoveryOracle, PatternEvaluation
from scrapers import MarketListing


class FakeRepo:
    def __init__(self) -> None:
        self.upserted = []
        self.snapshots = []
        self.recent = {}
        self.theme_baselines = {}
        self.recent_ai_insights = {}
        self.holdings = []

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

    def get_portfolio(self, status="holding"):  # noqa: ANN001
        if status != "holding":
            return []
        return list(self.holdings)


class DummyOracle(DiscoveryOracle):
    def __init__(self, repository, candidates):  # noqa: ANN001
        super().__init__(repository, gemini_api_key=None, min_ai_score=60)
        self._candidates = candidates
        self.ai_calls = 0
        self.historical_high_conf_required = False

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

    def test_composite_score_increases_with_historical_prior(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_prior_weight = 0.2
        baseline = oracle._calculate_composite_score(
            ai_score=70,
            demand_score=72,
            forecast_score=69,
            pattern_score=60,
            ai_fallback_used=False,
            historical_score=None,
        )
        boosted = oracle._calculate_composite_score(
            ai_score=70,
            demand_score=72,
            forecast_score=69,
            pattern_score=60,
            ai_fallback_used=False,
            historical_score=95,
        )
        self.assertGreater(boosted, baseline)

    def test_historical_prior_for_candidate_from_reference_cases(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_reference_min_samples = 12
        oracle._historical_reference_cases = []
        for idx in range(20):
            oracle._historical_reference_cases.append(
                {
                    "set_id": str(70000 + idx),
                    "theme": "Star Wars",
                    "theme_norm": "star wars",
                    "set_name": f"Case {idx}",
                    "msrp_usd": 90.0 + idx,
                    "roi_12m_pct": 32.0 + (idx % 5),
                    "win_12m": 1,
                    "source_dataset": "seed",
                    "pattern_tags": "[]",
                }
            )
        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 99.99,
        }

        prior = oracle._historical_prior_for_candidate(candidate)
        self.assertIsNotNone(prior)
        assert prior is not None
        self.assertGreaterEqual(int(prior.get("sample_size") or 0), 12)
        self.assertGreaterEqual(int(prior.get("prior_score") or 0), 60)
        self.assertEqual(prior.get("match_mode"), "direct")

    def test_historical_market_scope_filter_it_eu(self) -> None:
        headers = [
            "set_id",
            "set_name",
            "theme",
            "msrp_usd",
            "roi_12m_pct",
            "win_12m",
            "source_dataset",
            "pattern_tags",
            "end_date",
            "market_country",
            "market_region",
        ]
        rows = [
            {
                "set_id": "10001",
                "set_name": "Italian Case",
                "theme": "City",
                "msrp_usd": "20",
                "roi_12m_pct": "34.5",
                "win_12m": "1",
                "source_dataset": "seed_it",
                "pattern_tags": "[]",
                "end_date": "2025-12-01",
                "market_country": "IT",
                "market_region": "EU",
            },
            {
                "set_id": "10002",
                "set_name": "US Case",
                "theme": "City",
                "msrp_usd": "20",
                "roi_12m_pct": "41.0",
                "win_12m": "1",
                "source_dataset": "seed_us",
                "pattern_tags": "[]",
                "end_date": "2025-12-01",
                "market_country": "US",
                "market_region": "NA",
            },
            {
                "set_id": "10003",
                "set_name": "Unknown Scope",
                "theme": "City",
                "msrp_usd": "20",
                "roi_12m_pct": "12.0",
                "win_12m": "0",
                "source_dataset": "seed_unknown",
                "pattern_tags": "[]",
                "end_date": "2025-12-01",
                "market_country": "",
                "market_region": "",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "historical_scope.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            with patch.dict(
                os.environ,
                {
                    "HISTORICAL_REFERENCE_CASES_PATH": csv_path,
                    "HISTORICAL_ALLOWED_COUNTRIES": "IT",
                    "HISTORICAL_ALLOWED_REGIONS": "EU",
                    "HISTORICAL_INCLUDE_UNKNOWN_MARKET": "false",
                },
                clear=False,
            ):
                oracle = DiscoveryOracle(FakeRepo(), gemini_api_key=None, openrouter_api_key=None)

        loaded_ids = {str(row.get("set_id")) for row in oracle._historical_reference_cases}
        self.assertEqual(loaded_ids, {"10001"})
        self.assertEqual(int(oracle._historical_market_filter_stats.get("rows_loaded") or 0), 1)
        self.assertEqual(int(oracle._historical_market_filter_stats.get("rows_skipped_market_scope") or 0), 2)

    def test_historical_market_scope_infers_eu_for_mendeley_source(self) -> None:
        headers = [
            "set_id",
            "set_name",
            "theme",
            "msrp_usd",
            "roi_12m_pct",
            "win_12m",
            "source_dataset",
            "pattern_tags",
            "end_date",
            "market_country",
            "market_region",
        ]
        rows = [
            {
                "set_id": "20001",
                "set_name": "Legacy Mendeley",
                "theme": "City",
                "msrp_usd": "20",
                "roi_12m_pct": "31.0",
                "win_12m": "1",
                "source_dataset": "mendeley_whole_2018_2019",
                "pattern_tags": "[]",
                "end_date": "2019-04-01",
                "market_country": "",
                "market_region": "",
            },
            {
                "set_id": "20002",
                "set_name": "Unknown non legacy",
                "theme": "City",
                "msrp_usd": "20",
                "roi_12m_pct": "12.0",
                "win_12m": "0",
                "source_dataset": "unknown_dataset",
                "pattern_tags": "[]",
                "end_date": "2025-12-01",
                "market_country": "",
                "market_region": "",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "historical_scope_mendeley.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            with patch.dict(
                os.environ,
                {
                    "HISTORICAL_REFERENCE_CASES_PATH": csv_path,
                    "HISTORICAL_ALLOWED_COUNTRIES": "IT",
                    "HISTORICAL_ALLOWED_REGIONS": "EU",
                    "HISTORICAL_INCLUDE_UNKNOWN_MARKET": "false",
                },
                clear=False,
            ):
                oracle = DiscoveryOracle(FakeRepo(), gemini_api_key=None, openrouter_api_key=None)

        loaded_ids = {str(row.get("set_id")) for row in oracle._historical_reference_cases}
        self.assertEqual(loaded_ids, {"20001"})
        self.assertEqual(int(oracle._historical_market_filter_stats.get("rows_loaded") or 0), 1)
        self.assertEqual(int(oracle._historical_market_filter_stats.get("rows_inferred_market_scope") or 0), 1)

    def test_historical_prior_recency_weight_prefers_recent_cases(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_reference_min_samples = 20
        oracle.historical_recency_halflife_days = 900
        oracle.historical_recency_min_weight = 0.20
        oracle._historical_reference_cases = []

        for idx in range(15):
            row = {
                "set_id": str(95000 + idx),
                "theme": "Marvel",
                "theme_norm": "marvel",
                "set_name": f"Old {idx}",
                "msrp_usd": 60.0,
                "roi_12m_pct": -30.0,
                "win_12m": 0,
                "source_dataset": "legacy_old",
                "pattern_tags": "[]",
                "end_date": "2018-01-01",
                "observation_months": 24,
            }
            row["resolved_weight"] = oracle._historical_case_weight(row)
            oracle._historical_reference_cases.append(row)

        for idx in range(10):
            row = {
                "set_id": str(96000 + idx),
                "theme": "Marvel",
                "theme_norm": "marvel",
                "set_name": f"Recent {idx}",
                "msrp_usd": 60.0,
                "roi_12m_pct": 70.0,
                "win_12m": 1,
                "source_dataset": "ebay_sold_it_90d",
                "pattern_tags": "[]",
                "end_date": datetime.now(timezone.utc).date().isoformat(),
                "observation_months": 6,
            }
            row["resolved_weight"] = oracle._historical_case_weight(row)
            oracle._historical_reference_cases.append(row)

        candidate = {
            "set_id": "76281",
            "set_name": "X-Jet di X-Men",
            "theme": "Marvel",
            "source": "lego_proxy_reader",
            "current_price": 59.99,
        }
        prior = oracle._historical_prior_for_candidate(candidate)
        self.assertIsNotNone(prior)
        assert prior is not None
        self.assertGreater(float(prior.get("avg_roi_12m_pct") or 0.0), 0.0)
        self.assertLess(float(prior.get("effective_sample_size") or 0.0), float(prior.get("sample_size") or 0.0))

    def test_historical_prior_alias_mapping_marvel_to_super_heroes(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_reference_min_samples = 12
        oracle._historical_reference_cases = []
        for idx in range(30):
            oracle._historical_reference_cases.append(
                {
                    "set_id": str(81000 + idx),
                    "theme": "Super Heroes",
                    "theme_norm": "super heroes",
                    "set_name": f"SH Case {idx}",
                    "msrp_usd": 45.0 + (idx % 8),
                    "roi_12m_pct": 34.0 + (idx % 5),
                    "win_12m": 1,
                    "source_dataset": "seed",
                    "pattern_tags": "[]",
                }
            )

        candidate = {
            "set_id": "76281",
            "set_name": "X-Jet di X-Men",
            "theme": "Marvel",
            "source": "lego_proxy_reader",
            "current_price": 59.99,
        }
        prior = oracle._historical_prior_for_candidate(candidate)
        self.assertIsNotNone(prior)
        assert prior is not None
        self.assertEqual(prior.get("match_mode"), "alias")
        self.assertIn("super heroes", prior.get("matched_theme_keys") or [])
        self.assertGreaterEqual(int(prior.get("sample_size") or 0), 12)

    def test_historical_prior_uses_alias_when_direct_sample_is_too_small(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_reference_min_samples = 24
        oracle._historical_reference_cases = []
        for idx in range(8):
            oracle._historical_reference_cases.append(
                {
                    "set_id": str(90000 + idx),
                    "theme": "Ideas",
                    "theme_norm": "ideas",
                    "set_name": f"Ideas Case {idx}",
                    "msrp_usd": 200.0 + idx,
                    "roi_12m_pct": 20.0 + (idx % 3),
                    "win_12m": 0,
                    "source_dataset": "seed",
                    "pattern_tags": "[]",
                }
            )
        for idx in range(30):
            oracle._historical_reference_cases.append(
                {
                    "set_id": str(91000 + idx),
                    "theme": "Advanced Models",
                    "theme_norm": "advanced models",
                    "set_name": f"AM Case {idx}",
                    "msrp_usd": 210.0 + idx,
                    "roi_12m_pct": 24.0 + (idx % 6),
                    "win_12m": 0,
                    "source_dataset": "seed",
                    "pattern_tags": "[]",
                }
            )

        candidate = {
            "set_id": "21341",
            "set_name": "Disney Hocus Pocus: il cottage delle sorelle Sanderson",
            "theme": "Ideas",
            "source": "lego_proxy_reader",
            "current_price": 229.99,
        }
        prior = oracle._historical_prior_for_candidate(candidate)
        self.assertIsNotNone(prior)
        assert prior is not None
        self.assertEqual(prior.get("match_mode"), "alias")
        self.assertGreaterEqual(int(prior.get("sample_size") or 0), 12)

    def test_normalize_theme_key_handles_symbols_and_spacing(self) -> None:
        self.assertEqual(DiscoveryOracle._normalize_theme_key("  Marvel / Super-Heroes  "), "marvel super heroes")

    def test_guess_theme_from_name_covers_recent_catalog_patterns(self) -> None:
        self.assertEqual(
            DiscoveryOracle._guess_theme_from_name("Rover lunare NASA Apollo - LRV"),
            "Technic",
        )
        self.assertEqual(
            DiscoveryOracle._guess_theme_from_name("X-Jet di X-Men"),
            "Marvel",
        )
        self.assertEqual(
            DiscoveryOracle._guess_theme_from_name("In volo con la Dodo Airlines"),
            "Animal Crossing",
        )
        self.assertEqual(
            DiscoveryOracle._guess_theme_from_name("Disney Hocus Pocus: il cottage delle sorelle Sanderson"),
            "Ideas",
        )
        self.assertEqual(
            DiscoveryOracle._guess_theme_from_name("Parco giochi degli animali"),
            "Seasonal",
        )

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

    def test_effective_ai_single_call_shortlist_cap_raises_floor_for_multi_model_inventory(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_max_candidates = 4
        oracle.ai_dynamic_shortlist_enabled = True
        oracle.ai_dynamic_shortlist_floor = 4
        oracle.ai_dynamic_shortlist_multi_model_floor = 5
        oracle.ai_runtime["engine"] = "openrouter"
        oracle.ai_runtime["inventory_available"] = 2

        cap = oracle._effective_ai_single_call_shortlist_cap(41)
        self.assertEqual(cap, 5)

    def test_effective_ai_single_call_shortlist_cap_expands_with_single_model_inventory(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_max_candidates = 12
        oracle.ai_single_call_dynamic_max_candidates = 16
        oracle.ai_dynamic_shortlist_enabled = True
        oracle.ai_dynamic_shortlist_bonus = 1
        oracle.ai_dynamic_shortlist_per_model = 2
        oracle.ai_runtime["engine"] = "openrouter"
        oracle.ai_runtime["inventory_available"] = 1

        cap = oracle._effective_ai_single_call_shortlist_cap(41)
        self.assertEqual(cap, 15)

    def test_select_ai_shortlist_respects_strict_top_k_policy(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_single_call_max_candidates = 12
        oracle.ai_strict_final_top_k_only = True
        oracle.ai_strict_final_top_k = 3
        prepared = [{"set_id": f"p{idx}", "ai_shortlisted": False} for idx in range(1, 6)]

        shortlist, skipped = oracle._select_ai_shortlist(prepared)

        self.assertEqual(len(shortlist), 3)
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(bool(row.get("ai_shortlisted")) for row in shortlist))
        self.assertTrue(all(not bool(row.get("ai_shortlisted")) for row in skipped))

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

    def test_openrouter_runtime_prefers_strict_available_model_over_non_json(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_gemini_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key="fake", openrouter_api_key="fake-or-key")

        oracle.strict_ai_probe_validation = True
        oracle._openrouter_inventory_loaded = False

        with patch.object(
            oracle,
            "_fetch_openrouter_model_payloads",
            return_value=[
                {
                    "id": "vendor/model-a:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text->text"},
                    "context_length": 32000,
                },
                {
                    "id": "vendor/model-b:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text->text"},
                    "context_length": 32000,
                },
            ],
        ), patch.object(
            oracle,
            "_probe_all_openrouter_candidates",
            return_value=[
                {
                    "model": "vendor/model-a:free",
                    "available": True,
                    "status": "available",
                    "reason": "ok_text_non_json",
                },
                {
                    "model": "vendor/model-b:free",
                    "available": True,
                    "status": "available",
                    "reason": "ok_json",
                },
            ],
        ):
            oracle._initialize_openrouter_runtime()

        self.assertEqual(oracle._openrouter_model_id, "vendor/model-b:free")
        self.assertEqual(oracle.ai_runtime.get("mode"), "api_openrouter_inventory")
        self.assertEqual(int(oracle.ai_runtime.get("inventory_available_strict") or 0), 1)
        self.assertIn("vendor/model-b:free", oracle._openrouter_available_strict_candidates)

    def test_openrouter_runtime_uses_strict_reprobe_before_last_resort_non_json(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_gemini_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key="fake", openrouter_api_key="fake-or-key")

        oracle.strict_ai_probe_validation = True
        oracle._openrouter_inventory_loaded = False
        oracle.ai_probe_strict_reprobe_enabled = True
        oracle.ai_strict_model_required_main_shortlist = True

        with patch.object(
            oracle,
            "_fetch_openrouter_model_payloads",
            return_value=[
                {
                    "id": "vendor/model-a:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text->text"},
                    "context_length": 32000,
                },
                {
                    "id": "vendor/model-b:free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text->text"},
                    "context_length": 32000,
                },
            ],
        ), patch.object(
            oracle,
            "_probe_all_openrouter_candidates",
            return_value=[
                {
                    "model": "vendor/model-a:free",
                    "available": True,
                    "status": "available",
                    "reason": "ok_text_non_json",
                },
                {
                    "model": "vendor/model-b:free",
                    "available": False,
                    "status": "not_probed_early_stop",
                    "reason": "Early-stop after finding available model.",
                },
            ],
        ), patch.object(
            oracle,
            "_probe_openrouter_strict_candidates",
            return_value=[
                {
                    "model": "vendor/model-b:free",
                    "available": True,
                    "status": "available",
                    "reason": "ok_json",
                }
            ],
        ) as mocked_strict_reprobe:
            oracle._initialize_openrouter_runtime()

        mocked_strict_reprobe.assert_called_once()
        reprobe_pool = mocked_strict_reprobe.call_args.args[0]
        self.assertIn("vendor/model-b:free", reprobe_pool)
        self.assertEqual(oracle._openrouter_model_id, "vendor/model-b:free")
        self.assertEqual(oracle.ai_runtime.get("mode"), "api_openrouter_inventory")
        self.assertEqual(int(oracle.ai_runtime.get("inventory_available_strict") or 0), 1)
        self.assertIn("vendor/model-b:free", oracle._openrouter_available_strict_candidates)

    def test_openrouter_strict_reprobe_rejects_non_json_probe_results(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_probe_batch_size = 1
        oracle.ai_probe_strict_max_candidates = 3
        oracle.ai_probe_strict_budget_sec = 60.0
        oracle.ai_probe_strict_timeout_sec = 1.0
        oracle.ai_probe_strict_early_successes = 1

        with patch.object(
            oracle,
            "_probe_openrouter_model",
            side_effect=[
                (True, "ok_text_non_json"),
                (True, "ok_json"),
            ],
        ):
            report = oracle._probe_openrouter_strict_candidates(["m1", "m2"])

        by_model = {row["model"]: row for row in report}
        self.assertFalse(by_model["m1"]["available"])
        self.assertEqual(by_model["m1"]["status"], "invalid_output")
        self.assertTrue(by_model["m2"]["available"])
        self.assertEqual(by_model["m2"]["status"], "available")

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

    async def test_discovery_excludes_owned_sets_from_results(self) -> None:
        repo = FakeRepo()
        repo.holdings = [{"set_id": "75367"}]
        candidates = [
            {
                "set_id": "75367",
                "set_name": "Owned set",
                "theme": "Star Wars",
                "source": "lego_proxy_reader",
                "current_price": 129.99,
                "metadata": {},
                "mock_score": 90,
            },
            {
                "set_id": "10332",
                "set_name": "Candidate set",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 229.99,
                "metadata": {},
                "mock_score": 88,
            },
        ]
        oracle = DummyOracle(repo, candidates)
        oracle.min_composite_score = 1
        oracle.ai_no_signal_on_low_strict_pass = False

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)
        diagnostics = report["diagnostics"]

        ranked_ids = {str(row.get("set_id") or "") for row in report["ranked"]}
        selected_ids = {str(row.get("set_id") or "") for row in report["selected"]}
        self.assertNotIn("75367", ranked_ids)
        self.assertNotIn("75367", selected_ids)
        self.assertEqual(int(diagnostics.get("owned_holding_count") or 0), 1)
        self.assertEqual(int(diagnostics.get("owned_excluded_count") or 0), 1)

    async def test_discovery_diagnostics_expose_shortlist_quality_separately_from_total(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle._last_source_diagnostics = {
            "source_strategy": "external_first",
            "source_order": ["external_proxy", "playwright"],
            "selected_source": "external_proxy",
            "source_raw_counts": {"lego_proxy_reader": 2, "amazon_proxy_reader": 2},
            "source_dedup_counts": {"lego_proxy_reader": 2, "amazon_proxy_reader": 2},
            "source_failures": [],
            "source_signals": {},
            "dedup_candidates": 4,
            "fallback_source_used": False,
            "fallback_notes": [],
            "anti_bot_alert": False,
            "anti_bot_message": None,
            "root_cause_hint": None,
        }
        ranked_rows = [
            {
                "set_id": "101",
                "source": "lego_proxy_reader",
                "composite_score": 70,
                "forecast_score": 60,
                "forecast_probability_upside_12m": 64.0,
                "confidence_score": 58,
                "market_demand_score": 90,
                "ai_raw_score": 80,
                "ai_shortlisted": True,
                "ai_fallback_used": False,
            },
            {
                "set_id": "102",
                "source": "lego_proxy_reader",
                "composite_score": 69,
                "forecast_score": 59,
                "forecast_probability_upside_12m": 63.0,
                "confidence_score": 57,
                "market_demand_score": 89,
                "ai_raw_score": 78,
                "ai_shortlisted": True,
                "ai_fallback_used": True,
            },
            {
                "set_id": "103",
                "source": "amazon_proxy_reader",
                "composite_score": 68,
                "forecast_score": 58,
                "forecast_probability_upside_12m": 62.0,
                "confidence_score": 56,
                "market_demand_score": 88,
                "ai_raw_score": 77,
                "ai_shortlisted": True,
                "ai_fallback_used": False,
                "ai_source_origin": "cache_repository",
                "risk_note": "Output AI non JSON: score estratto da testo con parsing robusto.",
            },
            {
                "set_id": "104",
                "source": "amazon_proxy_reader",
                "composite_score": 67,
                "forecast_score": 57,
                "forecast_probability_upside_12m": 61.0,
                "confidence_score": 55,
                "market_demand_score": 87,
                "ai_raw_score": 75,
                "ai_shortlisted": False,
                "ai_fallback_used": True,
            },
        ]
        source_candidates = [{"set_id": row["set_id"]} for row in ranked_rows]

        with (
            patch.object(oracle, "_collect_source_candidates", new=AsyncMock(return_value=source_candidates)),
            patch.object(oracle, "_rank_and_persist_candidates", new=AsyncMock(return_value=ranked_rows)),
        ):
            report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)

        diagnostics = report["diagnostics"]
        self.assertEqual(int(diagnostics["ai_shortlist_effective_count"]), 3)
        self.assertEqual(int(diagnostics["ai_shortlist_fallback_count"]), 1)
        self.assertEqual(int(diagnostics["ai_shortlist_non_json_count"]), 1)
        self.assertEqual(int(diagnostics["ai_shortlist_strict_pass_count"]), 1)
        self.assertAlmostEqual(float(diagnostics["fallback_rate_shortlist"]), 1 / 3, places=4)
        self.assertAlmostEqual(float(diagnostics["non_json_rate_shortlist"]), 1 / 3, places=4)
        self.assertAlmostEqual(float(diagnostics["strict_pass_rate_shortlist"]), 1 / 3, places=4)
        self.assertAlmostEqual(float(diagnostics["fallback_rate_total"]), 0.5, places=4)
        self.assertAlmostEqual(float(diagnostics["non_json_rate_total"]), 0.25, places=4)
        self.assertAlmostEqual(float(diagnostics["non_json_rate_cache_total"]), 0.25, places=4)
        self.assertAlmostEqual(float(diagnostics["non_json_rate_fresh_total"]), 0.0, places=4)

    async def test_discovery_no_signal_when_strict_pass_shortlist_is_low(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_no_signal_on_low_strict_pass = True
        oracle.ai_no_signal_min_strict_pass_rate = 0.5
        oracle._last_source_diagnostics = {
            "source_strategy": "external_first",
            "source_order": ["external_proxy", "playwright"],
            "selected_source": "external_proxy",
            "source_raw_counts": {"lego_proxy_reader": 2, "amazon_proxy_reader": 2},
            "source_dedup_counts": {"lego_proxy_reader": 2, "amazon_proxy_reader": 2},
            "source_failures": [],
            "source_signals": {},
            "dedup_candidates": 4,
            "fallback_source_used": False,
            "fallback_notes": [],
            "anti_bot_alert": False,
            "anti_bot_message": None,
            "root_cause_hint": None,
        }
        ranked_rows = [
            {
                "set_id": "201",
                "source": "lego_proxy_reader",
                "composite_score": 70,
                "forecast_score": 60,
                "forecast_probability_upside_12m": 64.0,
                "confidence_score": 58,
                "market_demand_score": 90,
                "ai_raw_score": 80,
                "ai_shortlisted": True,
                "ai_fallback_used": False,
                "ai_strict_pass": False,
                "risk_note": "Output AI non JSON: score estratto da testo con parsing robusto.",
            },
            {
                "set_id": "202",
                "source": "amazon_proxy_reader",
                "composite_score": 69,
                "forecast_score": 59,
                "forecast_probability_upside_12m": 63.0,
                "confidence_score": 57,
                "market_demand_score": 89,
                "ai_raw_score": 78,
                "ai_shortlisted": True,
                "ai_fallback_used": True,
                "ai_strict_pass": False,
                "risk_note": "AI single-call batch non ha restituito output valido per questo set: applicato fallback euristico.",
            },
        ]
        source_candidates = [{"set_id": row["set_id"]} for row in ranked_rows]

        with (
            patch.object(oracle, "_collect_source_candidates", new=AsyncMock(return_value=source_candidates)),
            patch.object(oracle, "_rank_and_persist_candidates", new=AsyncMock(return_value=ranked_rows)),
        ):
            report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)

        diagnostics = report["diagnostics"]
        self.assertTrue(bool(diagnostics.get("no_signal_due_to_low_strict_pass")))
        self.assertTrue(bool(diagnostics.get("fallback_used")))
        self.assertEqual(report["selected"], [])

    async def test_discovery_keeps_low_conf_signals_when_trust_rate_is_sufficient(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_no_signal_on_low_strict_pass = True
        oracle.ai_no_signal_min_strict_pass_rate = 0.5
        oracle.ai_non_json_trust_weight = 0.6
        oracle._last_source_diagnostics = {
            "source_strategy": "external_first",
            "source_order": ["external_proxy", "playwright"],
            "selected_source": "external_proxy",
            "source_raw_counts": {"lego_proxy_reader": 2, "amazon_proxy_reader": 1},
            "source_dedup_counts": {"lego_proxy_reader": 2, "amazon_proxy_reader": 1},
            "source_failures": [],
            "source_signals": {},
            "dedup_candidates": 3,
            "fallback_source_used": False,
            "fallback_notes": [],
            "anti_bot_alert": False,
            "anti_bot_message": None,
            "root_cause_hint": None,
        }
        ranked_rows = [
            {
                "set_id": "301",
                "source": "lego_proxy_reader",
                "composite_score": 72,
                "forecast_score": 61,
                "forecast_probability_upside_12m": 66.0,
                "confidence_score": 59,
                "market_demand_score": 91,
                "ai_raw_score": 79,
                "ai_shortlisted": True,
                "ai_fallback_used": False,
                "ai_strict_pass": True,
            },
            {
                "set_id": "302",
                "source": "lego_proxy_reader",
                "composite_score": 70,
                "forecast_score": 60,
                "forecast_probability_upside_12m": 65.0,
                "confidence_score": 58,
                "market_demand_score": 90,
                "ai_raw_score": 76,
                "ai_shortlisted": True,
                "ai_fallback_used": False,
                "ai_strict_pass": False,
                "risk_note": "Output AI non JSON: score estratto da testo con parsing robusto.",
            },
            {
                "set_id": "303",
                "source": "amazon_proxy_reader",
                "composite_score": 69,
                "forecast_score": 59,
                "forecast_probability_upside_12m": 64.0,
                "confidence_score": 57,
                "market_demand_score": 88,
                "ai_raw_score": 74,
                "ai_shortlisted": True,
                "ai_fallback_used": False,
                "ai_strict_pass": False,
                "risk_note": "Output AI non JSON: score estratto da testo con parsing robusto.",
            },
            {
                "set_id": "304",
                "source": "amazon_proxy_reader",
                "composite_score": 65,
                "forecast_score": 55,
                "forecast_probability_upside_12m": 61.0,
                "confidence_score": 53,
                "market_demand_score": 85,
                "ai_raw_score": 70,
                "ai_shortlisted": True,
                "ai_fallback_used": True,
                "ai_strict_pass": False,
                "risk_note": "AI single-call batch non ha restituito output valido per questo set: applicato fallback euristico.",
            },
        ]
        source_candidates = [{"set_id": row["set_id"]} for row in ranked_rows]

        with (
            patch.object(oracle, "_collect_source_candidates", new=AsyncMock(return_value=source_candidates)),
            patch.object(oracle, "_rank_and_persist_candidates", new=AsyncMock(return_value=ranked_rows)),
        ):
            report = await oracle.discover_with_diagnostics(persist=False, top_limit=10, fallback_limit=3)

        diagnostics = report["diagnostics"]
        self.assertEqual(int(diagnostics.get("ai_shortlist_strict_pass_count") or 0), 1)
        self.assertAlmostEqual(float(diagnostics.get("strict_pass_rate_shortlist") or 0.0), 0.25, places=4)
        self.assertAlmostEqual(float(diagnostics.get("trust_pass_rate_shortlist") or 0.0), 0.55, places=4)
        self.assertFalse(bool(diagnostics.get("no_signal_due_to_low_strict_pass")))
        self.assertGreater(len(report.get("selected") or []), 0)

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
        oracle._historical_reference_cases = []

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
        oracle._historical_reference_cases = []

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
        self.assertEqual(selected[0]["signal_strength"], "HIGH_CONFIDENCE_STRICT")
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
        oracle.min_composite_score = 1
        oracle.ai_no_signal_on_low_strict_pass = False

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

    def test_gemini_free_tier_name_filter(self) -> None:
        self.assertTrue(DiscoveryOracle._is_gemini_free_tier_model_name("models/gemini-2.0-flash"))
        self.assertTrue(DiscoveryOracle._is_gemini_free_tier_model_name("models/gemini-2.0-flash-lite"))
        self.assertFalse(DiscoveryOracle._is_gemini_free_tier_model_name("models/gemini-2.5-pro"))
        self.assertFalse(DiscoveryOracle._is_gemini_free_tier_model_name("models/gemini-3-ultra"))

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

    def test_openrouter_free_model_detection_requires_suffix_when_enabled(self) -> None:
        zero_pricing_no_suffix = {
            "id": "vendor/model-zero-pricing",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"modality": "text->text"},
        }
        self.assertFalse(
            DiscoveryOracle._is_openrouter_free_model(
                zero_pricing_no_suffix,
                require_suffix_free=True,
            )
        )
        self.assertTrue(
            DiscoveryOracle._is_openrouter_free_model(
                zero_pricing_no_suffix,
                require_suffix_free=False,
            )
        )

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
                "ai_strict_pass": True,
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

    async def test_ai_persisted_non_json_cache_respects_short_ttl(self) -> None:
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
            "ai_investment_score": 78,
            "ai_analysis_summary": "Cache DB non-json",
            "eol_date_prediction": "2026-12-01",
            "last_seen_at": (now - timedelta(hours=10)).isoformat(),
            "metadata": {
                "ai_raw_score": 82,
                "ai_fallback_used": False,
                "ai_confidence": "LOW_CONFIDENCE",
                "ai_strict_pass": False,
                "ai_risk_note": "Output AI non JSON: score estratto da testo con parsing robusto.",
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
                "mock_score": 84,
                "mock_fallback": False,
            }
        ]
        oracle = DummyOracle(repo, candidates)
        oracle.ai_rank_max_candidates = 1
        oracle.ai_cache_ttl_sec = 86400.0
        oracle.ai_persisted_cache_ttl_sec = 172800.0
        oracle.ai_non_json_cache_ttl_sec = 6 * 3600.0

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=5, fallback_limit=1)

        self.assertEqual(oracle.ai_calls, 1)
        self.assertEqual(int(report["diagnostics"]["ranking"].get("ai_persisted_cache_hits") or 0), 0)
        self.assertEqual(int(report["diagnostics"]["ranking"].get("ai_cache_hits") or 0), 0)

    async def test_ai_persisted_cache_requires_strict_pass_metadata(self) -> None:
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
            "ai_analysis_summary": "Cache legacy without strict flag",
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
                "mock_score": 83,
                "mock_fallback": False,
            }
        ]
        oracle = DummyOracle(repo, candidates)
        oracle.ai_rank_max_candidates = 1
        oracle.ai_cache_ttl_sec = 3600.0
        oracle.ai_persisted_cache_ttl_sec = 172800.0

        report = await oracle.discover_with_diagnostics(persist=False, top_limit=5, fallback_limit=1)

        self.assertEqual(oracle.ai_calls, 1)
        self.assertEqual(int(report["diagnostics"]["ranking"].get("ai_persisted_cache_hits") or 0), 0)
        self.assertEqual(int(report["diagnostics"]["ranking"].get("ai_cache_hits") or 0), 0)

    def test_set_cached_ai_insight_skips_non_json_by_default(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_cache_ttl_sec = 86400.0
        oracle.ai_non_json_cache_ttl_sec = 7200.0
        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        insight = AIInsight(
            score=78,
            summary="Non json sample",
            predicted_eol_date="2026-05-01",
            fallback_used=False,
            confidence="LOW_CONFIDENCE",
            risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
        )

        oracle._set_cached_ai_insight(candidate, insight)
        key = oracle._ai_cache_key(candidate)
        self.assertIsNotNone(key)
        self.assertIsNone(oracle._ai_insight_cache.get(str(key)))

    def test_set_cached_ai_insight_allows_non_json_in_emergency_mode(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_cache_ttl_sec = 86400.0
        oracle.ai_non_json_cache_ttl_sec = 7200.0
        candidate = {
            "set_id": "75367",
            "set_name": "LEGO Star Wars",
            "theme": "Star Wars",
            "source": "lego_proxy_reader",
            "current_price": 129.99,
            "eol_date_prediction": "2026-05-01",
        }
        insight = AIInsight(
            score=78,
            summary="Non json sample",
            predicted_eol_date="2026-05-01",
            fallback_used=False,
            confidence="LOW_CONFIDENCE",
            risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
        )

        now_ts = time.time()
        oracle._set_cached_ai_insight(candidate, insight, allow_non_strict=True)
        key = oracle._ai_cache_key(candidate)
        self.assertIsNotNone(key)
        cached = oracle._ai_insight_cache.get(str(key))
        self.assertIsNotNone(cached)
        expires_at, _insight = cached
        self.assertLessEqual(float(expires_at - now_ts), 7205.0)

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

    async def test_top_pick_rescue_can_force_specific_set_ids(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_top_pick_rescue_enabled = True
        oracle.ai_top_pick_rescue_count = 1
        oracle.ai_top_pick_rescue_timeout_sec = 3.0

        candidate_primary = {
            "set_id": "11001",
            "set_name": "Primary",
            "theme": "City",
            "source": "lego_proxy_reader",
            "current_price": 29.99,
            "eol_date_prediction": "2026-05-01",
        }
        candidate_forced = {
            "set_id": "11002",
            "set_name": "Forced",
            "theme": "Icons",
            "source": "lego_proxy_reader",
            "current_price": 99.99,
            "eol_date_prediction": "2026-06-01",
        }

        prepared = [
            {
                "candidate": candidate_primary,
                "set_id": "11001",
                "theme": "City",
                "forecast": oracle.forecaster.forecast(candidate=candidate_primary, history_rows=[], theme_baseline={}),
                "history_30": [],
                "prefilter_score": 90,
                "prefilter_rank": 1,
                "ai_shortlisted": True,
            },
            {
                "candidate": candidate_forced,
                "set_id": "11002",
                "theme": "Icons",
                "forecast": oracle.forecaster.forecast(candidate=candidate_forced, history_rows=[], theme_baseline={}),
                "history_30": [],
                "prefilter_score": 40,
                "prefilter_rank": 8,
                "ai_shortlisted": False,
            },
        ]
        ranked = [
            {"set_id": "11001", "composite_score": 80, "forecast_score": 70, "market_demand_score": 80, "ai_fallback_used": False},
            {"set_id": "11002", "composite_score": 64, "forecast_score": 50, "market_demand_score": 70, "ai_fallback_used": True},
        ]
        ai_results = {
            "11001": AIInsight(
                score=82,
                summary="ok",
                predicted_eol_date="2026-05-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }

        with patch.object(
            oracle,
            "_get_ai_insight",
            new=AsyncMock(
                return_value=AIInsight(
                    score=79,
                    summary="rescued",
                    predicted_eol_date="2026-06-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            ),
        ) as mocked_get:
            stats = await oracle._rescue_top_pick_ai_scores(
                prepared=prepared,
                ranked=ranked,
                ai_results=ai_results,
                force_set_ids=["11002"],
                top_k_override=0,
            )

        self.assertEqual(mocked_get.await_count, 1)
        self.assertEqual(int(stats["ai_top_pick_rescue_attempts"]), 1)
        self.assertEqual(int(stats["ai_top_pick_rescue_successes"]), 1)
        self.assertIn("11002", ai_results)
        self.assertFalse(ai_results["11002"].fallback_used)

    async def test_rank_flow_guarantees_external_ai_attempt_for_final_top_three(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_top_pick_rescue_enabled = True
        oracle.ai_top_pick_rescue_count = 1
        oracle.ai_final_pick_guarantee_count = 3
        oracle.ai_final_pick_guarantee_rounds = 2

        source_candidates = [
            {
                "set_id": "12001",
                "set_name": "Seed candidate",
                "theme": "City",
                "source": "amazon_proxy_reader",
                "current_price": 25.0,
                "eol_date_prediction": "2026-05-01",
                "metadata": {},
            },
            {
                "set_id": "12002",
                "set_name": "Rescue alpha",
                "theme": "Marvel",
                "source": "lego_proxy_reader",
                "current_price": 89.0,
                "eol_date_prediction": "2026-05-10",
                "metadata": {},
            },
            {
                "set_id": "12003",
                "set_name": "Rescue beta",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 109.0,
                "eol_date_prediction": "2026-05-20",
                "metadata": {},
            },
        ]
        prepared: list[dict] = []
        for idx, candidate in enumerate(source_candidates, start=1):
            forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": candidate["set_id"],
                    "theme": candidate["theme"],
                    "forecast": forecast,
                    "history_30": [],
                    "prefilter_score": 100 - (idx * 10),
                    "prefilter_rank": idx,
                    "ai_shortlisted": idx == 1,
                }
            )

        shortlist = [prepared[0]]
        skipped = [prepared[1], prepared[2]]

        async def fake_get_ai_insight(candidate):  # noqa: ANN001
            set_id = str(candidate.get("set_id"))
            return AIInsight(
                score=84 if set_id == "12002" else 81,
                summary=f"rescued {set_id}",
                predicted_eol_date=str(candidate.get("eol_date_prediction")),
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )

        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        initial_ai_results = {
            "12001": AIInsight(
                score=40,
                summary="seed",
                predicted_eol_date="2026-05-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_get_ai_insight", new=AsyncMock(side_effect=fake_get_ai_insight)) as mocked_get,
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        self.assertIn("12002", ranked_by_set)
        self.assertIn("12003", ranked_by_set)
        self.assertFalse(bool(ranked_by_set["12002"].get("ai_fallback_used")))
        self.assertFalse(bool(ranked_by_set["12003"].get("ai_fallback_used")))
        self.assertGreaterEqual(mocked_get.await_count, 2)
        self.assertGreaterEqual(int(oracle._last_ranking_diagnostics.get("ai_top_pick_rescue_attempts") or 0), 2)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_final_pick_guarantee_pending_after_rounds") or 0), 0)

    async def test_rank_flow_guarantees_external_ai_attempt_for_final_top_three_in_single_call_mode(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_enabled = True
        oracle.ai_top_pick_rescue_count = 1
        oracle.ai_final_pick_guarantee_count = 3
        oracle.ai_final_pick_guarantee_rounds = 2

        source_candidates = [
            {
                "set_id": "13001",
                "set_name": "Seed candidate",
                "theme": "City",
                "source": "amazon_proxy_reader",
                "current_price": 25.0,
                "eol_date_prediction": "2026-05-01",
                "metadata": {},
            },
            {
                "set_id": "13002",
                "set_name": "Rescue alpha",
                "theme": "Marvel",
                "source": "lego_proxy_reader",
                "current_price": 89.0,
                "eol_date_prediction": "2026-05-10",
                "metadata": {},
            },
            {
                "set_id": "13003",
                "set_name": "Rescue beta",
                "theme": "Icons",
                "source": "lego_proxy_reader",
                "current_price": 109.0,
                "eol_date_prediction": "2026-05-20",
                "metadata": {},
            },
        ]
        prepared: list[dict] = []
        for idx, candidate in enumerate(source_candidates, start=1):
            forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": candidate["set_id"],
                    "theme": candidate["theme"],
                    "forecast": forecast,
                    "history_30": [],
                    "prefilter_score": 100 - (idx * 10),
                    "prefilter_rank": idx,
                    "ai_shortlisted": idx == 1,
                }
            )

        shortlist = [prepared[0]]
        skipped = [prepared[1], prepared[2]]
        initial_ai_results = {
            "13001": AIInsight(
                score=40,
                summary="seed",
                predicted_eol_date="2026-05-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }
        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }

        async def fake_batch(  # noqa: ANN001
            entries,
            *,
            deadline,
            allow_repair_calls=True,
            allow_failover_call=True,
            output_mode: str = "json_first",
            strict_only: bool = False,
        ):
            _ = (deadline, allow_repair_calls, allow_failover_call, output_mode, strict_only)
            scored = {}
            for entry in entries:
                set_id = str(entry.get("set_id") or "")
                if set_id in {"13002", "13003"}:
                    scored[set_id] = AIInsight(
                        score=83 if set_id == "13002" else 81,
                        summary=f"rescued {set_id}",
                        predicted_eol_date="2026-05-15",
                        fallback_used=False,
                        confidence="HIGH_CONFIDENCE",
                    )
            return scored, None

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_batch)) as mocked_batch,
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        self.assertIn("13002", ranked_by_set)
        self.assertIn("13003", ranked_by_set)
        self.assertFalse(bool(ranked_by_set["13002"].get("ai_fallback_used")))
        self.assertFalse(bool(ranked_by_set["13003"].get("ai_fallback_used")))
        self.assertGreaterEqual(mocked_batch.await_count, 1)
        total_rescue_like_attempts = (
            int(oracle._last_ranking_diagnostics.get("ai_top_pick_rescue_attempts") or 0)
            + int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_attempted") or 0)
        )
        self.assertGreaterEqual(total_rescue_like_attempts, 1)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_final_pick_guarantee_pending_after_rounds") or 0), 0)

    async def test_rank_flow_reuses_persisted_ai_cache_for_skipped_candidates(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_enabled = False
        oracle.ai_single_call_secondary_batch_enabled = False
        oracle.ai_non_shortlist_cache_rescue_enabled = True
        repo.recent_ai_insights["14002"] = {
            "ai_investment_score": 79,
            "ai_analysis_summary": "cached insight",
            "eol_date_prediction": "2026-06-01",
            "metadata": {
                "ai_raw_score": 79,
                "ai_confidence": "HIGH_CONFIDENCE",
                "ai_fallback_used": False,
                "ai_strict_pass": True,
            },
        }

        source_candidates = [
            {"set_id": "14001"},
            {"set_id": "14002"},
            {"set_id": "14003"},
        ]
        prepared = []
        for idx, set_id in enumerate(("14001", "14002", "14003"), start=1):
            candidate = {
                "set_id": set_id,
                "set_name": f"Set {set_id}",
                "theme": "City",
                "source": "lego_proxy_reader",
                "current_price": 39.99 + idx,
                "eol_date_prediction": "2026-10-01",
                "metadata": {},
            }
            forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": set_id,
                    "theme": candidate["theme"],
                    "forecast": forecast,
                    "history_30": [],
                    "prefilter_score": 100 - (idx * 10),
                    "prefilter_rank": idx,
                    "ai_shortlisted": idx == 1,
                }
            )
        shortlist = [prepared[0]]
        skipped = [prepared[1], prepared[2]]
        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        initial_ai_results = {
            "14001": AIInsight(
                score=74,
                summary="primary",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        self.assertFalse(bool(ranked_by_set["14002"].get("ai_fallback_used")))
        self.assertGreaterEqual(int(oracle._last_ranking_diagnostics.get("ai_skip_cache_rescued") or 0), 1)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_attempted") or 0), 0)

    async def test_rank_flow_strict_top_k_only_skips_cache_rescue_for_non_shortlisted(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_enabled = False
        oracle.ai_single_call_secondary_batch_enabled = True
        oracle.ai_non_shortlist_cache_rescue_enabled = True
        oracle.ai_strict_final_top_k_only = True
        oracle.ai_strict_final_top_k = 1
        repo.recent_ai_insights["15002"] = {
            "ai_investment_score": 79,
            "ai_analysis_summary": "cached insight",
            "eol_date_prediction": "2026-06-01",
            "metadata": {
                "ai_raw_score": 79,
                "ai_confidence": "HIGH_CONFIDENCE",
                "ai_fallback_used": False,
                "ai_strict_pass": True,
            },
        }

        source_candidates = [
            {"set_id": "15001"},
            {"set_id": "15002"},
            {"set_id": "15003"},
        ]
        prepared = []
        for idx, set_id in enumerate(("15001", "15002", "15003"), start=1):
            candidate = {
                "set_id": set_id,
                "set_name": f"Set {set_id}",
                "theme": "City",
                "source": "lego_proxy_reader",
                "current_price": 39.99 + idx,
                "eol_date_prediction": "2026-10-01",
                "metadata": {},
            }
            forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": set_id,
                    "theme": candidate["theme"],
                    "forecast": forecast,
                    "history_30": [],
                    "prefilter_score": 100 - (idx * 10),
                    "prefilter_rank": idx,
                    "ai_shortlisted": idx == 1,
                }
            )
        shortlist = [prepared[0]]
        skipped = [prepared[1], prepared[2]]
        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        initial_ai_results = {
            "15001": AIInsight(
                score=74,
                summary="primary",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        self.assertTrue(bool(ranked_by_set["15002"].get("ai_fallback_used")))
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_skip_cache_rescued") or 0), 0)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_attempted") or 0), 0)

    async def test_single_call_scoring_uses_chunked_batches(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_single_call_allow_repair_calls = True
        oracle.ai_single_call_batch_chunk_size = 2
        oracle.ai_single_call_batch_max_calls = 2
        oracle.ai_single_call_missing_rescue_enabled = False
        oracle.ai_scoring_hard_budget_sec = 75.0

        shortlist: list[dict[str, Any]] = []
        for idx in range(1, 6):
            set_id = f"9900{idx}"
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

        async def fake_batch(
            entries: list[dict[str, Any]],
            *,
            deadline: float,
            allow_repair_calls: bool = True,
            allow_failover_call: bool = True,
            output_mode: str = "json_first",
            strict_only: bool = False,
        ):
            _ = (deadline, allow_repair_calls, allow_failover_call, output_mode, strict_only)
            scored = {}
            for entry in entries:
                set_id = str(entry.get("set_id") or "")
                scored[set_id] = AIInsight(
                    score=80,
                    summary=f"strict {set_id}",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            return scored, None

        with patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_batch)) as mocked_batch:
            results, stats = await oracle._score_ai_shortlist(shortlist)

        self.assertEqual(mocked_batch.await_count, 2)
        strict_count = sum(1 for insight in results.values() if not bool(insight.fallback_used))
        fallback_count = sum(1 for insight in results.values() if bool(insight.fallback_used))
        self.assertEqual(strict_count, 4)
        self.assertEqual(fallback_count, 1)
        self.assertEqual(int(stats.get("ai_batch_scored_count") or 0), 4)

    async def test_top_pick_rescue_retries_non_strict_ai_results(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="x")
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_count = 1

        candidate = {
            "set_id": "99701",
            "set_name": "Set 99701",
            "theme": "City",
            "source": "lego_proxy_reader",
            "current_price": 69.99,
            "eol_date_prediction": "2026-10-01",
        }
        prepared = [
            {
                "set_id": "99701",
                "candidate": candidate,
            }
        ]
        ranked = [
            {
                "set_id": "99701",
                "composite_score": 75,
                "ai_investment_score": 75,
                "forecast_score": 60,
                "market_demand_score": 90,
            }
        ]
        ai_results = {
            "99701": AIInsight(
                score=79,
                summary="non json result",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="LOW_CONFIDENCE",
                risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
            )
        }

        async def fake_batch(
            entries: list[dict[str, Any]],
            *,
            deadline: float,
            allow_repair_calls: bool = True,
            allow_failover_call: bool = True,
            output_mode: str = "json_first",
            strict_only: bool = False,
        ):
            _ = (deadline, allow_repair_calls, allow_failover_call, output_mode, strict_only)
            return {
                "99701": AIInsight(
                    score=82,
                    summary="strict rescue",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            }, None

        with patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_batch)):
            stats = await oracle._rescue_top_pick_ai_scores(
                prepared=prepared,
                ranked=ranked,
                ai_results=ai_results,
            )

        rescued = ai_results["99701"]
        self.assertFalse(bool(rescued.fallback_used))
        self.assertNotIn("non json", str(rescued.risk_note or "").lower())
        self.assertGreaterEqual(int(stats.get("ai_top_pick_rescue_attempts") or 0), 1)
        self.assertGreaterEqual(int(stats.get("ai_top_pick_rescue_successes") or 0), 1)

    async def test_rank_flow_secondary_batch_scores_unresolved_skipped_candidates(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_enabled = False
        oracle.ai_non_shortlist_cache_rescue_enabled = False
        oracle.ai_single_call_secondary_batch_enabled = True
        oracle.ai_single_call_secondary_batch_max_candidates = 2
        oracle.ai_single_call_secondary_batch_min_budget_sec = 6.0
        oracle.ai_single_call_secondary_batch_timeout_sec = 10.0

        source_candidates = [
            {"set_id": "15001"},
            {"set_id": "15002"},
            {"set_id": "15003"},
        ]
        prepared = []
        for idx, set_id in enumerate(("15001", "15002", "15003"), start=1):
            candidate = {
                "set_id": set_id,
                "set_name": f"Set {set_id}",
                "theme": "City",
                "source": "lego_proxy_reader",
                "current_price": 49.99 + idx,
                "eol_date_prediction": "2026-09-01",
                "metadata": {},
            }
            forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": set_id,
                    "theme": candidate["theme"],
                    "forecast": forecast,
                    "history_30": [],
                    "prefilter_score": 100 - (idx * 10),
                    "prefilter_rank": idx,
                    "ai_shortlisted": idx == 1,
                }
            )
        shortlist = [prepared[0]]
        skipped = [prepared[1], prepared[2]]
        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        initial_ai_results = {
            "15001": AIInsight(
                score=72,
                summary="primary",
                predicted_eol_date="2026-09-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }

        async def fake_secondary_batch(  # noqa: ANN001
            entries,
            *,
            deadline,
            allow_repair_calls=True,
            allow_failover_call=True,
            output_mode: str = "json_first",
            strict_only: bool = False,
        ):
            _ = (deadline, allow_repair_calls, allow_failover_call, output_mode, strict_only)
            scored = {
                "15002": AIInsight(
                    score=81,
                    summary="secondary",
                    predicted_eol_date="2026-09-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            }
            return scored, None

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_secondary_batch)) as mocked_batch,
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        self.assertFalse(bool(ranked_by_set["15002"].get("ai_fallback_used")))
        self.assertGreaterEqual(mocked_batch.await_count, 1)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_attempted") or 0), 1)
        self.assertGreaterEqual(int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_scored") or 0), 1)

    async def test_rank_flow_secondary_batch_iterates_multiple_rounds(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_enabled = False
        oracle.ai_non_shortlist_cache_rescue_enabled = False
        oracle.ai_single_call_secondary_batch_enabled = True
        oracle.ai_single_call_secondary_batch_max_candidates = 2
        oracle.ai_single_call_secondary_batch_max_rounds = 3
        oracle.ai_single_call_secondary_batch_min_budget_sec = 6.0
        oracle.ai_single_call_secondary_batch_timeout_sec = 10.0
        oracle.ai_scoring_hard_budget_sec = 45.0

        source_candidates = [{"set_id": str(15500 + idx)} for idx in range(1, 6)]
        prepared = []
        for idx, set_id in enumerate(("15501", "15502", "15503", "15504", "15505"), start=1):
            candidate = {
                "set_id": set_id,
                "set_name": f"Set {set_id}",
                "theme": "City",
                "source": "lego_proxy_reader",
                "current_price": 39.99 + idx,
                "eol_date_prediction": "2026-10-01",
                "metadata": {},
            }
            forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": set_id,
                    "theme": candidate["theme"],
                    "forecast": forecast,
                    "history_30": [],
                    "prefilter_score": 100 - (idx * 8),
                    "prefilter_rank": idx,
                    "ai_shortlisted": idx == 1,
                }
            )
        shortlist = [prepared[0]]
        skipped = [prepared[1], prepared[2], prepared[3], prepared[4]]
        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        initial_ai_results = {
            "15501": AIInsight(
                score=71,
                summary="primary",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }

        async def fake_secondary_batch(  # noqa: ANN001
            entries,
            *,
            deadline,
            allow_repair_calls=True,
            allow_failover_call=True,
            output_mode: str = "json_first",
            strict_only: bool = False,
        ):
            _ = (deadline, allow_repair_calls, allow_failover_call, output_mode, strict_only)
            scored = {}
            for entry in entries:
                set_id = str(entry.get("set_id") or "")
                if not set_id:
                    continue
                scored[set_id] = AIInsight(
                    score=70 + int(set_id[-1]),
                    summary=f"secondary {set_id}",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            return scored, None

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_secondary_batch)) as mocked_batch,
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        self.assertEqual(mocked_batch.await_count, 2)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_rounds") or 0), 2)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_candidates") or 0), 4)
        self.assertGreaterEqual(int(oracle._last_ranking_diagnostics.get("ai_secondary_batch_scored") or 0), 4)
        self.assertFalse(bool(ranked_by_set["15502"].get("ai_fallback_used")))
        self.assertFalse(bool(ranked_by_set["15503"].get("ai_fallback_used")))
        self.assertFalse(bool(ranked_by_set["15504"].get("ai_fallback_used")))
        self.assertFalse(bool(ranked_by_set["15505"].get("ai_fallback_used")))

    async def test_rank_flow_secondary_batch_prioritizes_highest_uncovered_candidates(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_enabled = False
        oracle.ai_non_shortlist_cache_rescue_enabled = False
        oracle.ai_single_call_secondary_batch_enabled = True
        oracle.ai_single_call_secondary_batch_max_candidates = 2
        oracle.ai_single_call_secondary_batch_max_rounds = 1
        oracle.ai_single_call_secondary_batch_min_budget_sec = 6.0
        oracle.ai_single_call_secondary_batch_timeout_sec = 10.0
        oracle.ai_scoring_hard_budget_sec = 30.0

        source_candidates = [
            {"set_id": "16001"},
            {"set_id": "16002"},
            {"set_id": "16003"},
            {"set_id": "16004"},
        ]
        prepared = []
        for idx, set_id in enumerate(("16001", "16002", "16003", "16004"), start=1):
            candidate = {
                "set_id": set_id,
                "set_name": f"Set {set_id}",
                "theme": "City",
                "source": "lego_proxy_reader",
                "current_price": 49.99 + idx,
                "eol_date_prediction": "2026-10-01",
                "metadata": {},
            }
            forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": set_id,
                    "theme": candidate["theme"],
                    "forecast": forecast,
                    "history_30": [],
                    "prefilter_score": 80 - idx,
                    "prefilter_rank": idx,
                    "ai_shortlisted": idx == 1,
                }
            )
        shortlist = [prepared[0]]
        # Intentionally shuffled to ensure selection follows priority, not input order.
        skipped = [prepared[1], prepared[3], prepared[2]]
        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        initial_ai_results = {
            "16001": AIInsight(
                score=71,
                summary="primary",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }
        selected_orders = []

        async def fake_secondary_batch(  # noqa: ANN001
            entries,
            *,
            deadline,
            allow_repair_calls=True,
            allow_failover_call=True,
            output_mode: str = "json_first",
            strict_only: bool = False,
        ):
            _ = (deadline, allow_repair_calls, allow_failover_call, output_mode, strict_only)
            selected_orders.append([str(entry.get("set_id") or "") for entry in entries])
            scored = {}
            for entry in entries:
                set_id = str(entry.get("set_id") or "")
                if not set_id:
                    continue
                scored[set_id] = AIInsight(
                    score=77,
                    summary=f"secondary {set_id}",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            return scored, None

        priority_map = {"16002": 10.0, "16003": 99.0, "16004": 70.0}

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_secondary_ai_priority_value", side_effect=lambda row: priority_map[str(row.get("set_id") or "")]),
            patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_secondary_batch)),
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        self.assertEqual(selected_orders[0], ["16003", "16004"])

    async def test_rank_flow_non_json_high_priority_rescore_improves_top_candidate(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.openrouter_api_key = "test-key"
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_top_pick_rescue_enabled = False
        oracle.ai_non_shortlist_cache_rescue_enabled = False
        oracle.ai_single_call_secondary_batch_enabled = False
        oracle.ai_non_json_rescore_enabled = True
        oracle.ai_non_json_rescore_max_candidates = 2
        oracle.ai_non_json_rescore_min_composite = 1
        oracle.ai_non_json_rescore_min_budget_sec = 6.0
        oracle.ai_non_json_rescore_timeout_sec = 10.0
        oracle.ai_scoring_hard_budget_sec = 40.0

        source_candidates = [{"set_id": "16501"}]
        candidate = {
            "set_id": "16501",
            "set_name": "Set 16501",
            "theme": "City",
            "source": "lego_proxy_reader",
            "current_price": 59.99,
            "eol_date_prediction": "2026-10-01",
            "metadata": {},
        }
        forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
        prepared = [
            {
                "candidate": candidate,
                "set_id": "16501",
                "theme": candidate["theme"],
                "forecast": forecast,
                "history_30": [],
                "prefilter_score": 92,
                "prefilter_rank": 1,
                "ai_shortlisted": True,
            }
        ]
        shortlist = [prepared[0]]
        skipped: list[dict] = []
        ai_stats = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        initial_ai_results = {
            "16501": AIInsight(
                score=82,
                summary="non json",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="LOW_CONFIDENCE",
                risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
            )
        }

        async def fake_batch(  # noqa: ANN001
            entries,
            *,
            deadline,
            allow_repair_calls=True,
            allow_failover_call=True,
            output_mode: str = "json_first",
            strict_only: bool = False,
        ):
            _ = (entries, deadline, allow_repair_calls, allow_failover_call, output_mode, strict_only)
            return {
                "16501": AIInsight(
                    score=76,
                    summary="strict json refreshed",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                    risk_note=None,
                )
            }, None

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(oracle, "_score_ai_shortlist", new=AsyncMock(return_value=(initial_ai_results, ai_stats))),
            patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_batch)) as mocked_batch,
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        self.assertGreaterEqual(mocked_batch.await_count, 1)
        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        refreshed = ranked_by_set["16501"]
        self.assertFalse(bool(refreshed.get("ai_fallback_used")))
        self.assertNotIn("non json", str(refreshed.get("risk_note") or "").lower())
        self.assertEqual(str(refreshed.get("ai_source_origin")), "non_json_rescore_fresh")
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_non_json_rescore_attempted") or 0), 1)
        self.assertGreaterEqual(int(oracle._last_ranking_diagnostics.get("ai_non_json_rescore_scored") or 0), 1)
        self.assertGreaterEqual(int(oracle._last_ranking_diagnostics.get("ai_non_json_rescore_improved") or 0), 1)

    def test_build_ranked_payloads_uses_rescue_failure_note_instead_of_prefilter_note(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        candidate = {
            "set_id": "13001",
            "set_name": "Fallback test",
            "theme": "City",
            "source": "lego_proxy_reader",
            "current_price": 49.99,
            "eol_date_prediction": "2026-06-15",
            "metadata": {},
        }
        forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
        prepared = [
            {
                "candidate": candidate,
                "set_id": "13001",
                "theme": "City",
                "forecast": forecast,
                "history_30": [],
                "prefilter_score": 40,
                "prefilter_rank": 7,
                "ai_shortlisted": False,
                "ai_rescue_attempted": True,
                "ai_rescue_failed": True,
                "ai_rescue_reason": "timeout",
            }
        ]
        ranked, _opportunities = oracle._build_ranked_payloads(
            prepared=prepared,
            ai_results={},
            skipped_set_ids={"13001": {"prefilter_rank": 7}},
            shortlist_count=4,
        )
        note = str(ranked[0].get("risk_note") or "")
        self.assertIn("Tentativo AI esterno sui top pick non riuscito", note)
        self.assertNotIn("pre-filter rank", note.lower())

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

    async def test_score_ai_shortlist_batch_strict_only_rejects_non_json_model(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle._openrouter_candidates = ["vendor/model-pro:free"]
        oracle._openrouter_available_candidates = ["vendor/model-pro:free"]
        oracle._openrouter_available_strict_candidates = []
        oracle._openrouter_probe_report = [
            {
                "model": "vendor/model-pro:free",
                "available": True,
                "status": "available",
                "reason": "ok_text_non_json",
            }
        ]
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory_last_resort_non_json",
            "inventory_available": 1,
            "inventory_available_strict": 0,
            "probe_report": list(oracle._openrouter_probe_report),
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

        with patch.object(oracle, "_openrouter_generate") as mocked_generate:
            results, error = await oracle._score_ai_shortlist_batch(
                entries,
                deadline=time.monotonic() + 10.0,
                strict_only=True,
            )

        self.assertEqual(results, {})
        self.assertEqual(error, "strict_model_unavailable")
        mocked_generate.assert_not_called()

    async def test_score_ai_shortlist_batch_parses_non_json_without_repair_call(self) -> None:
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

        raw_text = (
            "1) set_id 75367 -> score 84/100. outlook positivo.\n"
            "2) set_id 76281 -> score 71/100. domanda media."
        )
        with patch.object(oracle, "_openrouter_generate", return_value=raw_text), patch.object(
            oracle,
            "_repair_openrouter_non_json_batch_output",
            new=AsyncMock(return_value={}),
        ) as mocked_repair:
            results, error = await oracle._score_ai_shortlist_batch(
                entries,
                deadline=time.monotonic() + 10.0,
                allow_repair_calls=False,
                allow_failover_call=False,
            )

        self.assertIsNone(error)
        self.assertEqual(set(results.keys()), {"75367", "76281"})
        self.assertEqual(results["75367"].score, 84)
        self.assertEqual(results["76281"].score, 71)
        mocked_repair.assert_not_awaited()

    async def test_score_ai_shortlist_batch_parsed_non_json_uses_repair_when_quality_low(self) -> None:
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
        oracle.ai_no_signal_min_strict_pass_rate = 0.5
        oracle.ai_model_quality_min_samples = 2
        oracle.ai_model_quality_min_trust_rate = 0.8
        oracle.ai_model_quality_max_non_json_rate = 0.25

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

        raw_non_json = (
            "set_id=75367|score=74|summary=non json a|predicted_eol_date=2026-12-01\n"
            "set_id=76281|score=71|summary=non json b|predicted_eol_date=2026-10-01"
        )
        repaired = {
            "75367": AIInsight(
                score=89,
                summary="strict repaired a",
                predicted_eol_date="2026-12-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            ),
            "76281": AIInsight(
                score=84,
                summary="strict repaired b",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            ),
        }

        with patch.object(oracle, "_openrouter_generate", return_value=raw_non_json), patch.object(
            oracle,
            "_repair_openrouter_non_json_batch_output",
            new=AsyncMock(return_value=repaired),
        ) as mocked_repair:
            results, error = await oracle._score_ai_shortlist_batch(
                entries,
                deadline=time.monotonic() + 10.0,
                allow_repair_calls=True,
                allow_failover_call=False,
            )

        self.assertIsNone(error)
        mocked_repair.assert_awaited_once()
        self.assertEqual(results["75367"].score, 89)
        self.assertEqual(results["76281"].score, 84)
        self.assertFalse(DiscoveryOracle._is_non_json_ai_note(results["75367"].risk_note))
        self.assertFalse(DiscoveryOracle._is_non_json_ai_note(results["76281"].risk_note))

    async def test_score_ai_shortlist_batch_quality_failover_switches_model(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle._openrouter_model_id = "vendor/model-a:free"
        oracle._openrouter_candidates = ["vendor/model-a:free", "vendor/model-b:free"]
        oracle._openrouter_available_candidates = ["vendor/model-a:free", "vendor/model-b:free"]
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-a:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 2,
        }
        oracle.ai_no_signal_min_strict_pass_rate = 0.5
        oracle.ai_model_quality_min_samples = 2
        oracle.ai_model_quality_min_trust_rate = 0.8
        oracle.ai_model_quality_max_non_json_rate = 0.4

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

        raw_non_json = (
            "set_id=75367|score=72|summary=non json a|predicted_eol_date=2026-12-01\n"
            "set_id=76281|score=69|summary=non json b|predicted_eol_date=2026-10-01"
        )
        strict_json = (
            '{"results": ['
            '{"set_id":"75367","score":86,"summary":"strict a","predicted_eol_date":"2026-12-01"},'
            '{"set_id":"76281","score":81,"summary":"strict b","predicted_eol_date":"2026-10-01"}'
            "]}"
        )

        async def fake_rotate(reason: str, strict_only: bool = False) -> bool:  # noqa: ARG001
            _ = strict_only
            oracle._openrouter_model_id = "vendor/model-b:free"
            return True

        with patch.object(
            oracle,
            "_openrouter_generate",
            side_effect=[raw_non_json, strict_json],
        ) as mocked_generate, patch.object(
            oracle,
            "_advance_openrouter_model_locked",
            new=AsyncMock(side_effect=fake_rotate),
        ) as mocked_rotate:
            results, error = await oracle._score_ai_shortlist_batch(
                entries,
                deadline=time.monotonic() + 10.0,
                allow_repair_calls=False,
                allow_failover_call=True,
            )

        self.assertIsNone(error)
        self.assertEqual(mocked_generate.call_count, 2)
        mocked_rotate.assert_awaited_once()
        self.assertEqual(results["75367"].score, 86)
        self.assertEqual(results["76281"].score, 81)
        self.assertFalse(bool(results["75367"].fallback_used))
        self.assertFalse(bool(results["76281"].fallback_used))
        self.assertFalse(DiscoveryOracle._is_non_json_ai_note(results["75367"].risk_note))
        self.assertFalse(DiscoveryOracle._is_non_json_ai_note(results["76281"].risk_note))

    async def test_rank_flow_reranks_once_when_shortlist_strict_pass_is_too_low(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle.ai_no_signal_on_low_strict_pass = True
        oracle.ai_no_signal_min_strict_pass_rate = 0.5
        oracle.ai_top_pick_rescue_enabled = False
        oracle.ai_single_call_secondary_batch_enabled = False
        oracle.ai_non_shortlist_cache_rescue_enabled = False
        oracle.ai_non_json_rescore_enabled = False
        oracle._openrouter_model_id = "vendor/model-a:free"
        oracle._openrouter_candidates = ["vendor/model-a:free", "vendor/model-b:free"]
        oracle._openrouter_available_candidates = ["vendor/model-a:free", "vendor/model-b:free"]
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-a:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 2,
        }

        candidate = {
            "set_id": "76281",
            "set_name": "X-Jet",
            "theme": "Marvel",
            "source": "lego_proxy_reader",
            "current_price": 79.99,
            "eol_date_prediction": "2026-10-01",
            "metadata": {},
        }
        forecast = oracle.forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})
        source_candidates = [candidate]
        prepared = [
            {
                "candidate": candidate,
                "set_id": "76281",
                "theme": "Marvel",
                "forecast": forecast,
                "history_30": [],
                "prefilter_score": 95,
                "prefilter_rank": 1,
                "ai_shortlisted": True,
            }
        ]
        shortlist = [prepared[0]]
        skipped: list[dict] = []
        ai_stats_template = {
            "ai_scored_count": 1,
            "ai_batch_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 1,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
        }
        low_strict = {
            "76281": AIInsight(
                score=79,
                summary="fallback score",
                predicted_eol_date="2026-10-01",
                fallback_used=True,
                confidence="LOW_CONFIDENCE",
                risk_note="AI single-call batch non ha restituito output valido per questo set: applicato fallback euristico.",
            )
        }
        strict = {
            "76281": AIInsight(
                score=82,
                summary="strict json",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )
        }

        def rotate_model(reason: str) -> bool:  # noqa: ARG001
            oracle._openrouter_model_id = "vendor/model-b:free"
            oracle.ai_runtime["model"] = "vendor/model-b:free"
            return True

        with (
            patch.object(oracle, "_prepare_quantitative_context", return_value=prepared),
            patch.object(oracle, "_select_ai_shortlist", return_value=(shortlist, skipped)),
            patch.object(
                oracle,
                "_score_ai_shortlist",
                new=AsyncMock(
                    side_effect=[
                        (low_strict, dict(ai_stats_template)),
                        (strict, dict(ai_stats_template)),
                    ]
                ),
            ) as mocked_score,
            patch.object(oracle, "_advance_openrouter_model", side_effect=rotate_model) as mocked_rotate,
            patch.object(oracle, "_is_ai_score_collapse", return_value=False),
        ):
            ranked = await oracle._rank_and_persist_candidates(source_candidates, persist=False)

        self.assertEqual(mocked_score.await_count, 2)
        mocked_rotate.assert_called_once()
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("rerank_attempt") or 0), 1)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_quality_total") or 0), 1)
        self.assertEqual(int(oracle._last_ranking_diagnostics.get("ai_quality_strict") or 0), 1)
        ranked_by_set = {str(row.get("set_id")): row for row in ranked}
        self.assertFalse(bool(ranked_by_set["76281"].get("ai_fallback_used")))
        self.assertNotIn("non json", str(ranked_by_set["76281"].get("risk_note") or "").lower())

    async def test_score_ai_shortlist_single_call_mode_uses_one_batch_for_all_pending(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_batch_scoring_enabled = True
        oracle.ai_scoring_hard_budget_sec = 8.0
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 1,
        }

        shortlist = []
        for set_id in ("75367", "76281", "42182"):
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

        async def fake_batch(entries, **kwargs):  # noqa: ANN001
            _ = kwargs
            out = {}
            for entry in entries:
                sid = str(entry["set_id"])
                out[sid] = AIInsight(
                    score=80,
                    summary=f"ok {sid}",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            return out, None

        with patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_batch)) as mocked_batch, patch.object(
            oracle,
            "_get_ai_insight",
            new=AsyncMock(side_effect=AssertionError("single-call mode should not use per-pick calls")),
        ):
            results, stats = await oracle._score_ai_shortlist(shortlist)

        self.assertEqual(len(results), 3)
        self.assertEqual(int(stats["ai_scored_count"]), 3)
        self.assertEqual(int(stats["ai_batch_scored_count"]), 3)
        self.assertEqual(mocked_batch.await_count, 1)

    async def test_score_ai_shortlist_single_call_mode_attempts_missing_rescue_once(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_single_call_allow_repair_calls = False
        oracle.ai_single_call_missing_rescue_enabled = True
        oracle.ai_single_call_missing_rescue_max_candidates = 2
        oracle.ai_single_call_missing_rescue_timeout_sec = 10.0
        oracle.ai_scoring_hard_budget_sec = 20.0
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory",
            "inventory_available": 1,
        }

        shortlist = []
        for set_id in ("75367", "76281", "42182"):
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

        batch_call_counter = {"n": 0}

        async def fake_batch(entries, **kwargs):  # noqa: ANN001
            batch_call_counter["n"] += 1
            if batch_call_counter["n"] == 1:
                return {
                    "75367": AIInsight(
                        score=80,
                        summary="ok 75367",
                        predicted_eol_date="2026-10-01",
                        fallback_used=False,
                        confidence="HIGH_CONFIDENCE",
                    )
                }, None
            return {
                "76281": AIInsight(
                    score=77,
                    summary="ok 76281",
                    predicted_eol_date="2026-10-01",
                    fallback_used=False,
                    confidence="HIGH_CONFIDENCE",
                )
            }, None

        with patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock(side_effect=fake_batch)) as mocked_batch:
            results, stats = await oracle._score_ai_shortlist(shortlist)

        self.assertEqual(mocked_batch.await_count, 2)
        first_kwargs = mocked_batch.await_args_list[0].kwargs
        second_kwargs = mocked_batch.await_args_list[1].kwargs
        self.assertFalse(bool(first_kwargs.get("allow_repair_calls")))
        self.assertTrue(bool(second_kwargs.get("allow_repair_calls")))
        self.assertFalse(bool(second_kwargs.get("allow_failover_call")))

        self.assertFalse(results["75367"].fallback_used)
        self.assertFalse(results["76281"].fallback_used)
        self.assertTrue(results["42182"].fallback_used)
        self.assertEqual(int(stats["ai_scored_count"]), 2)
        self.assertEqual(int(stats["ai_batch_scored_count"]), 2)
        self.assertEqual(int(stats["ai_single_call_rescue_attempted"]), 1)
        self.assertEqual(int(stats["ai_single_call_rescue_scored"]), 1)
        self.assertEqual(int(stats["ai_budget_exhausted"]), 1)

    async def test_score_ai_shortlist_disables_single_call_when_strict_runtime_unavailable(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_batch_scoring_enabled = True
        oracle.ai_strict_model_required_main_shortlist = True
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle._openrouter_available_candidates = ["vendor/model-pro:free"]
        oracle._openrouter_available_strict_candidates = []
        oracle._openrouter_probe_report = [
            {
                "model": "vendor/model-pro:free",
                "available": True,
                "status": "available",
                "reason": "ok_text_non_json",
            }
        ]
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory_last_resort_non_json",
            "inventory_available": 1,
            "inventory_available_strict": 0,
            "probe_report": list(oracle._openrouter_probe_report),
        }

        shortlist = []
        for set_id in ("75367", "76281"):
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

        async def fake_get_ai_insight(candidate):  # noqa: ANN001
            sid = str(candidate.get("set_id") or "")
            return AIInsight(
                score=81 if sid == "75367" else 76,
                summary=f"ok {sid}",
                predicted_eol_date="2026-10-01",
                fallback_used=False,
                confidence="HIGH_CONFIDENCE",
            )

        with patch.object(oracle, "_score_ai_shortlist_batch", new=AsyncMock()) as mocked_batch, patch.object(
            oracle,
            "_get_ai_insight",
            new=AsyncMock(side_effect=fake_get_ai_insight),
        ) as mocked_get_ai:
            results, stats = await oracle._score_ai_shortlist(shortlist)

        mocked_batch.assert_not_awaited()
        self.assertEqual(mocked_get_ai.await_count, 2)
        self.assertEqual(len(results), 2)
        self.assertFalse(bool(results["75367"].fallback_used))
        self.assertFalse(bool(results["76281"].fallback_used))
        self.assertEqual(int(stats["ai_scored_count"]), 2)
        self.assertEqual(int(stats["ai_batch_scored_count"]), 0)
        self.assertEqual(int(stats["ai_budget_exhausted"]), 0)

    def test_select_ai_shortlist_single_call_applies_cap(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_scoring_enabled = True
        oracle.ai_single_call_max_candidates = 2

        prepared = []
        for idx in range(5):
            prepared.append(
                {
                    "set_id": str(76000 + idx),
                    "candidate": {
                        "set_id": str(76000 + idx),
                        "set_name": f"Set {76000 + idx}",
                        "theme": "City",
                        "source": "lego_proxy_reader",
                        "current_price": 39.99,
                        "eol_date_prediction": "2026-10-01",
                    },
                    "ai_shortlisted": False,
                }
            )

        shortlist, skipped = oracle._select_ai_shortlist(prepared)
        self.assertEqual(len(shortlist), 3)  # minimum safety floor in single-call mode
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(bool(row.get("ai_shortlisted")) for row in shortlist))
        self.assertTrue(all(not bool(row.get("ai_shortlisted")) for row in skipped))

    def test_batch_insights_from_unstructured_text_parses_key_value_rows(self) -> None:
        candidates = [
            {"set_id": "75367", "set_name": "Set A", "theme": "Star Wars", "source": "lego_proxy_reader"},
            {"set_id": "76281", "set_name": "Set B", "theme": "Marvel", "source": "lego_proxy_reader"},
            {"set_id": "42182", "set_name": "Set C", "theme": "Technic", "source": "lego_proxy_reader"},
        ]
        raw_text = (
            "set_id=75367|score=88|summary=alta domanda|predicted_eol_date=2026-12-01\n"
            "set_id=76281|score=74|summary=buon upside|predicted_eol_date=2026-10-01\n"
            "set_id=42182|score=71|summary=liquidita stabile|predicted_eol_date=2026-09-01"
        )
        insights = DiscoveryOracle._batch_insights_from_unstructured_text(raw_text, candidates)
        self.assertEqual(set(insights.keys()), {"75367", "76281", "42182"})
        self.assertEqual(insights["75367"].score, 88)
        self.assertEqual(insights["76281"].score, 74)
        self.assertEqual(insights["42182"].score, 71)

    def test_batch_insights_from_unstructured_text_kv_mode_marks_strict(self) -> None:
        candidates = [
            {"set_id": "75367", "set_name": "Set A", "theme": "Star Wars", "source": "lego_proxy_reader"},
            {"set_id": "76281", "set_name": "Set B", "theme": "Marvel", "source": "lego_proxy_reader"},
        ]
        raw_text = (
            "set_id=75367|score=88|summary=alta domanda|predicted_eol_date=2026-12-01\n"
            "set_id=76281|score=74|summary=buon upside|predicted_eol_date=2026-10-01"
        )
        insights = DiscoveryOracle._batch_insights_from_unstructured_text(
            raw_text,
            candidates,
            treat_key_value_as_strict=True,
        )
        self.assertEqual(set(insights.keys()), {"75367", "76281"})
        self.assertEqual(insights["75367"].confidence, "HIGH_CONFIDENCE")
        self.assertFalse(DiscoveryOracle._is_non_json_ai_note(insights["75367"].risk_note))
        self.assertIn("kv-only", str(insights["75367"].risk_note or "").lower())

    def test_resolve_batch_output_mode_prefers_kv_only_on_probe_non_json(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle.ai_runtime["probe_report"] = [
            {
                "model": "vendor/model-pro:free",
                "status": "available",
                "reason": "ok_text_non_json",
            }
        ]
        mode, reason = oracle._resolve_batch_output_mode(
            provider="openrouter",
            model_name="vendor/model-pro:free",
            candidate_count=5,
        )
        self.assertEqual(mode, "kv_only")
        self.assertEqual(reason, "probe_non_json")

    def test_effective_single_call_batch_plan_degrades_when_kv_only(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_single_call_batch_chunk_size = 8
        oracle.ai_single_call_batch_max_calls = 2
        oracle.ai_single_call_degraded_batch_chunk_size = 4
        oracle.ai_single_call_degraded_batch_max_calls = 3
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle.ai_runtime.update(
            {
                "engine": "openrouter",
                "provider": "openrouter",
                "model": "vendor/model-pro:free",
                "probe_report": [
                    {
                        "model": "vendor/model-pro:free",
                        "status": "available",
                        "reason": "ok_text_non_json",
                    }
                ],
            }
        )
        chunk_size, max_calls, output_mode = oracle._effective_single_call_batch_plan(pending_count=12)
        self.assertEqual(output_mode, "kv_only")
        self.assertEqual(chunk_size, 4)
        self.assertEqual(max_calls, 3)

    def test_batch_insights_from_unstructured_text_parses_tagged_blocks(self) -> None:
        candidates = [
            {"set_id": "75367", "set_name": "Set A", "theme": "Star Wars", "source": "lego_proxy_reader"},
            {"set_id": "76281", "set_name": "Set B", "theme": "Marvel", "source": "lego_proxy_reader"},
        ]
        raw_text = (
            "[SET]\n"
            "set_id: 75367\n"
            "score: 88\n"
            "summary: alta domanda collezionisti\n"
            "predicted_eol_date: 2026-12-01\n"
            "[/SET]\n"
            "[SET set_id=\"76281\"]\n"
            "score=74\n"
            "summary=buon upside su 12 mesi\n"
            "predicted_eol_date=null\n"
            "[/SET]"
        )
        insights = DiscoveryOracle._batch_insights_from_unstructured_text(raw_text, candidates)
        self.assertEqual(set(insights.keys()), {"75367", "76281"})
        self.assertEqual(insights["75367"].score, 88)
        self.assertEqual(insights["76281"].score, 74)
        self.assertFalse(insights["75367"].fallback_used)
        self.assertIn("tagged non json", str(insights["75367"].risk_note or "").lower())

    def test_batch_insights_from_unstructured_text_order_fallback(self) -> None:
        candidates = [
            {"set_id": "75367", "set_name": "Set A", "theme": "Star Wars", "source": "lego_proxy_reader"},
            {"set_id": "76281", "set_name": "Set B", "theme": "Marvel", "source": "lego_proxy_reader"},
        ]
        raw_text = "1) Score 88/100, forte domanda.\n2) Score 74/100, upside moderato."
        insights = DiscoveryOracle._batch_insights_from_unstructured_text(raw_text, candidates)
        self.assertEqual(set(insights.keys()), {"75367", "76281"})
        self.assertEqual(insights["75367"].score, 88)
        self.assertEqual(insights["76281"].score, 74)

    def test_batch_insights_from_unstructured_text_order_fallback_can_parse_partial_cardinality(self) -> None:
        candidates = [
            {"set_id": "75367", "set_name": "Set A", "theme": "Star Wars", "source": "lego_proxy_reader"},
            {"set_id": "76281", "set_name": "Set B", "theme": "Marvel", "source": "lego_proxy_reader"},
            {"set_id": "42182", "set_name": "Set C", "theme": "Technic", "source": "lego_proxy_reader"},
        ]
        raw_text = "1) Score 88/100, forte domanda.\n2) Score 74/100, upside moderato."
        insights = DiscoveryOracle._batch_insights_from_unstructured_text(raw_text, candidates)
        self.assertEqual(set(insights.keys()), {"75367", "76281"})
        self.assertEqual(insights["75367"].score, 88)
        self.assertEqual(insights["76281"].score, 74)
        self.assertNotIn("42182", insights)

    def test_effective_confidence_score_boosts_with_historical_support(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_quality_guard_enabled = True
        oracle._historical_quality_profile = {"degraded": True}

        forecast = ForecastInsight(
            forecast_score=58,
            probability_upside_12m=0.64,
            expected_roi_12m_pct=42.5,
            interval_low_pct=12.0,
            interval_high_pct=58.0,
            target_roi_pct=30.0,
            estimated_months_to_target=8,
            confidence_score=58,
            data_points=40,
            rationale="test",
        )
        historical_prior = {
            "support_confidence": 80,
            "prior_score": 50,
            "effective_sample_size": 18.0,
        }
        ai = AIInsight(score=82, summary="ok", fallback_used=False, confidence="HIGH_CONFIDENCE")
        pattern_eval = PatternEvaluation(
            score=83,
            confidence_score=75,
            summary="Pattern forti",
            signals=[{"code": "retiring_window"}, {"code": "series_completism"}],
            features={},
        )

        effective = oracle._effective_confidence_score(
            forecast=forecast,
            historical_prior=historical_prior,
            ai=ai,
            pattern_eval=pattern_eval,
        )
        self.assertGreaterEqual(effective, 68)

    def test_effective_confidence_score_penalizes_non_json_ai_output(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle._historical_quality_profile = {"degraded": True}

        forecast = ForecastInsight(
            forecast_score=57,
            probability_upside_12m=0.65,
            expected_roi_12m_pct=44.1,
            interval_low_pct=10.0,
            interval_high_pct=56.0,
            target_roi_pct=30.0,
            estimated_months_to_target=8,
            confidence_score=58,
            data_points=40,
            rationale="test",
        )
        historical_prior = {
            "support_confidence": 80,
            "prior_score": 50,
            "effective_sample_size": 18.0,
        }
        pattern_eval = PatternEvaluation(
            score=83,
            confidence_score=75,
            summary="Pattern forti",
            signals=[{"code": "retiring_window"}, {"code": "series_completism"}],
            features={},
        )
        ai_json = AIInsight(score=82, summary="ok", fallback_used=False, confidence="HIGH_CONFIDENCE")
        ai_non_json = AIInsight(
            score=82,
            summary="ok",
            fallback_used=False,
            confidence="LOW_CONFIDENCE",
            risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
        )

        boosted = oracle._effective_confidence_score(
            forecast=forecast,
            historical_prior=historical_prior,
            ai=ai_json,
            pattern_eval=pattern_eval,
        )
        penalized = oracle._effective_confidence_score(
            forecast=forecast,
            historical_prior=historical_prior,
            ai=ai_non_json,
            pattern_eval=pattern_eval,
        )
        self.assertLess(penalized, boosted)

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

    def test_ai_guardrail_caps_non_json_outlier_scores(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_score_guardrail_enabled = True
        oracle.ai_score_soft_cap = 95
        oracle.ai_score_soft_cap_factor = 0.35
        oracle.ai_low_confidence_score_cap = 90
        oracle.ai_non_json_score_cap = 85

        candidate = {
            "set_id": "71486",
            "set_name": "Castello di Nocturnia",
            "eol_date_prediction": "2026-05-27",
        }
        insight = AIInsight(
            score=99,
            summary="output non strutturato",
            predicted_eol_date=None,
            fallback_used=False,
            confidence="LOW_CONFIDENCE",
            risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
        )

        normalized = oracle._normalize_ai_insight(insight, candidate)
        self.assertEqual(normalized.score, 85)
        self.assertEqual(normalized.model_raw_score, 99)
        self.assertIn("AI guardrail applicato", str(normalized.risk_note))

    def test_ai_guardrail_soft_caps_extreme_json_scores(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.ai_score_guardrail_enabled = True
        oracle.ai_score_soft_cap = 95
        oracle.ai_score_soft_cap_factor = 0.35

        candidate = {
            "set_id": "76281",
            "set_name": "X-Jet di X-Men",
            "eol_date_prediction": "2026-05-16",
        }
        insight = AIInsight(
            score=99,
            summary="output json valido",
            fallback_used=False,
            confidence="HIGH_CONFIDENCE",
        )

        normalized = oracle._normalize_ai_insight(insight, candidate)
        self.assertEqual(normalized.score, 96)
        self.assertEqual(normalized.model_raw_score, 99)

    def test_bootstrap_thresholds_can_promote_high_confidence_when_history_is_short(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_high_conf_required = False
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
            "ai_strict_pass": True,
            "composite_score": 71,
            "forecast_probability_upside_12m": 55.6,
            "confidence_score": 52,
            "forecast_data_points": 20,
        }
        long_history_row = {
            "set_id": "77051",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 71,
            "forecast_probability_upside_12m": 55.6,
            "confidence_score": 52,
            "forecast_data_points": 120,
        }

        self.assertTrue(oracle._is_high_confidence_pick(short_history_row))
        self.assertEqual(oracle._high_confidence_signal_strength(short_history_row), "HIGH_CONFIDENCE_BOOTSTRAP")
        self.assertFalse(oracle._is_high_confidence_pick(long_history_row))
        self.assertEqual(oracle._high_confidence_signal_strength(long_history_row), "LOW_CONFIDENCE")

    def test_bootstrap_is_blocked_when_only_non_json_model_is_available(self) -> None:
        repo = FakeRepo()
        with patch.object(DiscoveryOracle, "_initialize_openrouter_runtime", autospec=True):
            oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key="test-key")
        oracle.historical_high_conf_required = False
        oracle.bootstrap_thresholds_enabled = True
        oracle.bootstrap_min_history_points = 45
        oracle.bootstrap_min_upside_probability = 0.52
        oracle.bootstrap_min_confidence_score = 55
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60
        oracle.ai_disable_bootstrap_without_strict_model = True
        oracle.ai_strict_model_required_main_shortlist = True
        oracle._openrouter_model_id = "vendor/model-pro:free"
        oracle._openrouter_available_candidates = ["vendor/model-pro:free"]
        oracle._openrouter_available_strict_candidates = []
        oracle._openrouter_probe_report = [
            {
                "model": "vendor/model-pro:free",
                "available": True,
                "status": "available",
                "reason": "ok_text_non_json",
            }
        ]
        oracle.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": "vendor/model-pro:free",
            "mode": "api_openrouter_inventory_last_resort_non_json",
            "inventory_available": 1,
            "inventory_available_strict": 0,
            "probe_report": list(oracle._openrouter_probe_report),
        }

        row = {
            "set_id": "77051",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 71,
            "forecast_probability_upside_12m": 56.2,
            "confidence_score": 57,
            "forecast_data_points": 20,
        }

        self.assertFalse(oracle._is_high_confidence_pick(row))
        self.assertEqual(oracle._high_confidence_signal_strength(row), "LOW_CONFIDENCE")

    def test_low_confidence_note_mentions_bootstrap_when_active(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_high_conf_required = False
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

    def test_high_confidence_requires_ai_strict_pass(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_high_conf_required = False
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60

        row = {
            "set_id": "77047",
            "ai_fallback_used": False,
            "ai_strict_pass": False,
            "composite_score": 72,
            "forecast_probability_upside_12m": 65.0,
            "confidence_score": 72,
            "forecast_data_points": 60,
        }
        self.assertFalse(oracle._is_high_confidence_pick(row))
        note = oracle._build_low_confidence_note(row)
        self.assertIn("AI strict-pass assente", note)

    def test_adaptive_historical_thresholds_relax_gate_using_reference_distribution(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_quality_guard_enabled = False
        oracle.historical_quality_soft_gate_enabled = False
        oracle.historical_high_conf_required = True
        oracle.historical_high_conf_min_samples = 24
        oracle.historical_high_conf_min_win_rate_pct = 56.0
        oracle.historical_high_conf_min_support_confidence = 50
        oracle.historical_high_conf_min_prior_score = 60
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60
        oracle.adaptive_historical_thresholds_enabled = True
        oracle._historical_reference_cases = []

        themes = ["city", "ninjago", "friends", "technic", "icons", "star wars"]
        for theme_idx, theme in enumerate(themes):
            for idx in range(18):
                oracle._historical_reference_cases.append(
                    {
                        "set_id": str(95000 + (theme_idx * 100) + idx),
                        "theme": theme.title(),
                        "theme_norm": theme,
                        "set_name": f"{theme} case {idx}",
                        "msrp_usd": 40.0 + float(idx),
                        "roi_12m_pct": 18.0 + float((idx + theme_idx) % 6),
                        "win_12m": 1 if (idx % 2) else 0,
                        "source_dataset": "seed",
                        "pattern_tags": "[]",
                    }
                )

        oracle._adaptive_historical_thresholds = oracle._compute_adaptive_historical_thresholds()
        self.assertTrue(bool(oracle._adaptive_historical_thresholds.get("active")))

        row = {
            "set_id": "76281",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 74,
            "forecast_probability_upside_12m": 64.0,
            "confidence_score": 72,
            "forecast_data_points": 120,
            "historical_sample_size": 18,
            "historical_win_rate_12m_pct": 52.0,
            "historical_support_confidence": 60,
            "historical_prior_score": 65,
        }

        self.assertTrue(oracle._is_high_confidence_pick(row))
        self.assertEqual(oracle._high_confidence_signal_strength(row), "HIGH_CONFIDENCE_STRICT")
        oracle.adaptive_historical_thresholds_enabled = False
        self.assertFalse(oracle._is_high_confidence_pick(row))

    def test_high_confidence_requires_historical_evidence(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_quality_guard_enabled = False
        oracle.historical_quality_soft_gate_enabled = False
        oracle.adaptive_historical_thresholds_enabled = False
        oracle.historical_high_conf_required = True
        oracle.historical_high_conf_min_samples = 24
        oracle.historical_high_conf_min_win_rate_pct = 56.0
        oracle.historical_high_conf_min_support_confidence = 50
        oracle.historical_high_conf_min_prior_score = 60
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60

        row = {
            "set_id": "76281",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 74,
            "forecast_probability_upside_12m": 64.0,
            "confidence_score": 72,
            "forecast_data_points": 120,
            "historical_sample_size": 12,
            "historical_win_rate_12m_pct": 75.0,
            "historical_support_confidence": 62,
            "historical_prior_score": 77,
        }

        self.assertFalse(oracle._is_high_confidence_pick(row))
        note = oracle._build_low_confidence_note(row)
        self.assertIn("Evidenza storica insufficiente", note)

    def test_high_confidence_passes_when_historical_evidence_is_strong(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_quality_guard_enabled = False
        oracle.historical_quality_soft_gate_enabled = False
        oracle.adaptive_historical_thresholds_enabled = False
        oracle.historical_high_conf_required = True
        oracle.historical_high_conf_min_samples = 24
        oracle.historical_high_conf_min_win_rate_pct = 56.0
        oracle.historical_high_conf_min_support_confidence = 50
        oracle.historical_high_conf_min_prior_score = 60
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60

        row = {
            "set_id": "76281",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 74,
            "forecast_probability_upside_12m": 64.0,
            "confidence_score": 72,
            "forecast_data_points": 120,
            "historical_sample_size": 36,
            "historical_win_rate_12m_pct": 69.0,
            "historical_support_confidence": 63,
            "historical_prior_score": 76,
        }

        self.assertTrue(oracle._is_high_confidence_pick(row))

    def test_contextual_historical_gate_relaxes_threshold_for_strong_pattern(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_quality_guard_enabled = False
        oracle.historical_quality_soft_gate_enabled = False
        oracle.adaptive_historical_thresholds_enabled = False
        oracle.historical_high_conf_required = True
        oracle.historical_high_conf_min_samples = 24
        oracle.historical_high_conf_min_win_rate_pct = 56.0
        oracle.historical_high_conf_min_support_confidence = 50
        oracle.historical_high_conf_min_prior_score = 60
        oracle.historical_contextual_gate_enabled = True
        oracle.historical_context_strong_pattern_min_score = 75
        oracle.historical_context_max_win_rate_relax_pct = 10.0
        oracle.historical_context_max_support_relax = 6
        oracle.historical_context_max_prior_relax = 10
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60

        row = {
            "set_id": "76281",
            "theme": "Marvel",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 75,
            "forecast_probability_upside_12m": 65.0,
            "confidence_score": 72,
            "forecast_data_points": 120,
            "historical_sample_size": 48,
            "historical_win_rate_12m_pct": 49.0,
            "historical_support_confidence": 72,
            "historical_prior_score": 56,
            "historical_avg_roi_12m_pct": 35.0,
            "pattern_score": 86,
            "pattern_signals": [
                {"code": "exclusive_cult_license", "score": 95, "confidence": 0.88},
                {"code": "series_completism", "score": 85, "confidence": 0.82},
            ],
        }

        self.assertTrue(oracle._is_high_confidence_pick(row))

    def test_contextual_historical_gate_does_not_relax_weak_pattern(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.historical_quality_guard_enabled = False
        oracle.historical_quality_soft_gate_enabled = False
        oracle.adaptive_historical_thresholds_enabled = False
        oracle.historical_high_conf_required = True
        oracle.historical_high_conf_min_samples = 24
        oracle.historical_high_conf_min_win_rate_pct = 56.0
        oracle.historical_high_conf_min_support_confidence = 50
        oracle.historical_high_conf_min_prior_score = 60
        oracle.historical_contextual_gate_enabled = True
        oracle.historical_context_strong_pattern_min_score = 75
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60

        row = {
            "set_id": "77051",
            "theme": "Animal Crossing",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 73,
            "forecast_probability_upside_12m": 64.0,
            "confidence_score": 72,
            "forecast_data_points": 120,
            "historical_sample_size": 44,
            "historical_win_rate_12m_pct": 49.0,
            "historical_support_confidence": 71,
            "historical_prior_score": 56,
            "historical_avg_roi_12m_pct": 33.0,
            "pattern_score": 72,
            "pattern_signals": [
                {"code": "retiring_window", "score": 83, "confidence": 0.74},
            ],
        }

        self.assertFalse(oracle._is_high_confidence_pick(row))
        note = oracle._build_low_confidence_note(row)
        self.assertIn("Win-rate storico 12m sotto soglia", note)

    def test_historical_quality_report_flags_stale_and_generic_seed(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)

        synthetic = []
        for idx in range(30):
            synthetic.append(
                {
                    "set_id": str(99000 + idx),
                    "theme_norm": "star wars" if idx % 2 == 0 else "city",
                    "roi_12m_pct": 5.0,
                    "win_12m": 0,
                    "end_date": "2018-04-01",
                    "pattern_tags": '["general_collectible"]',
                    "pattern_tags_list": ["general_collectible"],
                }
            )

        profile = oracle._evaluate_historical_reference_quality(synthetic)
        self.assertTrue(bool(profile.get("degraded")))
        issues = " ".join(profile.get("issues") or [])
        self.assertIn("seed_datato", issues)
        self.assertIn("pattern_generico", issues)

    def test_historical_quality_soft_gate_allows_zero_sample_when_seed_is_degraded(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.adaptive_historical_thresholds_enabled = False
        oracle.historical_high_conf_required = True
        oracle.historical_quality_guard_enabled = True
        oracle.historical_quality_soft_gate_enabled = True
        oracle._historical_quality_profile = {
            "degraded": True,
            "global_win_rate_pct": 11.5,
        }
        oracle.min_upside_probability = 0.60
        oracle.min_confidence_score = 68
        oracle.min_composite_score = 60

        row = {
            "set_id": "11199",
            "ai_fallback_used": False,
            "ai_strict_pass": True,
            "composite_score": 74,
            "forecast_probability_upside_12m": 66.0,
            "confidence_score": 72,
            "forecast_data_points": 120,
            "historical_sample_size": 0,
            "historical_win_rate_12m_pct": 0.0,
            "historical_support_confidence": 0,
            "historical_prior_score": 0,
        }
        self.assertTrue(oracle._is_high_confidence_pick(row))

        row_low_sample = dict(row)
        row_low_sample["historical_sample_size"] = 7
        self.assertFalse(oracle._is_high_confidence_pick(row_low_sample))

    def test_historical_quality_degraded_relaxes_gate_thresholds(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.adaptive_historical_thresholds_enabled = False
        oracle.historical_quality_guard_enabled = True
        oracle.historical_quality_soft_gate_enabled = True
        oracle.historical_degraded_gate_relax_enabled = True
        oracle.historical_high_conf_min_samples = 24
        oracle.historical_high_conf_min_win_rate_pct = 56.0
        oracle.historical_high_conf_min_support_confidence = 50
        oracle.historical_high_conf_min_prior_score = 60
        oracle._historical_quality_profile = {
            "degraded": True,
            "tier": "degraded",
            "global_win_rate_pct": 11.5,
        }

        samples, win_rate, support, prior, adaptive = oracle._effective_historical_high_confidence_thresholds()
        self.assertFalse(adaptive)
        self.assertEqual(samples, 12)
        self.assertAlmostEqual(win_rate, 14.5, places=2)
        self.assertEqual(support, 40)
        self.assertEqual(prior, 35)

    def test_historical_quality_empty_seed_keeps_conservative_softening(self) -> None:
        repo = FakeRepo()
        oracle = DiscoveryOracle(repo, gemini_api_key=None, openrouter_api_key=None)
        oracle.adaptive_historical_thresholds_enabled = False
        oracle.historical_quality_guard_enabled = True
        oracle.historical_quality_soft_gate_enabled = True
        oracle.historical_degraded_gate_relax_enabled = True
        oracle.historical_high_conf_min_samples = 24
        oracle.historical_high_conf_min_win_rate_pct = 56.0
        oracle.historical_high_conf_min_support_confidence = 50
        oracle.historical_high_conf_min_prior_score = 60
        oracle._historical_quality_profile = {
            "degraded": True,
            "tier": "empty",
            "global_win_rate_pct": 0.0,
        }

        samples, win_rate, support, prior, adaptive = oracle._effective_historical_high_confidence_thresholds()
        self.assertFalse(adaptive)
        self.assertEqual(samples, 12)
        self.assertEqual(win_rate, 16.0)
        self.assertEqual(support, 45)
        self.assertEqual(prior, 50)

    def test_format_exception_for_log_timeout_has_message(self) -> None:
        err_type, err_message = DiscoveryOracle._format_exception_for_log(asyncio.TimeoutError())
        self.assertEqual(err_type, "TimeoutError")
        self.assertTrue(err_message)
        self.assertIn("timeout", err_message.lower())


if __name__ == "__main__":
    unittest.main()
