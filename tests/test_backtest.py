from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from backtest import BacktestObservation, OpportunityBacktester


class FakeRepo:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self._opportunities = [
            {
                "set_id": "75367",
                "set_name": "Venator",
                "theme": "Star Wars",
                "source": "lego_proxy_reader",
                "current_price": 100.0,
                "ai_investment_score": 76,
                "metadata": {
                    "composite_score": 76,
                    "forecast_probability_upside_12m": 72.0,
                    "forecast_confidence_score": 78,
                    "ai_fallback_used": False,
                },
                "discovered_at": (now - timedelta(days=200)).isoformat(),
                "last_seen_at": (now - timedelta(days=199)).isoformat(),
            }
        ]
        self._history = [
            {
                "set_id": "75367",
                "platform": "vinted",
                "price": 136.0,
                "recorded_at": (now - timedelta(days=120)).isoformat(),
                "listing_type": "new",
            }
        ]

    def get_recent_opportunities(self, *, days: int, limit: int, include_archived: bool = False):  # noqa: ANN001
        return self._opportunities

    def get_market_history_for_sets(self, set_ids, *, days: int):  # noqa: ANN001
        return self._history


class BacktestTests(unittest.TestCase):
    def test_run_builds_observations(self) -> None:
        repo = FakeRepo()
        backtester = OpportunityBacktester(target_roi_pct=30.0, horizon_days=180, top_k=3)

        report = backtester.run(repository=repo, lookback_days=365)

        self.assertEqual(report.sample_size, 1)
        self.assertAlmostEqual(report.observations[0].realized_roi_pct, 36.0, places=2)
        self.assertTrue(report.observations[0].hit_target)

    def test_evaluate_excludes_ai_fallback(self) -> None:
        backtester = OpportunityBacktester(target_roi_pct=30.0, horizon_days=180, top_k=3)
        observations = [
            BacktestObservation(
                set_id="A",
                discovery_date=datetime(2025, 1, 1, tzinfo=timezone.utc).date(),
                base_price=100.0,
                composite_score=80,
                probability_upside_pct=75.0,
                confidence_score=80,
                ai_fallback_used=False,
                realized_roi_pct=34.0,
                hit_target=True,
            ),
            BacktestObservation(
                set_id="B",
                discovery_date=datetime(2025, 1, 1, tzinfo=timezone.utc).date(),
                base_price=100.0,
                composite_score=82,
                probability_upside_pct=78.0,
                confidence_score=82,
                ai_fallback_used=True,
                realized_roi_pct=45.0,
                hit_target=True,
            ),
        ]

        metrics = backtester.evaluate(
            observations=observations,
            threshold_composite=60,
            threshold_probability=0.60,
            threshold_confidence=68,
        )
        self.assertEqual(metrics.selected_count, 1)
        self.assertAlmostEqual(metrics.precision, 1.0, places=3)

    def test_tune_thresholds_improves_signal_quality(self) -> None:
        backtester = OpportunityBacktester(target_roi_pct=30.0, horizon_days=180, top_k=3)
        rows: list[BacktestObservation] = []
        base_day = datetime(2025, 1, 1, tzinfo=timezone.utc).date()

        for idx in range(20):
            rows.append(
                BacktestObservation(
                    set_id=f"G{idx}",
                    discovery_date=base_day + timedelta(days=idx // 3),
                    base_price=100.0,
                    composite_score=76,
                    probability_upside_pct=72.0,
                    confidence_score=76,
                    ai_fallback_used=False,
                    realized_roi_pct=38.0,
                    hit_target=True,
                )
            )

        for idx in range(18):
            rows.append(
                BacktestObservation(
                    set_id=f"N{idx}",
                    discovery_date=base_day + timedelta(days=idx // 4),
                    base_price=100.0,
                    composite_score=52,
                    probability_upside_pct=46.0,
                    confidence_score=52,
                    ai_fallback_used=False,
                    realized_roi_pct=8.0,
                    hit_target=False,
                )
            )

        suggestion = backtester.tune_thresholds(
            observations=rows,
            current_composite=50,
            current_probability=0.45,
            current_confidence=50,
            min_selected=15,
        )
        baseline = backtester.evaluate(
            observations=rows,
            threshold_composite=50,
            threshold_probability=0.45,
            threshold_confidence=50,
        )
        self.assertGreaterEqual(suggestion.metrics.selected_count, 15)
        self.assertGreaterEqual(suggestion.metrics.precision, baseline.precision)
        self.assertGreaterEqual(suggestion.metrics.precision_at_k, baseline.precision_at_k)
        self.assertGreaterEqual(suggestion.objective_score, backtester._objective(baseline))


if __name__ == "__main__":
    unittest.main()
