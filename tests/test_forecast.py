from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from forecast import InvestmentForecaster


class ForecastTests(unittest.TestCase):
    def test_forecast_rich_history_produces_higher_confidence(self) -> None:
        forecaster = InvestmentForecaster(target_roi_pct=30.0)
        now = datetime.now(timezone.utc)
        history = [
            {
                "price": 95.0 + (idx % 7),
                "platform": "vinted" if idx % 2 == 0 else "subito",
                "recorded_at": (now - timedelta(days=idx)).isoformat(),
            }
            for idx in range(45)
        ]
        candidate = {
            "set_id": "10332",
            "set_name": "Piazza della citta medievale",
            "theme": "Icons",
            "source": "lego_proxy_reader",
            "current_price": 89.99,
            "eol_date_prediction": (now.date() + timedelta(days=45)).isoformat(),
        }
        theme_baseline = {
            "sample_size": 36.0,
            "avg_ai_score": 76.0,
            "avg_market_demand": 81.0,
            "std_ai_score": 5.0,
        }

        insight = forecaster.forecast(candidate=candidate, history_rows=history, theme_baseline=theme_baseline)

        self.assertGreaterEqual(insight.confidence_score, 70)
        self.assertGreater(insight.probability_upside_12m, 0.45)
        self.assertGreater(insight.forecast_score, 50)
        self.assertIsNotNone(insight.estimated_months_to_target)
        self.assertGreater(insight.interval_high_pct, insight.interval_low_pct)

    def test_forecast_sparse_history_drops_confidence(self) -> None:
        forecaster = InvestmentForecaster(target_roi_pct=30.0)
        now = datetime.now(timezone.utc)
        history = [
            {
                "price": 52.0,
                "platform": "amazon",
                "recorded_at": (now - timedelta(days=24)).isoformat(),
            }
        ]
        candidate = {
            "set_id": "60316",
            "set_name": "City Police",
            "theme": "City",
            "source": "amazon_bestsellers",
            "current_price": 49.99,
            "eol_date_prediction": None,
        }

        insight = forecaster.forecast(candidate=candidate, history_rows=history, theme_baseline={})

        self.assertLess(insight.confidence_score, 70)
        self.assertGreaterEqual(insight.forecast_score, 1)
        self.assertLessEqual(insight.forecast_score, 100)

    def test_forecast_handles_empty_history(self) -> None:
        forecaster = InvestmentForecaster(target_roi_pct=35.0)
        candidate = {
            "set_id": "40747",
            "set_name": "Narcisi",
            "theme": "Icons",
            "source": "lego_proxy_reader",
            "current_price": 14.99,
            "eol_date_prediction": None,
        }

        insight = forecaster.forecast(candidate=candidate, history_rows=[], theme_baseline={})

        self.assertEqual(insight.data_points, 0)
        self.assertGreaterEqual(insight.probability_upside_12m, 0.02)
        self.assertLessEqual(insight.probability_upside_12m, 0.98)
        self.assertIn("Confidenza dati", insight.rationale)


if __name__ == "__main__":
    unittest.main()
