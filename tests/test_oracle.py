from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from oracle import AIInsight, DiscoveryOracle
from scrapers import MarketListing


class FakeRepo:
    def __init__(self) -> None:
        self.upserted = []
        self.snapshots = []
        self.recent = {}

    def upsert_opportunity(self, record):  # noqa: ANN001
        self.upserted.append(record)
        return {"ok": True}

    def insert_market_snapshot(self, record):  # noqa: ANN001
        self.snapshots.append(record)
        return {"ok": True}

    def get_recent_market_prices(self, set_id, days=30, platform=None):  # noqa: ANN001
        return self.recent.get(set_id, [])


class DummyOracle(DiscoveryOracle):
    def __init__(self, repository, candidates):  # noqa: ANN001
        super().__init__(repository, gemini_api_key=None, min_ai_score=60)
        self._candidates = candidates

    async def _collect_source_candidates(self):
        return self._candidates

    async def _get_ai_insight(self, candidate):  # noqa: ANN001
        return AIInsight(
            score=int(candidate.get("mock_score", 50)),
            summary="mock summary",
            predicted_eol_date=candidate.get("eol_date_prediction"),
        )


class OracleTests(unittest.IsolatedAsyncioTestCase):
    async def test_discover_opportunities_filters_by_min_score(self) -> None:
        repo = FakeRepo()
        repo.recent["75367"] = [{"price": 110}] * 4

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

    def test_extract_json_from_wrapped_text(self) -> None:
        raw = "Risposta:\n{\"score\": 77, \"summary\": \"ok\"}\nfine"
        payload = DiscoveryOracle._extract_json(raw)

        self.assertEqual(payload["score"], 77)
        self.assertEqual(payload["summary"], "ok")


if __name__ == "__main__":
    unittest.main()
