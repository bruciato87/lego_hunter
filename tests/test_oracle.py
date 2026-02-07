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

    def test_extract_json_from_wrapped_text(self) -> None:
        raw = "Risposta:\n{\"score\": 77, \"summary\": \"ok\"}\nfine"
        payload = DiscoveryOracle._extract_json(raw)

        self.assertEqual(payload["score"], 77)
        self.assertEqual(payload["summary"], "ok")


if __name__ == "__main__":
    unittest.main()
