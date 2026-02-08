from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "sync_ebay_sold_history.py"
    spec = importlib.util.spec_from_file_location("sync_ebay_sold_history", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load sync_ebay_sold_history module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EbayHistorySyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_module()

    def test_extract_sold_prices_from_html(self) -> None:
        html = """
        <ul>
          <li class="s-item">
            <span class="s-item__title">LEGO 76281 X-Jet di X-Men</span>
            <span class="s-item__price">EUR 79,99</span>
          </li>
          <li class="s-item">
            <span class="s-item__title">Shop on eBay</span>
            <span class="s-item__price">EUR 10,00</span>
          </li>
          <li class="s-item">
            <span class="s-item__title">LEGO 42182 Rover NASA</span>
            <span class="s-item__price">€ 149,90</span>
          </li>
        </ul>
        """
        prices = self.mod.extract_sold_prices_from_html(html)
        self.assertEqual(prices, [79.99, 149.9])

    def test_build_search_url_includes_completed_and_sold_filters(self) -> None:
        url = self.mod._build_search_url("https://www.ebay.it", "76281 x-jet")
        self.assertIn("LH_Complete=1", url)
        self.assertIn("LH_Sold=1", url)
        self.assertIn("LH_ItemCondition=1000", url)
        self.assertIn("ebay.it/sch/i.html", url)

    def test_build_case_rows_produces_it_and_eu_rows(self) -> None:
        class FakeClient:
            def fetch_sold_prices(self, *, market: str, query: str):  # noqa: ANN001
                if market == "IT":
                    return [90.0, 100.0, 110.0, 120.0]
                if market == "DE":
                    return [95.0, 105.0, 115.0, 125.0]
                return []

        targets = [
            {
                "set_id": "76281",
                "set_name": "X-Jet di X-Men",
                "theme": "Marvel",
                "msrp_hint": 89.99,
            }
        ]
        rows = self.mod._build_case_rows(
            targets=targets,
            markets=["IT", "DE"],
            client=FakeClient(),
            min_sold_listings=4,
            target_roi_pct=20.0,
        )
        self.assertEqual(len(rows), 2)
        by_country = {row["market_country"]: row for row in rows}
        self.assertIn("IT", by_country)
        self.assertIn("DE", by_country)
        self.assertEqual(by_country["IT"]["market_region"], "EU")
        self.assertTrue(float(by_country["IT"]["roi_12m_pct"]) > 0.0)

    def test_build_case_rows_uses_progressive_query_variants(self) -> None:
        calls = []

        class FakeClient:
            def fetch_sold_prices(self, *, market: str, query: str):  # noqa: ANN001
                calls.append((market, query))
                if market == "IT" and query.startswith("77051 In volo"):
                    return [39.0, 42.0, 45.0, 47.0]
                return []

        targets = [
            {
                "set_id": "77051",
                "set_name": "In volo con la Dodo Airlines",
                "theme": "Animal Crossing",
                "msrp_hint": 34.99,
            }
        ]
        rows = self.mod._build_case_rows(
            targets=targets,
            markets=["IT"],
            client=FakeClient(),
            min_sold_listings=4,
            target_roi_pct=20.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], ("IT", "77051"))
        self.assertTrue(any(query.startswith("77051 In volo") for _, query in calls[1:]))


if __name__ == "__main__":
    unittest.main()
