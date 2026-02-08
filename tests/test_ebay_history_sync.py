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

    def test_extract_sold_prices_from_html_supports_s_card_dom(self) -> None:
        html = """
        <ul class="srp-river-results">
          <li class="s-card s-card--horizontal s-card--overflow" id="item1">
            <div class="su-card-container__content">
              <div class="s-card__title">LEGO Set 76281 - Marvel X-Men X-Jet - Scatola Nuova Sigillata</div>
              <div>EUR 53,28</div>
              <div>+EUR 18,44 di spese di spedizione stimate</div>
            </div>
          </li>
          <li class="s-card s-card--horizontal" id="item2">
            <div class="su-card-container__content">
              <div class="s-card__title">Shop on eBay</div>
              <div>EUR 9,99</div>
            </div>
          </li>
        </ul>
        """
        prices = self.mod.extract_sold_prices_from_html(html)
        self.assertEqual(prices, [53.28])

    def test_extract_vinted_listing_prices_from_html(self) -> None:
        html = """
        <div class="new-item-box__container">
          <a href="https://www.vinted.it/items/7868717435-set-lego-x-men-ref-76281?referrer=catalog"
             title="Set LEGO X-Men - Ref. 76281, brand: LEGO, condizioni: Nuovo con cartellino, taglia: Taglia unica, €65.00, €68.95 include la Protezione acquisti">
          </a>
        </div>
        <div class="new-item-box__container">
          <a href="https://www.vinted.it/items/7868719999-lego-used-76281?referrer=catalog"
             title="LEGO 76281 usato, €40.00">
          </a>
        </div>
        """
        prices_new = self.mod.extract_vinted_listing_prices_from_html(html, require_new=True)
        prices_all = self.mod.extract_vinted_listing_prices_from_html(html, require_new=False)
        self.assertEqual(prices_new, [65.0])
        self.assertEqual(prices_all, [65.0, 40.0])

    def test_build_search_url_includes_completed_and_sold_filters(self) -> None:
        url = self.mod._build_search_url("https://www.ebay.it", "76281 x-jet")
        self.assertIn("LH_Complete=1", url)
        self.assertIn("LH_Sold=1", url)
        self.assertIn("LH_ItemCondition=1000", url)
        self.assertIn("ebay.it/sch/i.html", url)

    def test_build_vinted_search_url(self) -> None:
        url = self.mod._build_vinted_search_url("https://www.vinted.it", "76281 x-jet")
        self.assertIn("vinted.it/catalog", url)
        self.assertIn("search_text=", url)
        self.assertIn("order=newest_first", url)

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

    def test_build_vinted_case_rows_produces_rows(self) -> None:
        class FakeVintedClient:
            def fetch_listing_prices(self, *, market: str, query: str, require_new: bool = True):  # noqa: ANN001
                _ = query
                _ = require_new
                if market == "IT":
                    return [70.0, 72.0, 75.0]
                if market == "DE":
                    return [68.0, 73.0, 79.0]
                return []

        targets = [
            {
                "set_id": "76281",
                "set_name": "X-Jet di X-Men",
                "theme": "Marvel",
                "msrp_hint": 89.99,
            }
        ]
        rows = self.mod._build_vinted_case_rows(
            targets=targets,
            markets=["IT", "DE"],
            client=FakeVintedClient(),
            min_listings=3,
            target_roi_pct=20.0,
        )
        self.assertEqual(len(rows), 2)
        by_country = {row["market_country"]: row for row in rows}
        self.assertEqual(by_country["IT"]["source_dataset"], "vinted_active_it_30d")
        self.assertEqual(by_country["DE"]["source_dataset"], "vinted_active_de_30d")
        self.assertEqual(by_country["IT"]["market_region"], "EU")
        self.assertEqual(by_country["IT"]["win_12m"], 0)


if __name__ == "__main__":
    unittest.main()
