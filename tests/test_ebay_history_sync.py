from __future__ import annotations

import importlib.util
import tempfile
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

    def test_build_case_rows_respects_market_cap(self) -> None:
        class FakeClient:
            def fetch_sold_prices(self, *, market: str, query: str):  # noqa: ANN001
                _ = query
                if market == "IT":
                    return [80.0, 82.0, 85.0, 88.0]
                if market == "DE":
                    return [90.0, 91.0, 92.0, 93.0]
                return []

        rows = self.mod._build_case_rows(
            targets=[{"set_id": "76281", "set_name": "X-Jet", "theme": "Marvel", "msrp_hint": 70.0}],
            markets=["IT", "DE"],
            client=FakeClient(),
            min_sold_listings=4,
            target_roi_pct=20.0,
            max_markets_per_set=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_country"], "IT")

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

    def test_build_vinted_case_rows_respects_market_cap(self) -> None:
        class FakeVintedClient:
            def fetch_listing_prices(self, *, market: str, query: str, require_new: bool = True):  # noqa: ANN001
                _ = query
                _ = require_new
                if market == "IT":
                    return [70.0, 72.0, 75.0]
                if market == "DE":
                    return [68.0, 73.0, 79.0]
                return []

        rows = self.mod._build_vinted_case_rows(
            targets=[{"set_id": "76281", "set_name": "X-Jet", "theme": "Marvel", "msrp_hint": 60.0}],
            markets=["IT", "DE"],
            client=FakeVintedClient(),
            min_listings=3,
            target_roi_pct=20.0,
            max_markets_per_set=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_country"], "IT")

    def test_merge_reference_rows_upserts_without_losing_old_history(self) -> None:
        existing = [
            {
                "set_id": "76281",
                "set_number": "76281-1",
                "set_name": "X-Jet",
                "theme": "Marvel",
                "source_dataset": "ebay_sold_it_90d",
                "market_country": "IT",
                "end_date": "2026-02-01",
                "roi_12m_pct": "10.0000",
                "sold_listing_count": "4",
            },
            {
                "set_id": "76281",
                "set_number": "76281-1",
                "set_name": "X-Jet",
                "theme": "Marvel",
                "source_dataset": "ebay_sold_it_90d",
                "market_country": "IT",
                "end_date": "2026-01-25",
                "roi_12m_pct": "8.0000",
                "sold_listing_count": "4",
            },
        ]
        incoming = [
            {
                "set_id": "76281",
                "set_number": "76281-1",
                "set_name": "X-Jet",
                "theme": "Marvel",
                "source_dataset": "ebay_sold_it_90d",
                "market_country": "IT",
                "end_date": "2026-02-01",
                "roi_12m_pct": "12.0000",
                "sold_listing_count": "6",
            },
            {
                "set_id": "76441",
                "set_number": "76441-1",
                "set_name": "Club dei Duellanti",
                "theme": "Harry Potter",
                "source_dataset": "vinted_active_it_30d",
                "market_country": "IT",
                "end_date": "2026-02-01",
                "roi_12m_pct": "15.0000",
                "sold_listing_count": "5",
            },
        ]

        merged, stats = self.mod.merge_reference_rows(existing, incoming)
        keys = {
            (
                row["set_id"],
                row["source_dataset"],
                row["market_country"],
                row["end_date"],
            )
            for row in merged
        }
        self.assertEqual(len(merged), 3)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["unchanged"], 0)
        self.assertIn(("76281", "ebay_sold_it_90d", "IT", "2026-01-25"), keys)
        self.assertIn(("76281", "ebay_sold_it_90d", "IT", "2026-02-01"), keys)
        self.assertIn(("76441", "vinted_active_it_30d", "IT", "2026-02-01"), keys)
        latest = next(
            row
            for row in merged
            if row["set_id"] == "76281" and row["end_date"] == "2026-02-01"
        )
        self.assertEqual(latest["roi_12m_pct"], "12.0000")
        self.assertEqual(latest["sold_listing_count"], "6")

    def test_load_and_write_rows_roundtrip_and_dedup_existing(self) -> None:
        rows = [
            {
                "set_id": "40220",
                "set_number": "40220-1",
                "set_name": "Autobus",
                "theme": "Icons",
                "source_dataset": "ebay_sold_it_90d",
                "market_country": "IT",
                "end_date": "2026-02-03",
                "roi_12m_pct": "9.1000",
                "sold_listing_count": "4",
            },
            # Duplicate same key: last one should win.
            {
                "set_id": "40220",
                "set_number": "40220-1",
                "set_name": "Autobus",
                "theme": "Icons",
                "source_dataset": "ebay_sold_it_90d",
                "market_country": "IT",
                "end_date": "2026-02-03",
                "roi_12m_pct": "11.5000",
                "sold_listing_count": "6",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "seed.csv"
            self.mod.write_rows(csv_path, rows)
            loaded = self.mod.load_existing_rows(csv_path)
            merged, stats = self.mod.merge_reference_rows(loaded, [])

        self.assertEqual(len(merged), 1)
        self.assertEqual(stats["dropped_existing_duplicates"], 1)
        self.assertEqual(merged[0]["roi_12m_pct"], "11.5000")
        self.assertEqual(merged[0]["sold_listing_count"], "6")


if __name__ == "__main__":
    unittest.main()
