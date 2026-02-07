from __future__ import annotations

import unittest

from scrapers import BaseStealthScraper


class ScraperHelpersTests(unittest.TestCase):
    def test_extract_set_id(self) -> None:
        self.assertEqual(
            BaseStealthScraper._extract_set_id("LEGO Star Wars 75367", "other"),
            "75367",
        )

    def test_extract_price(self) -> None:
        self.assertEqual(BaseStealthScraper._extract_price("Prezzo 149,99 €"), 149.99)
        self.assertEqual(BaseStealthScraper._extract_price("EUR 99.50"), 99.5)

    def test_normalize_condition(self) -> None:
        self.assertEqual(BaseStealthScraper._normalize_condition("NUOVO SIGILLATO"), "new")
        self.assertEqual(BaseStealthScraper._normalize_condition("usato buono"), "unknown")

    def test_guess_theme(self) -> None:
        self.assertEqual(BaseStealthScraper._guess_theme("LEGO Star Wars X-Wing"), "Star Wars")
        self.assertEqual(BaseStealthScraper._guess_theme("LEGO Botanicals"), "Unknown")


if __name__ == "__main__":
    unittest.main()
