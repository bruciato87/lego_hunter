from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote_plus

try:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright
except Exception:  # noqa: BLE001
    Browser = Any
    BrowserContext = Any
    Page = Any
    Playwright = Any
    async_playwright = None

try:
    from playwright_stealth import stealth_async
except Exception:  # noqa: BLE001

    async def stealth_async(_: Page) -> None:
        return


LOGGER = logging.getLogger(__name__)
PLAYWRIGHT_AVAILABLE = async_playwright is not None

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

NEW_KEYWORDS = ("nuovo", "sigillato", "new", "sealed", "misb")
SET_ID_RE = re.compile(r"\b(\d{4,6})\b")
PRICE_RE = re.compile(
    r"(?:€|eur)\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"|(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?:€|eur)",
    re.IGNORECASE,
)


@dataclass
class MarketListing:
    platform: str
    set_id: str
    set_name: str
    price: float
    listing_url: str
    condition: str = "unknown"
    seller_rating: Optional[float] = None
    source_note: Optional[str] = None


class BaseStealthScraper:
    def __init__(
        self,
        *,
        headless: bool = True,
        max_retries: int = 3,
        timeout_ms: int = 45000,
    ) -> None:
        self.headless = headless
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "BaseStealthScraper":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def start(self) -> None:
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. Install with: pip install playwright && playwright install chromium"
            )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": random.choice([1366, 1440, 1536]), "height": random.choice([768, 900, 864])},
            locale="it-IT",
        )

    async def close(self) -> None:
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

        self._context = None
        self._browser = None
        self._playwright = None

    async def _new_page(self) -> Page:
        if self._context is None:
            raise RuntimeError("Scraper context is not initialized")
        page = await self._context.new_page()
        page.set_default_timeout(self.timeout_ms)
        await stealth_async(page)
        return page

    async def _human_jitter(self, low: float = 2.0, high: float = 6.0) -> None:
        await asyncio.sleep(random.uniform(low, high))

    async def _with_retry(self, label: str, coro_factory):
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await coro_factory()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = (2 ** (attempt - 1)) + random.uniform(0.8, 2.2)
                LOGGER.warning("%s failed (%s/%s): %s. Retry in %.2fs", label, attempt, self.max_retries, exc, delay)
                await asyncio.sleep(delay)
        raise RuntimeError(f"{label} failed after {self.max_retries} retries") from last_exc

    @staticmethod
    def _extract_set_id(*texts: Optional[str]) -> Optional[str]:
        for text in texts:
            if not text:
                continue
            match = SET_ID_RE.search(text)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_price(*texts: Optional[str]) -> Optional[float]:
        for text in texts:
            if not text:
                continue
            match = PRICE_RE.search(text.replace("\u00a0", " "))
            if not match:
                continue
            raw = (match.group(1) or match.group(2) or "").replace(" ", "")
            if "," in raw and "." in raw:
                # Keep the right-most separator as decimal separator.
                if raw.rfind(",") > raw.rfind("."):
                    raw = raw.replace(".", "").replace(",", ".")
                else:
                    raw = raw.replace(",", "")
            elif "," in raw:
                raw = raw.replace(",", ".")
            try:
                return float(raw)
            except ValueError:
                continue
        return None

    @staticmethod
    def _normalize_condition(*texts: Optional[str]) -> str:
        merged = " ".join(filter(None, texts)).lower()
        if any(keyword in merged for keyword in NEW_KEYWORDS):
            return "new"
        return "unknown"

    @staticmethod
    def _guess_theme(name: str) -> str:
        lowered = name.lower()
        keyword_map = [
            ("Star Wars", ("star wars", "guerre stellari")),
            ("Technic", ("technic", "ingranaggi", "escavatore", "gru")),
            ("City", ("city", "citta", "polizia", "vigili del fuoco", "ambulanza")),
            ("Icons", ("icons", "creator expert", "modular", "medieval", "castello")),
            ("Botanicals", ("botanical", "botanicals", "narcisi", "fiori", "bouquet", "rose", "orchidea")),
            ("Harry Potter", ("harry potter", "hogwarts")),
            ("Marvel", ("marvel", "avengers", "spider-man", "spiderman")),
            ("Ninjago", ("ninjago",)),
            ("Friends", ("friends",)),
            ("Architecture", ("architecture",)),
        ]
        for theme, keywords in keyword_map:
            if any(keyword in lowered for keyword in keywords):
                return theme
        return "Unknown"


class LegoRetiringScraper(BaseStealthScraper):
    LEGO_RETIRING_URL = "https://www.lego.com/it-it/categories/retiring-soon"

    async def fetch_retiring_sets(self, limit: int = 40) -> list[Dict[str, Any]]:
        async def _run() -> list[Dict[str, Any]]:
            page = await self._new_page()
            try:
                await page.goto(self.LEGO_RETIRING_URL, wait_until="domcontentloaded")
                await self._human_jitter()

                cards: list[Dict[str, str]] = await page.evaluate(
                    """
                    () => {
                      const anchors = Array.from(document.querySelectorAll('a[href*="/product/"]'));
                      const seen = new Set();
                      const out = [];

                      for (const anchor of anchors) {
                        const href = anchor.href || '';
                        if (!href || seen.has(href)) continue;
                        seen.add(href);

                        const card = anchor.closest('article,li,section,div');
                        const text = (anchor.textContent || '').trim();
                        const blob = card ? (card.innerText || '') : text;

                        if (!text || text.length < 4) continue;
                        out.push({ href, name: text, blob });
                      }
                      return out;
                    }
                    """
                )

                results: list[Dict[str, Any]] = []
                for item in cards:
                    href = item.get("href") or ""
                    name = (item.get("name") or "").strip()
                    blob = item.get("blob") or ""

                    set_id = self._extract_set_id(href, name, blob)
                    if not set_id:
                        continue

                    price = self._extract_price(blob)
                    results.append(
                        {
                            "set_id": set_id,
                            "set_name": name,
                            "theme": self._guess_theme(name),
                            "source": "lego_retiring",
                            "current_price": price,
                            "eol_date_prediction": (date.today() + timedelta(days=75)).isoformat(),
                            "listing_url": href,
                            "metadata": {"raw_blob": blob[:1000]},
                        }
                    )

                dedup: dict[str, Dict[str, Any]] = {}
                for row in results:
                    dedup[row["set_id"]] = row
                return list(dedup.values())[:limit]
            finally:
                await page.close()

        return await self._with_retry("fetch_retiring_sets", _run)


class AmazonBestsellerScraper(BaseStealthScraper):
    AMAZON_BESTSELLERS_URL = "https://www.amazon.it/gp/bestsellers/toys/635019031"

    async def fetch_bestsellers(self, limit: int = 40) -> list[Dict[str, Any]]:
        async def _run() -> list[Dict[str, Any]]:
            page = await self._new_page()
            try:
                await page.goto(self.AMAZON_BESTSELLERS_URL, wait_until="domcontentloaded")
                await self._human_jitter()

                cards: list[Dict[str, str]] = await page.evaluate(
                    """
                    () => {
                      const links = Array.from(document.querySelectorAll('a[href*="/dp/"]'));
                      const out = [];
                      const seen = new Set();

                      for (const link of links) {
                        const href = link.href || '';
                        if (!href || seen.has(href)) continue;
                        seen.add(href);

                        const card = link.closest('div,li,article');
                        const text = (link.textContent || '').trim();
                        const blob = card ? (card.innerText || '') : text;

                        if (!text || text.length < 5) continue;
                        out.push({ href, name: text, blob });
                      }
                      return out;
                    }
                    """
                )

                results: list[Dict[str, Any]] = []
                for item in cards:
                    name = (item.get("name") or "").strip()
                    if "lego" not in name.lower():
                        continue

                    href = item.get("href") or ""
                    blob = item.get("blob") or ""
                    set_id = self._extract_set_id(name, blob, href)
                    if not set_id:
                        continue

                    price = self._extract_price(blob)
                    results.append(
                        {
                            "set_id": set_id,
                            "set_name": name,
                            "theme": self._guess_theme(name),
                            "source": "amazon_bestsellers",
                            "current_price": price,
                            "eol_date_prediction": None,
                            "listing_url": href,
                            "metadata": {"raw_blob": blob[:1000]},
                        }
                    )

                dedup: dict[str, Dict[str, Any]] = {}
                for row in results:
                    dedup[row["set_id"]] = row
                return list(dedup.values())[:limit]
            finally:
                await page.close()

        return await self._with_retry("fetch_bestsellers", _run)


class VintedScraper(BaseStealthScraper):
    async def search_new_sealed(self, query: str, limit: int = 10) -> list[MarketListing]:
        encoded_query = quote_plus(f"{query} nuovo sigillato")
        url = f"https://www.vinted.it/catalog?search_text={encoded_query}"

        async def _run() -> list[MarketListing]:
            page = await self._new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded")
                await self._human_jitter()

                cards: list[Dict[str, str]] = await page.evaluate(
                    """
                    () => {
                      const links = Array.from(document.querySelectorAll('a[href*="/items/"]'));
                      const out = [];
                      const seen = new Set();

                      for (const link of links) {
                        const href = link.href || '';
                        if (!href || seen.has(href)) continue;
                        seen.add(href);

                        const card = link.closest('article,li,div');
                        const title = (link.textContent || '').trim();
                        const blob = card ? (card.innerText || '') : title;
                        if (!title || title.length < 4) continue;

                        out.push({ href, title, blob });
                      }
                      return out;
                    }
                    """
                )

                listings: list[MarketListing] = []
                for item in cards:
                    title = item.get("title") or ""
                    blob = item.get("blob") or ""
                    condition = self._normalize_condition(title, blob)
                    if condition != "new":
                        continue

                    set_id = self._extract_set_id(query, title, blob)
                    price = self._extract_price(blob)
                    href = item.get("href") or ""
                    if not set_id or price is None or not href:
                        continue

                    listings.append(
                        MarketListing(
                            platform="vinted",
                            set_id=set_id,
                            set_name=title,
                            price=price,
                            listing_url=href,
                            condition=condition,
                            source_note="secondary_market",
                        )
                    )

                listings.sort(key=lambda row: row.price)
                return listings[:limit]
            finally:
                await page.close()

        return await self._with_retry(f"vinted_search:{query}", _run)


class SubitoScraper(BaseStealthScraper):
    async def search_new_sealed(self, query: str, limit: int = 10) -> list[MarketListing]:
        encoded_query = quote_plus(f"{query} lego nuovo sigillato")
        url = f"https://www.subito.it/annunci-italia/vendita/usato/?q={encoded_query}"

        async def _run() -> list[MarketListing]:
            page = await self._new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded")
                await self._human_jitter()

                cards: list[Dict[str, str]] = await page.evaluate(
                    """
                    () => {
                      const links = Array.from(document.querySelectorAll('a[href*="subito.it"]'));
                      const out = [];
                      const seen = new Set();

                      for (const link of links) {
                        const href = link.href || '';
                        if (!href || seen.has(href)) continue;
                        seen.add(href);

                        const card = link.closest('article,li,div');
                        const title = (link.textContent || '').trim();
                        const blob = card ? (card.innerText || '') : title;
                        if (!title || title.length < 4) continue;

                        out.push({ href, title, blob });
                      }
                      return out;
                    }
                    """
                )

                listings: list[MarketListing] = []
                for item in cards:
                    title = item.get("title") or ""
                    blob = item.get("blob") or ""
                    condition = self._normalize_condition(title, blob)
                    if condition != "new":
                        continue

                    set_id = self._extract_set_id(query, title, blob)
                    price = self._extract_price(blob)
                    href = item.get("href") or ""
                    if not set_id or price is None or not href:
                        continue

                    listings.append(
                        MarketListing(
                            platform="subito",
                            set_id=set_id,
                            set_name=title,
                            price=price,
                            listing_url=href,
                            condition=condition,
                            source_note="secondary_market",
                        )
                    )

                listings.sort(key=lambda row: row.price)
                return listings[:limit]
            finally:
                await page.close()

        return await self._with_retry(f"subito_search:{query}", _run)


class EbayItScraper(BaseStealthScraper):
    async def search_new_sealed(self, query: str, limit: int = 10) -> list[MarketListing]:
        encoded_query = quote_plus(f"{query} lego nuovo sigillato")
        url = (
            "https://www.ebay.it/sch/i.html"
            f"?_nkw={encoded_query}&LH_ItemCondition=1000&LH_BIN=1&rt=nc"
        )

        async def _run() -> list[MarketListing]:
            page = await self._new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded")
                await self._human_jitter()

                cards: list[Dict[str, str]] = await page.evaluate(
                    """
                    () => {
                      const items = Array.from(document.querySelectorAll('li.s-item'));
                      const out = [];
                      const seen = new Set();

                      for (const item of items) {
                        const link = item.querySelector('a.s-item__link');
                        const href = link ? (link.href || '') : '';
                        if (!href || seen.has(href)) continue;
                        seen.add(href);

                        const titleEl = item.querySelector('.s-item__title');
                        const priceEl = item.querySelector('.s-item__price');
                        const title = titleEl ? (titleEl.textContent || '').trim() : '';
                        const price = priceEl ? (priceEl.textContent || '').trim() : '';
                        const blob = (item.innerText || '').trim();
                        if (!title || title.length < 4) continue;
                        out.push({ href, title, price, blob });
                      }
                      return out;
                    }
                    """
                )

                listings: list[MarketListing] = []
                for item in cards:
                    title = item.get("title") or ""
                    blob = item.get("blob") or ""
                    if "shop on ebay" in title.lower():
                        continue

                    condition = self._normalize_condition(title, blob)
                    if condition != "new":
                        continue

                    set_id = self._extract_set_id(query, title, blob)
                    price = self._extract_price(item.get("price"), blob)
                    href = item.get("href") or ""
                    if not set_id or price is None or not href:
                        continue

                    listings.append(
                        MarketListing(
                            platform="ebay",
                            set_id=set_id,
                            set_name=title,
                            price=price,
                            listing_url=href,
                            condition=condition,
                            source_note="secondary_market",
                        )
                    )

                listings.sort(key=lambda row: row.price)
                return listings[:limit]
            finally:
                await page.close()

        return await self._with_retry(f"ebay_it_search:{query}", _run)


class SecondaryMarketValidator:
    async def compare_secondary_prices(
        self,
        opportunities: Iterable[Dict[str, Any]],
        *,
        per_set_limit: int = 3,
    ) -> Dict[str, list[MarketListing]]:
        results: Dict[str, list[MarketListing]] = {}

        async with VintedScraper() as vinted, SubitoScraper() as subito, EbayItScraper() as ebay_it:
            for candidate in opportunities:
                set_id = str(candidate.get("set_id") or "").strip()
                set_name = str(candidate.get("set_name") or "").strip()
                if not set_id and not set_name:
                    continue

                query = f"{set_id} {set_name}".strip()

                vinted_listings, subito_listings, ebay_listings = await asyncio.gather(
                    vinted.search_new_sealed(query, limit=per_set_limit),
                    subito.search_new_sealed(query, limit=per_set_limit),
                    ebay_it.search_new_sealed(query, limit=per_set_limit),
                    return_exceptions=True,
                )

                listings: list[MarketListing] = []
                if isinstance(vinted_listings, Exception):
                    LOGGER.warning("Vinted validation failed for %s: %s", query, vinted_listings)
                else:
                    listings.extend(vinted_listings)

                if isinstance(subito_listings, Exception):
                    LOGGER.warning("Subito validation failed for %s: %s", query, subito_listings)
                else:
                    listings.extend(subito_listings)

                if isinstance(ebay_listings, Exception):
                    LOGGER.warning("eBay.it validation failed for %s: %s", query, ebay_listings)
                else:
                    listings.extend(ebay_listings)

                if listings:
                    listings.sort(key=lambda row: row.price)
                    results[set_id or set_name] = listings[:per_set_limit]

        return results


__all__ = [
    "AmazonBestsellerScraper",
    "EbayItScraper",
    "LegoRetiringScraper",
    "MarketListing",
    "SecondaryMarketValidator",
    "SubitoScraper",
    "VintedScraper",
]
