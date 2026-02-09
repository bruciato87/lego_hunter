from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote_plus

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import LegoHunterRepository


LOGGER = logging.getLogger("ebay_history_sync")
DEFAULT_OUTPUT_PATH = Path("data/historical_seed/ebay_reference_cases.csv")
HISTORICAL_FIELDNAMES = [
    "set_id",
    "set_number",
    "set_name",
    "theme",
    "release_year",
    "msrp_usd",
    "start_date",
    "end_date",
    "observation_months",
    "start_price_usd",
    "price_12m_usd",
    "price_24m_usd",
    "roi_12m_pct",
    "roi_24m_pct",
    "annualized_roi_pct",
    "max_drawdown_pct",
    "win_12m",
    "win_24m",
    "source_dataset",
    "pattern_tags",
    "market_country",
    "market_region",
    "market_scope",
    "recency_weight",
    "case_weight",
    "sold_listing_count",
    "sold_avg_price",
    "sold_stdev_price",
]

EBAY_MARKETS = {
    "IT": "https://www.ebay.it",
    "DE": "https://www.ebay.de",
    "FR": "https://www.ebay.fr",
    "ES": "https://www.ebay.es",
    "NL": "https://www.ebay.nl",
}
VINTED_MARKETS = {
    "IT": "https://www.vinted.it",
    "DE": "https://www.vinted.de",
    "FR": "https://www.vinted.fr",
    "ES": "https://www.vinted.es",
    "NL": "https://www.vinted.nl",
}
EU_FALLBACK_MARKETS = ("DE", "FR", "ES", "NL")
PRICE_RE = re.compile(
    r"(?:€|eur)\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"|(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?:€|eur)",
    re.IGNORECASE,
)
EBAY_S_ITEM_BLOCK_RE = re.compile(r"<li[^>]+class=\"[^\"]*s-item[^\"]*\"[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
EBAY_S_CARD_BLOCK_RE = re.compile(r"<li[^>]+class=\"[^\"]*s-card[^\"]*\"[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"s-item__title[^>]*>(.*?)<", re.IGNORECASE | re.DOTALL)
PRICE_SPAN_RE = re.compile(r"s-item__price[^>]*>(.*?)<", re.IGNORECASE | re.DOTALL)
EBAY_CARD_TITLE_RE = re.compile(r"s-card__title[^>]*>(.*?)<", re.IGNORECASE | re.DOTALL)
EBAY_CARD_TITLE_SPAN_RE = re.compile(
    r"s-card__title[^>]*>.*?<span[^>]*>(.*?)</span>",
    re.IGNORECASE | re.DOTALL,
)
EBAY_CARD_TITLE_BLOCK_RE = re.compile(
    r"s-card__title[^>]*>(.*?)</div>",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
SET_ID_RE = re.compile(r"\b(\d{4,6})\b")
VINTED_ITEM_LINK_RE = re.compile(
    r'<a[^>]+href="(?P<url>https?://www\.vinted\.[^"]*/items/\d+[^"]*)"(?P<attrs>[^>]*)>',
    re.IGNORECASE,
)
VINTED_NEW_KEYWORDS = (
    "nuovo con cartellino",
    "nuovo",
    "new with tags",
    "brand new",
    "new with label",
)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _normalize_set_number(raw_set_id: str) -> str:
    text = str(raw_set_id or "").strip()
    if not text:
        return ""
    if "-" in text:
        return text
    if text.isdigit():
        return f"{text}-1"
    return text


def _parse_csv_list(raw_value: str) -> list[str]:
    out: list[str] = []
    for item in str(raw_value or "").split(","):
        normalized = item.strip().upper()
        if normalized:
            out.append(normalized)
    return out


def _build_query_variants(set_id: str, set_name: str) -> list[str]:
    variants: list[str] = []
    base_id = str(set_id or "").strip()
    if base_id:
        variants.append(base_id)

    normalized_name = re.sub(r"[^0-9A-Za-z ]+", " ", str(set_name or ""))
    words = [word for word in normalized_name.split() if len(word) >= 2]
    if base_id and words:
        variants.append(f"{base_id} {' '.join(words[:4])}")
    if base_id and words:
        variants.append(f"{base_id} {' '.join(words[:2])}")

    # Keep order and remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [base_id]


def _clean_html_text(value: str) -> str:
    return TAG_RE.sub(" ", value or "").replace("&nbsp;", " ").strip()


def _parse_price(text: str) -> Optional[float]:
    match = PRICE_RE.search((text or "").replace("\u00a0", " "))
    if not match:
        return None
    raw = (match.group(1) or match.group(2) or "").replace(" ", "")
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _extract_all_prices(text: str) -> list[float]:
    prices: list[float] = []
    for match in PRICE_RE.finditer((text or "").replace("\u00a0", " ")):
        raw = (match.group(1) or match.group(2) or "").replace(" ", "")
        if "," in raw and "." in raw:
            if raw.rfind(",") > raw.rfind("."):
                raw = raw.replace(".", "").replace(",", ".")
            else:
                raw = raw.replace(",", "")
        elif "," in raw:
            raw = raw.replace(".", "").replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            prices.append(value)
    return prices


def _extract_primary_price_from_block(block: str) -> Optional[float]:
    # Price spans are still present on old eBay markup.
    price_match = PRICE_SPAN_RE.search(block or "")
    if price_match:
        price_value = _parse_price(_clean_html_text(price_match.group(1)))
        if price_value is not None:
            return price_value

    text = _clean_html_text(block or "")
    prices = _extract_all_prices(text)
    if not prices:
        return None
    # Use the first listed price; shipping/fee numbers usually come after.
    return prices[0]


def _extract_ebay_title(block: str) -> str:
    source = block or ""
    for pattern in (TITLE_RE, EBAY_CARD_TITLE_SPAN_RE, EBAY_CARD_TITLE_BLOCK_RE, EBAY_CARD_TITLE_RE):
        match = pattern.search(source)
        if not match:
            continue
        text = _clean_html_text(match.group(1))
        if text:
            return text
    title_attr = re.search(r'title="([^"]+)"', block or "", re.IGNORECASE)
    if title_attr:
        return _clean_html_text(title_attr.group(1))
    return ""


def _extract_ebay_listing_blocks(html: str) -> list[str]:
    blocks = EBAY_S_ITEM_BLOCK_RE.findall(html or "")
    if blocks:
        return blocks
    return EBAY_S_CARD_BLOCK_RE.findall(html or "")


def _build_search_url(base_url: str, query: str, condition_new: bool = True) -> str:
    encoded = quote_plus(f"{query} lego")
    params = f"_nkw={encoded}&LH_Complete=1&LH_Sold=1&rt=nc"
    if condition_new:
        params += "&LH_ItemCondition=1000"
    return f"{base_url}/sch/i.html?{params}"


def _build_vinted_search_url(base_url: str, query: str) -> str:
    encoded = quote_plus(f"{query} lego")
    return f"{base_url}/catalog?search_text={encoded}&order=newest_first"


def extract_sold_prices_from_html(html: str) -> list[float]:
    prices: list[float] = []
    for block in _extract_ebay_listing_blocks(html):
        title = _extract_ebay_title(block)
        if not title or "shop on ebay" in title.lower():
            continue
        price_value = _extract_primary_price_from_block(block)
        if price_value is None:
            continue
        prices.append(price_value)
    return prices


def extract_vinted_listing_prices_from_html(html: str, *, require_new: bool = True) -> list[float]:
    text = html or ""
    prices: list[float] = []
    seen_urls: set[str] = set()
    for match in VINTED_ITEM_LINK_RE.finditer(text):
        url = str(match.group("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        attrs = str(match.group("attrs") or "")
        title_attr_match = re.search(r'title="([^"]+)"', attrs, re.IGNORECASE)
        title_attr = _clean_html_text(title_attr_match.group(1) if title_attr_match else "")

        snippet_start = max(0, match.start() - 600)
        snippet_end = min(len(text), match.end() + 1000)
        snippet = _clean_html_text(text[snippet_start:snippet_end])

        combined_text = f"{title_attr} {snippet}".strip().lower()
        if "lego" not in combined_text:
            continue
        if require_new and not any(keyword in combined_text for keyword in VINTED_NEW_KEYWORDS):
            continue

        price = _parse_price(title_attr)
        if price is None:
            price = _parse_price(snippet)
        if price is None:
            continue
        prices.append(price)
    return prices


def _extract_set_id(*texts: Optional[str]) -> str:
    for text in texts:
        match = SET_ID_RE.search(str(text or ""))
        if match:
            return match.group(1)
    return ""


def _country_to_region(country: str) -> str:
    return "EU" if country in {"IT", "DE", "FR", "ES", "NL"} else "OTHER"


def _normalize_row_for_storage(row: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for field in HISTORICAL_FIELDNAMES:
        value = row.get(field, "")
        normalized[field] = "" if value is None else str(value)
    return normalized


def _historical_row_key(row: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("set_id") or "").strip(),
        str(row.get("source_dataset") or "").strip().lower(),
        str(row.get("market_country") or "").strip().upper(),
        str(row.get("end_date") or "").strip(),
    )


def _row_sort_key(row: Dict[str, Any]) -> tuple[str, str, str, str]:
    end_date = str(row.get("end_date") or "").strip()
    set_id = str(row.get("set_id") or "").strip()
    source_dataset = str(row.get("source_dataset") or "").strip().lower()
    market_country = str(row.get("market_country") or "").strip().upper()
    # Desc on end_date for easier inspection of freshest records.
    return (end_date, set_id, source_dataset, market_country)


def load_existing_rows(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        return []
    rows: list[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not row:
                continue
            rows.append(_normalize_row_for_storage(row))
    return rows


def merge_reference_rows(
    existing_rows: list[Dict[str, Any]],
    incoming_rows: list[Dict[str, Any]],
) -> tuple[list[Dict[str, str]], Dict[str, int]]:
    index: Dict[tuple[str, str, str, str], Dict[str, str]] = {}
    dropped_existing_duplicates = 0
    for row in existing_rows:
        normalized = _normalize_row_for_storage(row)
        key = _historical_row_key(normalized)
        if not all(key):
            continue
        if key in index:
            dropped_existing_duplicates += 1
        index[key] = normalized

    added = 0
    updated = 0
    unchanged = 0
    dropped_incoming_duplicates = 0
    for row in incoming_rows:
        normalized = _normalize_row_for_storage(row)
        key = _historical_row_key(normalized)
        if not all(key):
            continue
        previous = index.get(key)
        if previous is None:
            index[key] = normalized
            added += 1
            continue
        if previous == normalized:
            unchanged += 1
            continue
        # Same key with changed payload -> update in place.
        if key in index:
            updated += 1
            dropped_incoming_duplicates += 1
        index[key] = normalized

    merged_rows = sorted(index.values(), key=_row_sort_key, reverse=True)
    stats = {
        "existing_valid": len(index) - added,
        "incoming_valid": added + updated + unchanged,
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "dropped_existing_duplicates": dropped_existing_duplicates,
        "dropped_incoming_duplicates": dropped_incoming_duplicates,
        "merged_total": len(merged_rows),
    }
    return merged_rows, stats


def _discover_targets_from_supabase(limit: int, lookback_days: int) -> list[Dict[str, Any]]:
    repository = LegoHunterRepository.from_env()
    opportunities = repository.get_recent_opportunities(days=lookback_days, limit=max(50, limit * 3))
    portfolio = repository.get_portfolio(status="holding")

    by_set: Dict[str, Dict[str, Any]] = {}
    for row in opportunities:
        set_id = str(row.get("set_id") or "").strip()
        if not set_id:
            continue
        current = by_set.get(set_id)
        score = int(row.get("ai_investment_score") or 0)
        if current is not None and int(current.get("ai_score") or 0) >= score:
            continue
        by_set[set_id] = {
            "set_id": set_id,
            "set_name": str(row.get("set_name") or "").strip() or set_id,
            "theme": str(row.get("theme") or "").strip() or "Unknown",
            "msrp_hint": _safe_float(row.get("current_price")),
            "ai_score": score,
        }

    for row in portfolio:
        set_id = str(row.get("set_id") or "").strip()
        if not set_id:
            continue
        by_set.setdefault(
            set_id,
            {
                "set_id": set_id,
                "set_name": str(row.get("set_name") or "").strip() or set_id,
                "theme": str(row.get("theme") or "").strip() or "Unknown",
                "msrp_hint": _safe_float(row.get("purchase_price")),
                "ai_score": 0,
            },
        )

    ranked = sorted(
        by_set.values(),
        key=lambda item: (int(item.get("ai_score") or 0), str(item.get("set_id") or "")),
        reverse=True,
    )
    return ranked[:limit]


def _targets_from_set_ids(set_ids: Iterable[str]) -> list[Dict[str, Any]]:
    targets: list[Dict[str, Any]] = []
    for raw in set_ids:
        set_id = _extract_set_id(raw) or str(raw or "").strip()
        if not set_id:
            continue
        targets.append(
            {
                "set_id": set_id,
                "set_name": set_id,
                "theme": "Unknown",
                "msrp_hint": None,
                "ai_score": 0,
            }
        )
    return targets


class EbaySoldClient:
    def __init__(self, *, timeout_sec: float = 18.0, max_retries: int = 3) -> None:
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.session = requests.Session()

    def fetch_sold_prices(self, *, market: str, query: str) -> list[float]:
        base_url = EBAY_MARKETS.get(market)
        if not base_url:
            return []
        url = _build_search_url(base_url, query)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
            }
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout_sec)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}")
                return extract_sold_prices_from_html(response.text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = (1.2 ** attempt) + random.uniform(0.3, 1.1)
                time.sleep(sleep_sec)
        LOGGER.warning("eBay sold fetch failed | market=%s query=%s error=%s", market, query, last_exc)
        return []


class VintedCatalogClient:
    def __init__(self, *, timeout_sec: float = 18.0, max_retries: int = 3) -> None:
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.session = requests.Session()

    def fetch_listing_prices(self, *, market: str, query: str, require_new: bool = True) -> list[float]:
        base_url = VINTED_MARKETS.get(market)
        if not base_url:
            return []
        url = _build_vinted_search_url(base_url, query)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
            }
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout_sec)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}")
                return extract_vinted_listing_prices_from_html(response.text, require_new=require_new)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = (1.2 ** attempt) + random.uniform(0.3, 1.1)
                time.sleep(sleep_sec)
        LOGGER.warning("Vinted fetch failed | market=%s query=%s error=%s", market, query, last_exc)
        return []


def _build_case_rows(
    *,
    targets: list[Dict[str, Any]],
    markets: list[str],
    client: EbaySoldClient,
    min_sold_listings: int,
    target_roi_pct: float,
    max_markets_per_set: int = 5,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()
    start_date = (today - timedelta(days=90)).isoformat()
    for target in targets:
        set_id = str(target.get("set_id") or "").strip()
        set_name = str(target.get("set_name") or "").strip() or set_id
        if not set_id:
            continue
        theme = str(target.get("theme") or "").strip() or "Unknown"
        msrp_hint = _safe_float(target.get("msrp_hint"))
        if msrp_hint is None or msrp_hint <= 0:
            msrp_hint = None
        queries = _build_query_variants(set_id, set_name)
        target_rows_before = len(rows)
        market_rows = 0

        for market in markets:
            sold_prices: list[float] = []
            for query in queries:
                chunk = client.fetch_sold_prices(market=market, query=query)
                if chunk:
                    sold_prices.extend(chunk)
                if len(sold_prices) >= min_sold_listings:
                    break
            if sold_prices:
                sold_prices = sorted({round(price, 2) for price in sold_prices})
            if len(sold_prices) < min_sold_listings:
                continue
            sold_median = float(median(sold_prices))
            sold_avg = float(sum(sold_prices) / len(sold_prices))
            baseline = msrp_hint if msrp_hint and msrp_hint > 0 else sold_avg
            if baseline <= 0:
                continue
            roi_pct = ((sold_median - baseline) / baseline) * 100.0

            count = len(sold_prices)
            stdev = 0.0
            if count > 1:
                mean = sold_avg
                stdev = math.sqrt(sum((price - mean) ** 2 for price in sold_prices) / count)
            confidence_factor = max(0.55, min(1.35, 1.0 + (math.log10(count + 1) * 0.22) - min(0.35, stdev / max(1.0, sold_avg))))
            recency_weight = 1.25 if market == "IT" else 1.05

            rows.append(
                {
                    "set_id": set_id,
                    "set_number": _normalize_set_number(set_id),
                    "set_name": set_name,
                    "theme": theme,
                    "release_year": "",
                    "msrp_usd": f"{baseline:.4f}",
                    "start_date": start_date,
                    "end_date": today.isoformat(),
                    "observation_months": 3,
                    "start_price_usd": f"{baseline:.4f}",
                    "price_12m_usd": f"{sold_median:.4f}",
                    "price_24m_usd": "",
                    "roi_12m_pct": f"{roi_pct:.4f}",
                    "roi_24m_pct": "",
                    "annualized_roi_pct": "",
                    "max_drawdown_pct": "",
                    "win_12m": int(roi_pct >= target_roi_pct),
                    "win_24m": "",
                    "source_dataset": f"ebay_sold_{market.lower()}_90d",
                    "pattern_tags": json.dumps(["secondary_market_signal", "ebay_sold"], ensure_ascii=True),
                    "market_country": market,
                    "market_region": _country_to_region(market),
                    "market_scope": "country",
                    "recency_weight": f"{recency_weight:.3f}",
                    "case_weight": f"{confidence_factor:.3f}",
                    "sold_listing_count": count,
                    "sold_avg_price": f"{sold_avg:.4f}",
                    "sold_stdev_price": f"{stdev:.4f}",
                }
            )
            market_rows += 1
            if market_rows >= max(1, int(max_markets_per_set)):
                break
        if len(rows) == target_rows_before:
            LOGGER.debug("No sold data found for set=%s (%s) across markets=%s", set_id, set_name, ",".join(markets))
    return rows


def _build_vinted_case_rows(
    *,
    targets: list[Dict[str, Any]],
    markets: list[str],
    client: VintedCatalogClient,
    min_listings: int,
    target_roi_pct: float,
    max_markets_per_set: int = 3,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()
    start_date = (today - timedelta(days=30)).isoformat()

    for target in targets:
        set_id = str(target.get("set_id") or "").strip()
        set_name = str(target.get("set_name") or "").strip() or set_id
        if not set_id:
            continue
        theme = str(target.get("theme") or "").strip() or "Unknown"
        msrp_hint = _safe_float(target.get("msrp_hint"))
        if msrp_hint is None or msrp_hint <= 0:
            msrp_hint = None
        queries = _build_query_variants(set_id, set_name)
        target_rows_before = len(rows)
        market_rows = 0

        for market in markets:
            listing_prices: list[float] = []
            for query in queries:
                chunk = client.fetch_listing_prices(market=market, query=query, require_new=True)
                if chunk:
                    listing_prices.extend(chunk)
                if len(listing_prices) >= min_listings:
                    break
            if listing_prices:
                listing_prices = sorted({round(price, 2) for price in listing_prices})
            if len(listing_prices) < min_listings:
                continue

            listing_median = float(median(listing_prices))
            listing_avg = float(sum(listing_prices) / len(listing_prices))
            # Vinted shows ask prices (not sold): apply conservative haircut.
            conservative_realized_price = listing_median * 0.88
            baseline = msrp_hint if msrp_hint and msrp_hint > 0 else listing_avg
            if baseline <= 0:
                continue
            roi_pct = ((conservative_realized_price - baseline) / baseline) * 100.0

            count = len(listing_prices)
            stdev = 0.0
            if count > 1:
                mean = listing_avg
                stdev = math.sqrt(sum((price - mean) ** 2 for price in listing_prices) / count)
            confidence_factor = max(
                0.45,
                min(
                    1.05,
                    0.72 + (math.log10(count + 1) * 0.20) - min(0.28, stdev / max(1.0, listing_avg)),
                ),
            )
            recency_weight = 1.12 if market == "IT" else 0.95

            rows.append(
                {
                    "set_id": set_id,
                    "set_number": _normalize_set_number(set_id),
                    "set_name": set_name,
                    "theme": theme,
                    "release_year": "",
                    "msrp_usd": f"{baseline:.4f}",
                    "start_date": start_date,
                    "end_date": today.isoformat(),
                    "observation_months": 1,
                    "start_price_usd": f"{baseline:.4f}",
                    "price_12m_usd": f"{conservative_realized_price:.4f}",
                    "price_24m_usd": "",
                    "roi_12m_pct": f"{roi_pct:.4f}",
                    "roi_24m_pct": "",
                    "annualized_roi_pct": "",
                    "max_drawdown_pct": "",
                    "win_12m": int(roi_pct >= target_roi_pct),
                    "win_24m": "",
                    "source_dataset": f"vinted_active_{market.lower()}_30d",
                    "pattern_tags": json.dumps(["secondary_market_signal", "vinted_active"], ensure_ascii=True),
                    "market_country": market,
                    "market_region": _country_to_region(market),
                    "market_scope": "country",
                    "recency_weight": f"{recency_weight:.3f}",
                    "case_weight": f"{confidence_factor:.3f}",
                    "sold_listing_count": count,
                    "sold_avg_price": f"{listing_avg:.4f}",
                    "sold_stdev_price": f"{stdev:.4f}",
                }
            )
            market_rows += 1
            if market_rows >= max(1, int(max_markets_per_set)):
                break
        if len(rows) == target_rows_before:
            LOGGER.debug("No Vinted listings found for set=%s (%s) across markets=%s", set_id, set_name, ",".join(markets))
    return rows


def write_rows(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=HISTORICAL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row_for_storage(row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build IT/EU historical reference cases from secondary market data (eBay sold + Vinted active)."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--markets", default="IT", help="Comma-separated markets (IT,DE,FR,ES,NL)")
    parser.add_argument("--include-eu-fallback", action="store_true", default=False)
    parser.add_argument("--include-vinted", action="store_true", default=False)
    parser.add_argument("--set-ids", default="", help="Optional comma-separated set IDs")
    parser.add_argument("--max-sets", type=int, default=120)
    parser.add_argument("--max-vinted-targets", type=int, default=45)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-sold-listings", type=int, default=4)
    parser.add_argument("--max-markets-per-set", type=int, default=1)
    parser.add_argument("--min-vinted-listings", type=int, default=3)
    parser.add_argument("--max-vinted-markets-per-set", type=int, default=1)
    parser.add_argument("--target-roi-pct", type=float, default=20.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    markets = _parse_csv_list(args.markets)
    if args.include_eu_fallback:
        for market in EU_FALLBACK_MARKETS:
            if market not in markets:
                markets.append(market)
    markets = [market for market in markets if market in EBAY_MARKETS]
    if not markets:
        markets = ["IT"]

    set_ids = [item.strip() for item in str(args.set_ids or "").split(",") if item.strip()]
    if set_ids:
        targets = _targets_from_set_ids(set_ids)
    else:
        targets = _discover_targets_from_supabase(
            limit=max(10, int(args.max_sets)),
            lookback_days=max(30, int(args.lookback_days)),
        )

    LOGGER.info(
        "Secondary market sync start | targets=%s markets=%s include_vinted=%s",
        len(targets),
        ",".join(markets),
        bool(args.include_vinted),
    )
    ebay_client = EbaySoldClient()
    ebay_rows = _build_case_rows(
        targets=targets,
        markets=markets,
        client=ebay_client,
        min_sold_listings=max(1, int(args.min_sold_listings)),
        target_roi_pct=float(args.target_roi_pct),
        max_markets_per_set=max(1, int(args.max_markets_per_set)),
    )
    rows = list(ebay_rows)
    vinted_rows: list[Dict[str, Any]] = []
    if args.include_vinted:
        vinted_targets_limit = max(1, int(args.max_vinted_targets))
        vinted_targets = targets[:vinted_targets_limit]
        vinted_client = VintedCatalogClient()
        vinted_rows = _build_vinted_case_rows(
            targets=vinted_targets,
            markets=[market for market in markets if market in VINTED_MARKETS],
            client=vinted_client,
            min_listings=max(1, int(args.min_vinted_listings)),
            target_roi_pct=float(args.target_roi_pct),
            max_markets_per_set=max(1, int(args.max_vinted_markets_per_set)),
        )
        rows.extend(vinted_rows)

    existing_rows = load_existing_rows(args.out)
    merged_rows, merge_stats = merge_reference_rows(existing_rows, rows)
    write_rows(args.out, merged_rows)
    LOGGER.info(
        "Secondary market sync completed | ebay_rows=%s vinted_rows=%s incoming_total=%s existing_rows=%s added=%s updated=%s unchanged=%s merged_total=%s dropped_existing_dups=%s dropped_incoming_dups=%s out=%s",
        len(ebay_rows),
        len(vinted_rows),
        len(rows),
        len(existing_rows),
        merge_stats.get("added", 0),
        merge_stats.get("updated", 0),
        merge_stats.get("unchanged", 0),
        merge_stats.get("merged_total", 0),
        merge_stats.get("dropped_existing_duplicates", 0),
        merge_stats.get("dropped_incoming_duplicates", 0),
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
