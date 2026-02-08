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

EBAY_MARKETS = {
    "IT": "https://www.ebay.it",
    "DE": "https://www.ebay.de",
    "FR": "https://www.ebay.fr",
    "ES": "https://www.ebay.es",
    "NL": "https://www.ebay.nl",
}
EU_FALLBACK_MARKETS = ("DE", "FR", "ES", "NL")
PRICE_RE = re.compile(
    r"(?:€|eur)\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"|(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?:€|eur)",
    re.IGNORECASE,
)
LISTING_BLOCK_RE = re.compile(r"<li[^>]+class=\"[^\"]*s-item[^\"]*\"[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"s-item__title[^>]*>(.*?)<", re.IGNORECASE | re.DOTALL)
PRICE_SPAN_RE = re.compile(r"s-item__price[^>]*>(.*?)<", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
SET_ID_RE = re.compile(r"\b(\d{4,6})\b")
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


def _build_search_url(base_url: str, query: str, condition_new: bool = True) -> str:
    encoded = quote_plus(f"{query} lego")
    params = f"_nkw={encoded}&LH_Complete=1&LH_Sold=1&rt=nc"
    if condition_new:
        params += "&LH_ItemCondition=1000"
    return f"{base_url}/sch/i.html?{params}"


def extract_sold_prices_from_html(html: str) -> list[float]:
    prices: list[float] = []
    for block in LISTING_BLOCK_RE.findall(html or ""):
        title_match = TITLE_RE.search(block)
        title = _clean_html_text(title_match.group(1) if title_match else "")
        if not title or "shop on ebay" in title.lower():
            continue
        price_match = PRICE_SPAN_RE.search(block)
        price_text = _clean_html_text(price_match.group(1) if price_match else block)
        price_value = _parse_price(price_text)
        if price_value is None:
            continue
        prices.append(price_value)
    return prices


def _extract_set_id(*texts: Optional[str]) -> str:
    for text in texts:
        match = SET_ID_RE.search(str(text or ""))
        if match:
            return match.group(1)
    return ""


def _country_to_region(country: str) -> str:
    return "EU" if country in {"IT", "DE", "FR", "ES", "NL"} else "OTHER"


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


def _build_case_rows(
    *,
    targets: list[Dict[str, Any]],
    markets: list[str],
    client: EbaySoldClient,
    min_sold_listings: int,
    target_roi_pct: float,
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
        query = f"{set_id} {set_name}".strip()

        for market in markets:
            sold_prices = client.fetch_sold_prices(market=market, query=query)
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
    return rows


def write_rows(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build IT/EU historical reference cases from eBay sold listings.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--markets", default="IT", help="Comma-separated markets (IT,DE,FR,ES,NL)")
    parser.add_argument("--include-eu-fallback", action="store_true", default=False)
    parser.add_argument("--set-ids", default="", help="Optional comma-separated set IDs")
    parser.add_argument("--max-sets", type=int, default=120)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-sold-listings", type=int, default=4)
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

    LOGGER.info("eBay sold sync start | targets=%s markets=%s", len(targets), ",".join(markets))
    client = EbaySoldClient()
    rows = _build_case_rows(
        targets=targets,
        markets=markets,
        client=client,
        min_sold_listings=max(1, int(args.min_sold_listings)),
        target_roi_pct=float(args.target_roi_pct),
    )
    write_rows(args.out, rows)
    LOGGER.info("eBay sold sync completed | rows=%s out=%s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
