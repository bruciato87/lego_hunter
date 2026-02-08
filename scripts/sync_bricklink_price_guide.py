from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

try:
    from requests_oauthlib import OAuth1
except Exception:  # noqa: BLE001
    OAuth1 = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import LegoHunterRepository


LOGGER = logging.getLogger("bricklink_sync")
DEFAULT_OUTPUT_PATH = Path("data/historical_seed/bricklink_reference_cases.csv")
DEFAULT_API_BASE = "https://api.bricklink.com/api/store/v1"
EU_FALLBACK_COUNTRIES = (
    "IT",
    "DE",
    "FR",
    "ES",
    "NL",
    "BE",
    "AT",
    "IE",
    "PT",
    "PL",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


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
    values = []
    for chunk in str(raw_value or "").split(","):
        value = chunk.strip().upper()
        if not value:
            continue
        values.append(value)
    return values


def _country_to_region(country_code: str) -> str:
    if country_code in set(EU_FALLBACK_COUNTRIES) or country_code in {"CH", "NO", "GB", "UK", "SE", "DK", "FI"}:
        return "EU"
    return "OTHER"


class BrickLinkClient:
    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        token_value: str,
        token_secret: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout_sec: float = 16.0,
    ) -> None:
        if OAuth1 is None:
            raise RuntimeError(
                "requests-oauthlib not installed. Install dependencies from requirements.txt first."
            )
        self.api_base = api_base.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.session = requests.Session()
        self.auth = OAuth1(
            client_key=consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=token_value,
            resource_owner_secret=token_secret,
            signature_method="HMAC-SHA1",
            signature_type="AUTH_HEADER",
        )

    def get_price_guide(
        self,
        *,
        set_number: str,
        country_code: str,
        guide_type: str = "sold",
        new_or_used: str = "N",
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.api_base}/items/SET/{set_number}/price"
        params = {
            "guide_type": guide_type,
            "new_or_used": new_or_used,
            "country_code": country_code,
        }
        response = self.session.get(url, params=params, auth=self.auth, timeout=self.timeout_sec)
        if response.status_code >= 400:
            message = response.text[:300]
            LOGGER.warning(
                "BrickLink price guide request failed | set=%s country=%s status=%s body=%s",
                set_number,
                country_code,
                response.status_code,
                message,
            )
            return None

        try:
            payload = response.json()
        except ValueError:
            LOGGER.warning("BrickLink non-JSON response | set=%s country=%s", set_number, country_code)
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            LOGGER.warning("BrickLink response missing data field | set=%s country=%s", set_number, country_code)
            return None
        return data


def _extract_price_metrics(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    avg_price = _safe_float(data.get("avg_price"))
    min_price = _safe_float(data.get("min_price"))
    max_price = _safe_float(data.get("max_price"))
    qty_avg_price = _safe_float(data.get("qty_avg_price"))
    total_quantity = _safe_float(data.get("total_quantity"))
    unit_quantity = _safe_float(data.get("unit_quantity"))

    if avg_price is None:
        return None

    return {
        "avg_price": avg_price,
        "min_price": min_price if min_price is not None else avg_price,
        "max_price": max_price if max_price is not None else avg_price,
        "qty_avg_price": qty_avg_price if qty_avg_price is not None else avg_price,
        "total_quantity": int(total_quantity or 0),
        "unit_quantity": int(unit_quantity or 0),
    }


def _discover_targets_from_supabase(limit: int, lookback_days: int) -> list[Dict[str, Any]]:
    repository = LegoHunterRepository.from_env()
    opportunities = repository.get_recent_opportunities(days=lookback_days, limit=max(50, limit * 3))
    portfolio = repository.get_portfolio(status="holding")

    by_set: Dict[str, Dict[str, Any]] = {}
    for row in opportunities:
        set_id = str(row.get("set_id") or "").strip()
        if not set_id:
            continue
        score = int(row.get("ai_investment_score") or 0)
        current = by_set.get(set_id)
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
        if set_id in by_set:
            continue
        by_set[set_id] = {
            "set_id": set_id,
            "set_name": str(row.get("set_name") or "").strip() or set_id,
            "theme": str(row.get("theme") or "").strip() or "Unknown",
            "msrp_hint": _safe_float(row.get("purchase_price")),
            "ai_score": 0,
        }

    sorted_targets = sorted(
        by_set.values(),
        key=lambda row: (int(row.get("ai_score") or 0), str(row.get("set_id") or "")),
        reverse=True,
    )
    return sorted_targets[:limit]


def _targets_from_set_ids(set_ids: Iterable[str]) -> list[Dict[str, Any]]:
    targets: list[Dict[str, Any]] = []
    for raw in set_ids:
        set_id = str(raw or "").strip()
        if not set_id:
            continue
        normalized = set_id.split("-", 1)[0] if "-" in set_id else set_id
        targets.append(
            {
                "set_id": normalized,
                "set_name": normalized,
                "theme": "Unknown",
                "msrp_hint": None,
                "ai_score": 0,
            }
        )
    return targets


def _resolve_market_countries(primary_countries: list[str], *, include_eu_fallback: bool) -> list[str]:
    resolved = []
    seen = set()
    for country in primary_countries:
        upper = country.upper()
        if upper and upper not in seen:
            seen.add(upper)
            resolved.append(upper)
    if include_eu_fallback:
        for country in EU_FALLBACK_COUNTRIES:
            if country not in seen:
                seen.add(country)
                resolved.append(country)
    return resolved


def _build_case_row(
    *,
    target: Dict[str, Any],
    country_code: str,
    metrics: Dict[str, Any],
    today: date,
    target_roi_pct: float,
) -> Dict[str, Any]:
    set_id = str(target.get("set_id") or "").strip()
    set_name = str(target.get("set_name") or "").strip() or set_id
    theme = str(target.get("theme") or "").strip() or "Unknown"
    set_number = _normalize_set_number(set_id)

    msrp_hint = _safe_float(target.get("msrp_hint"))
    avg_price = float(metrics.get("avg_price") or 0.0)
    if msrp_hint is None or msrp_hint <= 0:
        msrp_hint = avg_price
    if msrp_hint <= 0:
        msrp_hint = 1.0

    roi_12m = ((avg_price - msrp_hint) / msrp_hint) * 100.0
    total_quantity = int(metrics.get("total_quantity") or 0)
    case_weight = min(1.5, 0.75 + (math.log10(total_quantity + 1.0) * 0.30))
    recency_weight = 1.20 if country_code == "IT" else 1.0

    start_date = (today - timedelta(days=180)).isoformat()
    end_date = today.isoformat()
    market_region = _country_to_region(country_code)

    return {
        "set_id": set_id,
        "set_number": set_number,
        "set_name": set_name,
        "theme": theme,
        "release_year": "",
        "msrp_usd": f"{msrp_hint:.4f}",
        "start_date": start_date,
        "end_date": end_date,
        "observation_months": 6,
        "start_price_usd": f"{msrp_hint:.4f}",
        "price_12m_usd": f"{avg_price:.4f}",
        "price_24m_usd": "",
        "roi_12m_pct": f"{roi_12m:.4f}",
        "roi_24m_pct": "",
        "annualized_roi_pct": "",
        "max_drawdown_pct": "",
        "win_12m": int(roi_12m >= target_roi_pct),
        "win_24m": "",
        "source_dataset": f"bricklink_priceguide_sold_{country_code.lower()}",
        "pattern_tags": json.dumps(["regional_market_signal"], ensure_ascii=True),
        "market_country": country_code,
        "market_region": market_region,
        "market_scope": "country",
        "recency_weight": f"{recency_weight:.3f}",
        "case_weight": f"{case_weight:.3f}",
        "total_quantity": total_quantity,
        "qty_avg_price": f"{float(metrics.get('qty_avg_price') or avg_price):.4f}",
    }


def build_rows(
    *,
    client: BrickLinkClient,
    targets: list[Dict[str, Any]],
    countries: list[str],
    min_total_quantity: int,
    guide_type: str,
    new_or_used: str,
    target_roi_pct: float,
) -> list[Dict[str, Any]]:
    today = datetime.utcnow().date()
    rows: list[Dict[str, Any]] = []
    for target in targets:
        set_id = str(target.get("set_id") or "").strip()
        set_number = _normalize_set_number(set_id)
        if not set_number:
            continue

        best: Optional[Dict[str, Any]] = None
        for country_code in countries:
            payload = client.get_price_guide(
                set_number=set_number,
                country_code=country_code,
                guide_type=guide_type,
                new_or_used=new_or_used,
            )
            if payload is None:
                continue
            metrics = _extract_price_metrics(payload)
            if metrics is None:
                continue
            total_quantity = int(metrics.get("total_quantity") or 0)
            if total_quantity < min_total_quantity:
                continue
            candidate = {"country": country_code, "metrics": metrics}
            if best is None:
                best = candidate
                continue
            if int(candidate["metrics"].get("total_quantity") or 0) > int(best["metrics"].get("total_quantity") or 0):
                best = candidate

        if best is None:
            continue

        rows.append(
            _build_case_row(
                target=target,
                country_code=str(best["country"]),
                metrics=dict(best["metrics"]),
                today=today,
                target_roi_pct=target_roi_pct,
            )
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
        "total_quantity",
        "qty_avg_price",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _required_env(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env: {name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync BrickLink sold price guide into historical seed CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--countries", default="IT", help="Comma-separated priority countries (ISO-2).")
    parser.add_argument("--include-eu-fallback", action="store_true", default=False)
    parser.add_argument("--set-ids", default="", help="Optional comma-separated set IDs/set numbers.")
    parser.add_argument("--max-sets", type=int, default=120)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-total-quantity", type=int, default=2)
    parser.add_argument("--guide-type", default="sold", choices=["sold", "stock"])
    parser.add_argument("--new-or-used", default="N", choices=["N", "U"])
    parser.add_argument("--target-roi-pct", type=float, default=20.0)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    consumer_key = _required_env("BRICKLINK_CONSUMER_KEY")
    consumer_secret = _required_env("BRICKLINK_CONSUMER_SECRET")
    token_value = _required_env("BRICKLINK_TOKEN_VALUE")
    token_secret = _required_env("BRICKLINK_TOKEN_SECRET")
    client = BrickLinkClient(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        token_value=token_value,
        token_secret=token_secret,
        api_base=args.api_base,
    )

    set_ids = [item.strip() for item in str(args.set_ids or "").split(",") if item.strip()]
    if set_ids:
        targets = _targets_from_set_ids(set_ids)
    else:
        targets = _discover_targets_from_supabase(limit=max(10, int(args.max_sets)), lookback_days=max(30, int(args.lookback_days)))

    countries = _resolve_market_countries(
        _parse_csv_list(args.countries),
        include_eu_fallback=bool(args.include_eu_fallback),
    )
    LOGGER.info(
        "BrickLink sync start | targets=%s countries=%s guide_type=%s new_or_used=%s",
        len(targets),
        ",".join(countries),
        args.guide_type,
        args.new_or_used,
    )

    rows = build_rows(
        client=client,
        targets=targets,
        countries=countries,
        min_total_quantity=max(1, int(args.min_total_quantity)),
        guide_type=args.guide_type,
        new_or_used=args.new_or_used,
        target_roi_pct=float(args.target_roi_pct),
    )
    write_rows(args.out, rows)
    LOGGER.info("BrickLink sync completed | rows=%s out=%s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
