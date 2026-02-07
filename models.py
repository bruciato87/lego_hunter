from __future__ import annotations

import logging
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, Optional

from supabase import Client, create_client

LOGGER = logging.getLogger(__name__)


@dataclass
class OpportunityRadarRecord:
    set_id: str
    set_name: str
    theme: Optional[str] = None
    source: str = "other"
    eol_date_prediction: Optional[str] = None
    market_demand_score: int = 0
    ai_investment_score: int = 1
    ai_analysis_summary: str = ""
    current_price: Optional[float] = None
    currency: str = "EUR"
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    is_archived: bool = False


@dataclass
class MarketTimeSeriesRecord:
    set_id: str
    platform: str
    price: float
    recorded_at: Optional[str] = None
    set_name: Optional[str] = None
    listing_type: str = "unknown"
    shipping_cost: float = 0.0
    currency: str = "EUR"
    seller_name: Optional[str] = None
    seller_rating: Optional[float] = None
    stock_status: Optional[str] = None
    listing_url: Optional[str] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioRecord:
    set_id: str
    set_name: str
    purchase_price: float
    purchase_date: str
    quantity: int = 1
    theme: Optional[str] = None
    purchase_platform: Optional[str] = None
    shipping_in_cost: float = 0.0
    estimated_market_price: Optional[float] = None
    status: str = "holding"
    notes: Optional[str] = None


@dataclass
class FiscalLogRecord:
    event_date: str
    platform: str
    event_type: str
    gross_amount: float
    units: int = 1
    set_id: Optional[str] = None
    shipping_cost: float = 0.0
    fees: float = 0.0
    notes: Optional[str] = None


class LegoHunterRepository:
    """Supabase data access layer with retry-safe helpers."""

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        *,
        max_retries: int = 3,
        retry_base_delay: float = 1.5,
    ) -> None:
        self.client: Client = create_client(supabase_url, supabase_key)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    @classmethod
    def from_env(cls) -> "LegoHunterRepository":
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise RuntimeError("Missing SUPABASE_URL/SUPABASE_KEY environment variables")

        return cls(supabase_url=supabase_url, supabase_key=supabase_key)

    def _with_retry(self, operation_name: str, fn):
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.7)
                LOGGER.warning(
                    "Operation '%s' failed (attempt %s/%s): %s. Retrying in %.2fs",
                    operation_name,
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(f"Supabase operation '{operation_name}' failed") from last_exc

    def upsert_opportunity(self, record: OpportunityRadarRecord) -> Dict[str, Any]:
        payload = asdict(record)
        now_iso = datetime.now(timezone.utc).isoformat()
        payload["discovered_at"] = payload.get("discovered_at") or now_iso
        payload["last_seen_at"] = now_iso

        result = self._with_retry(
            "upsert_opportunity",
            lambda: self.client.table("opportunity_radar")
            .upsert(payload, on_conflict="set_id,source")
            .execute(),
        )
        data = result.data or []
        return data[0] if data else payload

    def get_top_opportunities(self, limit: int = 3, min_score: int = 60) -> list[Dict[str, Any]]:
        result = self._with_retry(
            "get_top_opportunities",
            lambda: self.client.table("opportunity_radar")
            .select("*")
            .eq("is_archived", False)
            .gte("ai_investment_score", min_score)
            .order("ai_investment_score", desc=True)
            .order("market_demand_score", desc=True)
            .order("last_seen_at", desc=True)
            .limit(limit)
            .execute(),
        )
        return result.data or []

    def search_opportunities(self, query_text: str, limit: int = 10) -> list[Dict[str, Any]]:
        safe_query = query_text.strip()
        if not safe_query:
            return []

        result = self._with_retry(
            "search_opportunities",
            lambda: self.client.table("opportunity_radar")
            .select("*")
            .eq("is_archived", False)
            .or_(f"set_name.ilike.%{safe_query}%,set_id.ilike.%{safe_query}%,theme.ilike.%{safe_query}%")
            .order("ai_investment_score", desc=True)
            .order("last_seen_at", desc=True)
            .limit(limit)
            .execute(),
        )
        return result.data or []

    def insert_market_snapshot(self, record: MarketTimeSeriesRecord) -> Dict[str, Any]:
        payload = asdict(record)
        payload["recorded_at"] = payload.get("recorded_at") or datetime.now(timezone.utc).isoformat()

        result = self._with_retry(
            "insert_market_snapshot",
            lambda: self.client.table("market_time_series").insert(payload).execute(),
        )
        data = result.data or []
        return data[0] if data else payload

    def insert_market_snapshots(self, records: Iterable[MarketTimeSeriesRecord]) -> list[Dict[str, Any]]:
        payloads = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for record in records:
            payload = asdict(record)
            payload["recorded_at"] = payload.get("recorded_at") or now_iso
            payloads.append(payload)

        if not payloads:
            return []

        result = self._with_retry(
            "insert_market_snapshots",
            lambda: self.client.table("market_time_series").insert(payloads).execute(),
        )
        return result.data or payloads

    def upsert_portfolio_item(self, record: PortfolioRecord) -> Dict[str, Any]:
        payload = asdict(record)
        result = self._with_retry(
            "upsert_portfolio_item",
            lambda: self.client.table("portfolio").upsert(payload, on_conflict="set_id").execute(),
        )
        data = result.data or []
        return data[0] if data else payload

    def get_portfolio(self, status: str = "holding") -> list[Dict[str, Any]]:
        result = self._with_retry(
            "get_portfolio",
            lambda: self.client.table("portfolio")
            .select("*")
            .eq("status", status)
            .order("purchase_date", desc=False)
            .execute(),
        )
        return result.data or []

    def get_latest_price(self, set_id: str, platform: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = self.client.table("market_time_series").select("*").eq("set_id", set_id)
        if platform:
            query = query.eq("platform", platform)

        result = self._with_retry(
            "get_latest_price",
            lambda: query.order("recorded_at", desc=True).limit(1).execute(),
        )
        rows = result.data or []
        return rows[0] if rows else None

    def get_best_secondary_price(self, set_id: str) -> Optional[Dict[str, Any]]:
        result = self._with_retry(
            "get_best_secondary_price",
            lambda: self.client.table("market_time_series")
            .select("*")
            .eq("set_id", set_id)
            .in_("platform", ["vinted", "subito"])
            .order("price", desc=False)
            .limit(1)
            .execute(),
        )
        rows = result.data or []
        return rows[0] if rows else None

    def get_recent_market_prices(
        self,
        set_id: str,
        *,
        days: int = 30,
        platform: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        since = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)
        since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()

        query = (
            self.client.table("market_time_series")
            .select("set_id,platform,price,recorded_at,listing_type")
            .eq("set_id", set_id)
            .gte("recorded_at", since_iso)
        )
        if platform:
            query = query.eq("platform", platform)

        result = self._with_retry(
            "get_recent_market_prices",
            lambda: query.order("recorded_at", desc=True).execute(),
        )
        return result.data or []

    def get_recent_opportunities(
        self,
        *,
        days: int = 365,
        limit: int = 2000,
        include_archived: bool = False,
    ) -> list[Dict[str, Any]]:
        since = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)
        since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()

        query = (
            self.client.table("opportunity_radar")
            .select(
                "set_id,set_name,theme,source,current_price,ai_investment_score,"
                "market_demand_score,metadata,discovered_at,last_seen_at,eol_date_prediction"
            )
            .gte("discovered_at", since_iso)
            .order("discovered_at", desc=True)
            .limit(limit)
        )
        if not include_archived:
            query = query.eq("is_archived", False)

        result = self._with_retry("get_recent_opportunities", lambda: query.execute())
        return result.data or []

    def get_market_history_for_sets(
        self,
        set_ids: Iterable[str],
        *,
        days: int = 540,
    ) -> list[Dict[str, Any]]:
        unique_ids = [str(item).strip() for item in set_ids if str(item).strip()]
        if not unique_ids:
            return []

        since = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)
        since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
        collected: list[Dict[str, Any]] = []

        for chunk in self._chunks(unique_ids, chunk_size=80):
            query = (
                self.client.table("market_time_series")
                .select("set_id,platform,price,recorded_at,listing_type")
                .in_("set_id", chunk)
                .gte("recorded_at", since_iso)
                .order("recorded_at", desc=False)
            )
            result = self._with_retry("get_market_history_for_sets", lambda: query.execute())
            collected.extend(result.data or [])

        return collected

    @staticmethod
    def _chunks(items: list[str], *, chunk_size: int) -> Iterable[list[str]]:
        if chunk_size <= 0:
            yield items
            return
        for idx in range(0, len(items), chunk_size):
            yield items[idx : idx + chunk_size]

    def get_theme_radar_baseline(
        self,
        theme: str,
        *,
        days: int = 180,
        limit: int = 120,
    ) -> Dict[str, float]:
        safe_theme = str(theme or "").strip()
        if not safe_theme:
            return {
                "sample_size": 0.0,
                "avg_ai_score": 0.0,
                "avg_market_demand": 0.0,
                "std_ai_score": 0.0,
            }

        since = datetime.now(timezone.utc).timestamp() - (days * 24 * 60 * 60)
        since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
        query = (
            self.client.table("opportunity_radar")
            .select("ai_investment_score,market_demand_score,last_seen_at")
            .eq("is_archived", False)
            .ilike("theme", safe_theme)
            .gte("last_seen_at", since_iso)
            .order("last_seen_at", desc=True)
            .limit(limit)
        )

        result = self._with_retry("get_theme_radar_baseline", lambda: query.execute())
        rows = result.data or []
        if not rows:
            return {
                "sample_size": 0.0,
                "avg_ai_score": 0.0,
                "avg_market_demand": 0.0,
                "std_ai_score": 0.0,
            }

        ai_scores = [float(row.get("ai_investment_score") or 0.0) for row in rows]
        demand_scores = [float(row.get("market_demand_score") or 0.0) for row in rows]
        std_ai_score = statistics.pstdev(ai_scores) if len(ai_scores) > 1 else 0.0

        return {
            "sample_size": float(len(rows)),
            "avg_ai_score": float(statistics.fmean(ai_scores)),
            "avg_market_demand": float(statistics.fmean(demand_scores)),
            "std_ai_score": float(std_ai_score),
        }

    def insert_fiscal_log(self, record: FiscalLogRecord) -> Dict[str, Any]:
        payload = asdict(record)
        result = self._with_retry(
            "insert_fiscal_log",
            lambda: self.client.table("fiscal_log").insert(payload).execute(),
        )
        data = result.data or []
        return data[0] if data else payload

    def get_fiscal_sales_summary(
        self,
        *,
        year: Optional[int] = None,
        platform: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        if year is None:
            year = date.today().year

        start = date(year, 1, 1).isoformat()
        end = date(year, 12, 31).isoformat()

        query = (
            self.client.table("fiscal_log")
            .select("platform,gross_amount,event_date")
            .eq("event_type", "sell")
            .gte("event_date", start)
            .lte("event_date", end)
        )
        if platform:
            query = query.eq("platform", platform)

        result = self._with_retry("get_fiscal_sales_summary", lambda: query.execute())
        summary: Dict[str, Dict[str, float]] = {}

        for row in result.data or []:
            row_platform = row.get("platform") or "unknown"
            gross_amount = float(row.get("gross_amount") or 0.0)
            summary.setdefault(row_platform, {"transactions": 0.0, "gross_total": 0.0})
            summary[row_platform]["transactions"] += 1
            summary[row_platform]["gross_total"] += gross_amount

        summary["_all"] = {
            "transactions": sum(item["transactions"] for item in summary.values()),
            "gross_total": sum(item["gross_total"] for item in summary.values()),
        }
        return summary


__all__ = [
    "LegoHunterRepository",
    "OpportunityRadarRecord",
    "MarketTimeSeriesRecord",
    "PortfolioRecord",
    "FiscalLogRecord",
]
