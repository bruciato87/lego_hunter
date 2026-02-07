from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, Optional


@dataclass
class ForecastInsight:
    forecast_score: int
    probability_upside_12m: float
    expected_roi_12m_pct: float
    interval_low_pct: float
    interval_high_pct: float
    target_roi_pct: float
    estimated_months_to_target: Optional[int]
    confidence_score: int
    data_points: int
    rationale: str


class InvestmentForecaster:
    """Quantitative forecaster driven by the time-series data moat."""

    def __init__(self, *, target_roi_pct: float = 30.0) -> None:
        self.target_roi_pct = float(max(5.0, min(150.0, target_roi_pct)))

    def forecast(
        self,
        *,
        candidate: Dict[str, Any],
        history_rows: Iterable[Dict[str, Any]],
        theme_baseline: Optional[Dict[str, float]] = None,
    ) -> ForecastInsight:
        rows = [row for row in history_rows if self._safe_float(row.get("price")) is not None]
        prices = [self._safe_float(row.get("price")) or 0.0 for row in rows]
        rows_sorted = sorted(
            rows,
            key=lambda row: self._parse_datetime(row.get("recorded_at")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        now_utc = datetime.now(timezone.utc)
        prices_30 = self._prices_in_window(rows_sorted, now_utc=now_utc, days=30)
        prices_90 = self._prices_in_window(rows_sorted, now_utc=now_utc, days=90)
        secondary_30 = self._prices_in_window(
            rows_sorted,
            now_utc=now_utc,
            days=30,
            platforms={"vinted", "subito"},
        )

        median_30 = self._median_or_none(prices_30)
        median_90 = self._median_or_none(prices_90)
        median_secondary_30 = self._median_or_none(secondary_30)
        momentum = self._safe_ratio((median_30 or 0.0) - (median_90 or 0.0), median_90)

        volatility = 0.0
        if len(prices_90) >= 3:
            mean_90 = statistics.fmean(prices_90)
            if mean_90 > 0:
                volatility = min(2.0, statistics.pstdev(prices_90) / mean_90)

        unique_days_90 = len(
            {
                self._parse_datetime(row.get("recorded_at")).date()
                for row in rows_sorted
                if self._parse_datetime(row.get("recorded_at"))
                and (now_utc - self._parse_datetime(row.get("recorded_at"))).days <= 90
            }
        )
        platform_diversity = len({str(row.get("platform") or "").lower() for row in rows_sorted if row.get("platform")})
        liquidity = min(1.0, (len(prices_30) / 30.0) * 0.55 + (unique_days_90 / 60.0) * 0.45)

        primary_price = self._safe_float(candidate.get("current_price"))
        secondary_gap_pct = 0.0
        if primary_price and primary_price > 0 and median_secondary_30:
            secondary_gap_pct = ((median_secondary_30 - primary_price) / primary_price) * 100.0

        days_to_eol = self._days_to_eol(candidate.get("eol_date_prediction"))
        eol_urgency = self._eol_urgency(days_to_eol)

        theme_tailwind = self._theme_tailwind(theme_baseline or {})
        confidence = self._confidence_score(
            rows=rows_sorted,
            unique_days_90=unique_days_90,
            platform_diversity=platform_diversity,
            eol_known=days_to_eol is not None,
            now_utc=now_utc,
        )

        expected_roi = self._expected_roi(
            momentum=momentum,
            volatility=volatility,
            secondary_gap_pct=secondary_gap_pct,
            eol_urgency=eol_urgency,
            liquidity=liquidity,
            theme_tailwind=theme_tailwind,
        )
        probability = self._probability_upside(
            expected_roi=expected_roi,
            confidence_score=confidence,
            liquidity=liquidity,
            volatility=volatility,
        )

        uncertainty = 10.0 + (100 - confidence) * 0.22 + volatility * 18.0
        interval_low = expected_roi - uncertainty
        interval_high = expected_roi + uncertainty

        forecast_score = int(
            round(
                self._clamp(
                    0.52 * (probability * 100.0)
                    + 0.28 * self._clamp(expected_roi, 0.0, 100.0)
                    + 0.20 * confidence,
                    1.0,
                    100.0,
                )
            )
        )
        months_to_target = self._estimate_months_to_target(expected_roi)

        rationale = self._build_rationale(
            momentum=momentum,
            secondary_gap_pct=secondary_gap_pct,
            eol_urgency=eol_urgency,
            liquidity=liquidity,
            volatility=volatility,
            confidence=confidence,
            sample_size=len(prices),
        )

        return ForecastInsight(
            forecast_score=forecast_score,
            probability_upside_12m=probability,
            expected_roi_12m_pct=round(expected_roi, 2),
            interval_low_pct=round(interval_low, 2),
            interval_high_pct=round(interval_high, 2),
            target_roi_pct=self.target_roi_pct,
            estimated_months_to_target=months_to_target,
            confidence_score=confidence,
            data_points=len(prices),
            rationale=rationale,
        )

    def _expected_roi(
        self,
        *,
        momentum: float,
        volatility: float,
        secondary_gap_pct: float,
        eol_urgency: float,
        liquidity: float,
        theme_tailwind: float,
    ) -> float:
        momentum_norm = self._clamp((momentum + 0.2) / 0.6, 0.0, 1.0)
        gap_norm = self._clamp((secondary_gap_pct + 10.0) / 35.0, 0.0, 1.0)

        expected = (
            6.5
            + 27.0 * eol_urgency
            + 18.0 * momentum_norm
            + 14.0 * gap_norm
            + 12.0 * liquidity
            + 10.0 * theme_tailwind
            - 17.0 * min(1.0, volatility)
        )
        return self._clamp(expected, -20.0, 180.0)

    def _probability_upside(
        self,
        *,
        expected_roi: float,
        confidence_score: int,
        liquidity: float,
        volatility: float,
    ) -> float:
        z = (
            -1.2
            + (expected_roi / 35.0)
            + ((confidence_score - 50.0) / 36.0)
            + (liquidity - 0.45) * 1.1
            - min(1.0, volatility) * 0.9
        )
        prob = 1.0 / (1.0 + math.exp(-z))
        return self._clamp(prob, 0.02, 0.98)

    @staticmethod
    def _theme_tailwind(theme_baseline: Dict[str, float]) -> float:
        sample = float(theme_baseline.get("sample_size") or 0.0)
        avg_ai = float(theme_baseline.get("avg_ai_score") or 0.0)
        avg_demand = float(theme_baseline.get("avg_market_demand") or 0.0)
        if sample <= 0:
            return 0.5

        ai_component = InvestmentForecaster._clamp(avg_ai / 100.0, 0.0, 1.0)
        demand_component = InvestmentForecaster._clamp(avg_demand / 100.0, 0.0, 1.0)
        sample_component = InvestmentForecaster._clamp(sample / 40.0, 0.0, 1.0)
        return InvestmentForecaster._clamp(
            (ai_component * 0.45) + (demand_component * 0.35) + (sample_component * 0.20),
            0.0,
            1.0,
        )

    @staticmethod
    def _confidence_score(
        *,
        rows: list[Dict[str, Any]],
        unique_days_90: int,
        platform_diversity: int,
        eol_known: bool,
        now_utc: datetime,
    ) -> int:
        sample_size = len(rows)
        sample_component = InvestmentForecaster._clamp(sample_size / 60.0, 0.0, 1.0)
        day_component = InvestmentForecaster._clamp(unique_days_90 / 45.0, 0.0, 1.0)
        platform_component = InvestmentForecaster._clamp(platform_diversity / 3.0, 0.0, 1.0)
        eol_component = 1.0 if eol_known else 0.65

        recency_component = 0.2
        if rows:
            latest = InvestmentForecaster._parse_datetime(rows[0].get("recorded_at"))
            if latest:
                age_days = max(0, (now_utc - latest).days)
                if age_days <= 2:
                    recency_component = 1.0
                elif age_days <= 7:
                    recency_component = 0.85
                elif age_days <= 15:
                    recency_component = 0.65
                elif age_days <= 30:
                    recency_component = 0.45

        score = (
            0.32 * sample_component
            + 0.22 * day_component
            + 0.16 * platform_component
            + 0.18 * recency_component
            + 0.12 * eol_component
        )
        return int(round(InvestmentForecaster._clamp(score, 0.0, 1.0) * 100.0))

    @staticmethod
    def _days_to_eol(raw_date: Any) -> Optional[int]:
        if not raw_date:
            return None
        try:
            value = date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            return None
        return (value - date.today()).days

    @staticmethod
    def _eol_urgency(days_to_eol: Optional[int]) -> float:
        if days_to_eol is None:
            return 0.38
        if days_to_eol <= 0:
            return 1.0
        return InvestmentForecaster._clamp((180.0 - float(days_to_eol)) / 180.0, 0.0, 1.0)

    def _estimate_months_to_target(self, expected_roi_12m_pct: float) -> Optional[int]:
        if expected_roi_12m_pct <= 0:
            return None
        monthly_growth = expected_roi_12m_pct / 12.0
        if monthly_growth <= 0:
            return None
        months = int(round(self.target_roi_pct / monthly_growth))
        return int(self._clamp(float(months), 1.0, 36.0))

    @staticmethod
    def _prices_in_window(
        rows: list[Dict[str, Any]],
        *,
        now_utc: datetime,
        days: int,
        platforms: Optional[set[str]] = None,
    ) -> list[float]:
        results: list[float] = []
        platform_filter = {item.lower() for item in platforms} if platforms else None
        for row in rows:
            parsed = InvestmentForecaster._parse_datetime(row.get("recorded_at"))
            if not parsed:
                continue
            age_days = (now_utc - parsed).days
            if age_days < 0 or age_days > days:
                continue
            platform = str(row.get("platform") or "").lower()
            if platform_filter and platform not in platform_filter:
                continue
            value = InvestmentForecaster._safe_float(row.get("price"))
            if value is None or value <= 0:
                continue
            results.append(value)
        return results

    @staticmethod
    def _median_or_none(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return float(statistics.median(values))

    @staticmethod
    def _safe_ratio(num: float, den: Optional[float]) -> float:
        if den is None or den == 0:
            return 0.0
        return num / den

    @staticmethod
    def _safe_float(raw: Any) -> Optional[float]:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    @staticmethod
    def _parse_datetime(raw: Any) -> Optional[datetime]:
        if not raw:
            return None
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _build_rationale(
        *,
        momentum: float,
        secondary_gap_pct: float,
        eol_urgency: float,
        liquidity: float,
        volatility: float,
        confidence: int,
        sample_size: int,
    ) -> str:
        trend = "rialzista" if momentum >= 0.03 else ("in consolidamento" if momentum >= -0.02 else "debole")
        gap = "premio sul secondario" if secondary_gap_pct >= 0 else "sconto sul secondario"
        eol_hint = "EOL vicino" if eol_urgency >= 0.7 else ("EOL intermedio" if eol_urgency >= 0.4 else "EOL lontano")
        liquidity_hint = "liquidita buona" if liquidity >= 0.6 else "liquidita limitata"
        risk_hint = "volatilita alta" if volatility >= 0.45 else "volatilita contenuta"
        return (
            f"Trend {trend}; {gap}; {eol_hint}; {liquidity_hint}; {risk_hint}. "
            f"Confidenza dati {confidence}/100 su {sample_size} snapshot."
        )


__all__ = ["InvestmentForecaster", "ForecastInsight"]
