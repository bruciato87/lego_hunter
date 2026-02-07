from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional


@dataclass
class BacktestObservation:
    set_id: str
    discovery_date: date
    base_price: float
    composite_score: int
    probability_upside_pct: float
    confidence_score: int
    ai_fallback_used: bool
    realized_roi_pct: float
    hit_target: bool


@dataclass
class BacktestMetrics:
    sample_size: int
    selected_count: int
    precision: float
    recall: float
    f1: float
    coverage: float
    precision_at_k: float
    avg_realized_roi_selected: float
    brier_score: float
    threshold_composite: int
    threshold_probability: float
    threshold_confidence: int


@dataclass
class ThresholdSuggestion:
    composite_score: int
    probability_upside: float
    confidence_score: int
    objective_score: float
    metrics: BacktestMetrics
    changed: bool


@dataclass
class BacktestReport:
    observations: list[BacktestObservation]
    lookback_days: int
    horizon_days: int
    target_roi_pct: float

    @property
    def sample_size(self) -> int:
        return len(self.observations)


class OpportunityBacktester:
    def __init__(
        self,
        *,
        target_roi_pct: float = 30.0,
        horizon_days: int = 180,
        top_k: int = 3,
    ) -> None:
        self.target_roi_pct = float(max(5.0, min(150.0, target_roi_pct)))
        self.horizon_days = int(max(30, min(540, horizon_days)))
        self.top_k = int(max(1, min(10, top_k)))

    def run(
        self,
        *,
        repository,
        lookback_days: int = 365,
        max_opportunities: int = 2000,
    ) -> BacktestReport:
        opportunities = repository.get_recent_opportunities(
            days=lookback_days,
            limit=max_opportunities,
            include_archived=False,
        )
        set_ids = sorted({str(row.get("set_id") or "").strip() for row in opportunities if row.get("set_id")})
        if not opportunities or not set_ids:
            return BacktestReport(
                observations=[],
                lookback_days=lookback_days,
                horizon_days=self.horizon_days,
                target_roi_pct=self.target_roi_pct,
            )

        history_rows = repository.get_market_history_for_sets(
            set_ids,
            days=lookback_days + self.horizon_days + 30,
        )
        history_by_set: Dict[str, list[Dict[str, Any]]] = {}
        for row in history_rows:
            set_id = str(row.get("set_id") or "").strip()
            if not set_id:
                continue
            history_by_set.setdefault(set_id, []).append(row)

        observations: list[BacktestObservation] = []
        for row in opportunities:
            set_id = str(row.get("set_id") or "").strip()
            if not set_id:
                continue

            discovery_dt = self._parse_datetime(row.get("discovered_at")) or self._parse_datetime(row.get("last_seen_at"))
            if discovery_dt is None:
                continue

            base_price = self._safe_float(row.get("current_price"))
            if base_price is None or base_price <= 0:
                continue

            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            composite = self._to_int(
                row.get("composite_score") or metadata.get("composite_score") or row.get("ai_investment_score") or 0
            )
            probability_pct = self._to_float(
                row.get("forecast_probability_upside_12m") or metadata.get("forecast_probability_upside_12m") or 0.0
            )
            confidence = self._to_int(row.get("confidence_score") or metadata.get("forecast_confidence_score") or 0)
            ai_fallback = bool(metadata.get("ai_fallback_used", False) or row.get("ai_fallback_used", False))

            realized_roi = self._realized_roi_for_window(
                base_price=base_price,
                discovered_at=discovery_dt,
                history_rows=history_by_set.get(set_id, []),
            )
            if realized_roi is None:
                continue

            observations.append(
                BacktestObservation(
                    set_id=set_id,
                    discovery_date=discovery_dt.date(),
                    base_price=base_price,
                    composite_score=max(1, min(100, composite)),
                    probability_upside_pct=max(0.0, min(100.0, probability_pct)),
                    confidence_score=max(0, min(100, confidence)),
                    ai_fallback_used=ai_fallback,
                    realized_roi_pct=realized_roi,
                    hit_target=realized_roi >= self.target_roi_pct,
                )
            )

        return BacktestReport(
            observations=observations,
            lookback_days=lookback_days,
            horizon_days=self.horizon_days,
            target_roi_pct=self.target_roi_pct,
        )

    def evaluate(
        self,
        *,
        observations: Iterable[BacktestObservation],
        threshold_composite: int,
        threshold_probability: float,
        threshold_confidence: int,
    ) -> BacktestMetrics:
        rows = list(observations)
        sample_size = len(rows)
        if sample_size == 0:
            return BacktestMetrics(
                sample_size=0,
                selected_count=0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                coverage=0.0,
                precision_at_k=0.0,
                avg_realized_roi_selected=0.0,
                brier_score=1.0,
                threshold_composite=threshold_composite,
                threshold_probability=threshold_probability,
                threshold_confidence=threshold_confidence,
            )

        predicted = [row for row in rows if self._is_selected(row, threshold_composite, threshold_probability, threshold_confidence)]
        selected_count = len(predicted)
        actual_positive = sum(1 for row in rows if row.hit_target)
        tp = sum(1 for row in predicted if row.hit_target)
        fp = selected_count - tp
        fn = actual_positive - tp

        precision = tp / selected_count if selected_count > 0 else 0.0
        recall = tp / actual_positive if actual_positive > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        coverage = selected_count / sample_size if sample_size > 0 else 0.0
        avg_roi_selected = statistics.fmean(row.realized_roi_pct for row in predicted) if predicted else 0.0
        precision_at_k = self._precision_at_k(predicted)
        brier = self._brier_score(rows)

        return BacktestMetrics(
            sample_size=sample_size,
            selected_count=selected_count,
            precision=precision,
            recall=recall,
            f1=f1,
            coverage=coverage,
            precision_at_k=precision_at_k,
            avg_realized_roi_selected=avg_roi_selected,
            brier_score=brier,
            threshold_composite=threshold_composite,
            threshold_probability=threshold_probability,
            threshold_confidence=threshold_confidence,
        )

    def tune_thresholds(
        self,
        *,
        observations: Iterable[BacktestObservation],
        current_composite: int,
        current_probability: float,
        current_confidence: int,
        min_selected: int = 15,
    ) -> ThresholdSuggestion:
        rows = list(observations)
        if len(rows) < max(20, min_selected):
            metrics = self.evaluate(
                observations=rows,
                threshold_composite=current_composite,
                threshold_probability=current_probability,
                threshold_confidence=current_confidence,
            )
            return ThresholdSuggestion(
                composite_score=current_composite,
                probability_upside=current_probability,
                confidence_score=current_confidence,
                objective_score=self._objective(metrics),
                metrics=metrics,
                changed=False,
            )

        composite_grid = [45, 50, 55, 60, 65, 70, 75, 80]
        probability_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        confidence_grid = [45, 50, 55, 60, 65, 70, 75, 80]

        best_metrics: Optional[BacktestMetrics] = None
        best_objective = -math.inf

        for composite_thr in composite_grid:
            for prob_thr in probability_grid:
                for conf_thr in confidence_grid:
                    metrics = self.evaluate(
                        observations=rows,
                        threshold_composite=composite_thr,
                        threshold_probability=prob_thr,
                        threshold_confidence=conf_thr,
                    )
                    if metrics.selected_count < min_selected:
                        continue
                    score = self._objective(metrics)
                    if score > best_objective:
                        best_objective = score
                        best_metrics = metrics

        if best_metrics is None:
            best_metrics = self.evaluate(
                observations=rows,
                threshold_composite=current_composite,
                threshold_probability=current_probability,
                threshold_confidence=current_confidence,
            )
            best_objective = self._objective(best_metrics)

        changed = (
            best_metrics.threshold_composite != current_composite
            or abs(best_metrics.threshold_probability - current_probability) > 1e-9
            or best_metrics.threshold_confidence != current_confidence
        )
        return ThresholdSuggestion(
            composite_score=best_metrics.threshold_composite,
            probability_upside=best_metrics.threshold_probability,
            confidence_score=best_metrics.threshold_confidence,
            objective_score=best_objective,
            metrics=best_metrics,
            changed=changed,
        )

    def _realized_roi_for_window(
        self,
        *,
        base_price: float,
        discovered_at: datetime,
        history_rows: list[Dict[str, Any]],
    ) -> Optional[float]:
        if base_price <= 0:
            return None
        window_end = discovered_at + timedelta(days=self.horizon_days)

        secondary_prices: list[float] = []
        all_prices: list[float] = []

        for row in history_rows:
            recorded_at = self._parse_datetime(row.get("recorded_at"))
            if recorded_at is None:
                continue
            if recorded_at <= discovered_at or recorded_at > window_end:
                continue
            price = self._safe_float(row.get("price"))
            if price is None or price <= 0:
                continue
            all_prices.append(price)
            platform = str(row.get("platform") or "").lower()
            if platform in {"vinted", "subito"}:
                secondary_prices.append(price)

        future_prices = secondary_prices if secondary_prices else all_prices
        if not future_prices:
            return None

        max_future = max(future_prices)
        return ((max_future - base_price) / base_price) * 100.0

    def _precision_at_k(self, predicted_rows: list[BacktestObservation]) -> float:
        if not predicted_rows:
            return 0.0
        grouped: Dict[date, list[BacktestObservation]] = {}
        for row in predicted_rows:
            grouped.setdefault(row.discovery_date, []).append(row)

        hits = 0
        count = 0
        for day_rows in grouped.values():
            ranked = sorted(
                day_rows,
                key=lambda item: (
                    item.composite_score,
                    item.probability_upside_pct,
                    item.confidence_score,
                ),
                reverse=True,
            )
            for row in ranked[: self.top_k]:
                count += 1
                if row.hit_target:
                    hits += 1
        if count == 0:
            return 0.0
        return hits / count

    @staticmethod
    def _brier_score(rows: list[BacktestObservation]) -> float:
        if not rows:
            return 1.0
        errors: list[float] = []
        for row in rows:
            p = max(0.0, min(1.0, row.probability_upside_pct / 100.0))
            y = 1.0 if row.hit_target else 0.0
            errors.append((p - y) ** 2)
        return statistics.fmean(errors) if errors else 1.0

    @staticmethod
    def _objective(metrics: BacktestMetrics) -> float:
        calibration = 1.0 - max(0.0, min(1.0, metrics.brier_score))
        return (
            (metrics.precision_at_k * 0.45)
            + (metrics.precision * 0.30)
            + (metrics.recall * 0.10)
            + (calibration * 0.10)
            + (max(0.0, min(1.0, metrics.coverage / 0.35)) * 0.05)
        )

    @staticmethod
    def _is_selected(
        row: BacktestObservation,
        threshold_composite: int,
        threshold_probability: float,
        threshold_confidence: int,
    ) -> bool:
        if row.ai_fallback_used:
            return False
        return (
            row.composite_score >= threshold_composite
            and row.probability_upside_pct >= (threshold_probability * 100.0)
            and row.confidence_score >= threshold_confidence
        )

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
    def _safe_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


__all__ = [
    "BacktestObservation",
    "BacktestMetrics",
    "ThresholdSuggestion",
    "BacktestReport",
    "OpportunityBacktester",
]
