from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

from models import LegoHunterRepository


@dataclass(frozen=True)
class FiscalThresholds:
    warning_transactions: int = 25
    warning_amount: float = 1800.0
    critical_transactions: int = 29
    critical_amount: float = 1990.0


class FiscalGuardian:
    """DAC7 guardrail: blocks SELL signals near reporting thresholds."""

    def __init__(
        self,
        repository: LegoHunterRepository,
        thresholds: FiscalThresholds = FiscalThresholds(),
    ) -> None:
        self.repository = repository
        self.thresholds = thresholds

    def check_safety_status(self, reference_date: Optional[date] = None) -> Dict[str, Any]:
        if reference_date is None:
            reference_date = date.today()

        summary = self.repository.get_fiscal_sales_summary(year=reference_date.year)
        all_channels = summary.get("_all", {"transactions": 0.0, "gross_total": 0.0})
        total_transactions = int(all_channels.get("transactions", 0.0))
        total_gross = float(all_channels.get("gross_total", 0.0))

        status = "GREEN"
        if (
            total_transactions >= self.thresholds.critical_transactions
            or total_gross >= self.thresholds.critical_amount
        ):
            status = "RED"
        elif (
            total_transactions > self.thresholds.warning_transactions
            or total_gross > self.thresholds.warning_amount
        ):
            status = "YELLOW"

        return {
            "year": reference_date.year,
            "status": status,
            "allow_buy_signals": True,
            "allow_sell_signals": status == "GREEN",
            "totals": {
                "transactions": total_transactions,
                "gross_amount": round(total_gross, 2),
            },
            "thresholds": {
                "warning": {
                    "transactions": self.thresholds.warning_transactions,
                    "gross_amount": self.thresholds.warning_amount,
                },
                "critical": {
                    "transactions": self.thresholds.critical_transactions,
                    "gross_amount": self.thresholds.critical_amount,
                },
            },
            "per_platform": {
                key: value for key, value in summary.items() if key != "_all"
            },
            "message": self._build_message(status, total_transactions, total_gross),
        }

    def _build_message(self, status: str, transactions: int, gross_total: float) -> str:
        if status == "RED":
            return (
                "STOP ROSSO DAC7: vendite bloccate. "
                f"Transazioni={transactions}, Fatturato={gross_total:.2f} EUR."
            )
        if status == "YELLOW":
            return (
                "WARNING DAC7: vicinanza ai limiti, vendite bloccate in sicurezza. "
                f"Transazioni={transactions}, Fatturato={gross_total:.2f} EUR."
            )
        return (
            "Stato fiscale sicuro: segnali di acquisto e vendita consentiti. "
            f"Transazioni={transactions}, Fatturato={gross_total:.2f} EUR."
        )


__all__ = ["FiscalGuardian", "FiscalThresholds"]
