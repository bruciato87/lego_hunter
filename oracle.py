from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional

try:
    import google.generativeai as genai
except Exception:  # noqa: BLE001
    genai = None

from models import LegoHunterRepository, MarketTimeSeriesRecord, OpportunityRadarRecord
from scrapers import AmazonBestsellerScraper, LegoRetiringScraper, SecondaryMarketValidator

LOGGER = logging.getLogger(__name__)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AIInsight:
    score: int
    summary: str
    predicted_eol_date: Optional[str] = None


class DiscoveryOracle:
    """Discovery engine for identifying potential LEGO opportunities."""

    def __init__(
        self,
        repository: LegoHunterRepository,
        *,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-1.5-flash",
        min_ai_score: int = 60,
    ) -> None:
        self.repository = repository
        self.min_ai_score = min_ai_score
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model

        self._model = None
        if self.gemini_api_key and genai is not None:
            genai.configure(api_key=self.gemini_api_key)
            self._model = genai.GenerativeModel(self.gemini_model)
        else:
            if not self.gemini_api_key:
                LOGGER.warning("Gemini API key missing: fallback heuristic scoring will be used")
            elif genai is None:
                LOGGER.warning(
                    "google-generativeai package not installed: fallback heuristic scoring will be used"
                )

    async def discover_opportunities(self, *, persist: bool = True, top_limit: int = 25) -> list[Dict[str, Any]]:
        """Scan sources, enrich with AI score, and store top opportunities."""
        source_candidates = await self._collect_source_candidates()
        if not source_candidates:
            return []

        ranked: list[Dict[str, Any]] = []
        for candidate in source_candidates:
            ai = await self._get_ai_insight(candidate)
            demand = self._estimate_market_demand(candidate, ai.score)

            opportunity = OpportunityRadarRecord(
                set_id=candidate["set_id"],
                set_name=candidate["set_name"],
                theme=candidate.get("theme"),
                source=candidate.get("source", "unknown"),
                eol_date_prediction=ai.predicted_eol_date or candidate.get("eol_date_prediction"),
                market_demand_score=demand,
                ai_investment_score=ai.score,
                ai_analysis_summary=ai.summary,
                current_price=candidate.get("current_price"),
                metadata={
                    "listing_url": candidate.get("listing_url"),
                    "source_metadata": candidate.get("metadata", {}),
                },
            )

            payload = opportunity.__dict__.copy()
            payload["market_demand_score"] = demand
            payload["ai_investment_score"] = ai.score
            ranked.append(payload)

            if not persist:
                continue

            try:
                self.repository.upsert_opportunity(opportunity)
                if candidate.get("current_price") is not None:
                    self.repository.insert_market_snapshot(
                        MarketTimeSeriesRecord(
                            set_id=candidate["set_id"],
                            set_name=candidate["set_name"],
                            platform="lego" if "lego" in candidate.get("source", "") else "amazon",
                            listing_type="new",
                            price=float(candidate["current_price"]),
                            shipping_cost=0.0,
                            listing_url=candidate.get("listing_url"),
                            raw_payload=candidate,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to persist opportunity %s: %s", candidate.get("set_id"), exc)

        ranked.sort(key=lambda row: (row["ai_investment_score"], row["market_demand_score"]), reverse=True)
        filtered = [row for row in ranked if row["ai_investment_score"] >= self.min_ai_score]
        return filtered[:top_limit]

    async def validate_secondary_deals(self, opportunities: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Fetch Vinted/Subito offers for discovered sets and persist secondary snapshots."""
        if not opportunities:
            return []

        validator = SecondaryMarketValidator()
        results = await validator.compare_secondary_prices(opportunities)
        merged: list[Dict[str, Any]] = []

        for opportunity in opportunities:
            key = str(opportunity.get("set_id") or opportunity.get("set_name"))
            listings = results.get(key, [])
            primary_price = float(opportunity.get("current_price") or 0.0)

            best_secondary = None
            if listings:
                best_secondary = min(listings, key=lambda row: row.price)
                try:
                    self.repository.insert_market_snapshot(
                        MarketTimeSeriesRecord(
                            set_id=best_secondary.set_id,
                            set_name=best_secondary.set_name,
                            platform=best_secondary.platform,
                            listing_type=best_secondary.condition,
                            price=best_secondary.price,
                            shipping_cost=0.0,
                            listing_url=best_secondary.listing_url,
                            raw_payload={
                                "source_note": best_secondary.source_note,
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to save secondary snapshot %s: %s", key, exc)

            discount_pct = 0.0
            if best_secondary and primary_price > 0:
                discount_pct = ((primary_price - best_secondary.price) / primary_price) * 100

            merged.append(
                {
                    **opportunity,
                    "secondary_best_price": best_secondary.price if best_secondary else None,
                    "secondary_platform": best_secondary.platform if best_secondary else None,
                    "secondary_url": best_secondary.listing_url if best_secondary else None,
                    "discount_vs_primary_pct": round(discount_pct, 2),
                }
            )

        merged.sort(
            key=lambda row: (
                row.get("discount_vs_primary_pct") or 0.0,
                row.get("ai_investment_score") or 0,
            ),
            reverse=True,
        )
        return merged

    async def _collect_source_candidates(self) -> list[Dict[str, Any]]:
        async with LegoRetiringScraper() as lego_scraper, AmazonBestsellerScraper() as amazon_scraper:
            lego_task = lego_scraper.fetch_retiring_sets(limit=50)
            amazon_task = amazon_scraper.fetch_bestsellers(limit=50)
            lego_data, amazon_data = await asyncio.gather(lego_task, amazon_task, return_exceptions=True)

        candidates: list[Dict[str, Any]] = []

        if isinstance(lego_data, Exception):
            LOGGER.warning("Lego source failed: %s", lego_data)
        else:
            candidates.extend(lego_data)

        if isinstance(amazon_data, Exception):
            LOGGER.warning("Amazon source failed: %s", amazon_data)
        else:
            candidates.extend(amazon_data)

        dedup: Dict[str, Dict[str, Any]] = {}
        for row in candidates:
            set_id = str(row.get("set_id") or "").strip()
            if not set_id:
                continue

            current = dedup.get(set_id)
            if current is None:
                dedup[set_id] = row
                continue

            # Prefer official LEGO source when both exist.
            if current.get("source") != "lego_retiring" and row.get("source") == "lego_retiring":
                dedup[set_id] = row

        return list(dedup.values())

    async def _get_ai_insight(self, candidate: Dict[str, Any]) -> AIInsight:
        if self._model is None:
            return self._heuristic_ai_fallback(candidate)

        prompt = self._build_gemini_prompt(candidate)
        try:
            text = await asyncio.to_thread(self._gemini_generate, prompt)
            payload = self._extract_json(text)
            score = int(payload.get("score", 50))
            score = max(1, min(100, score))
            summary = str(payload.get("summary") or "No summary")[:1200]
            predicted_eol_date = payload.get("predicted_eol_date") or candidate.get("eol_date_prediction")
            return AIInsight(score=score, summary=summary, predicted_eol_date=predicted_eol_date)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Gemini scoring failed for %s: %s", candidate.get("set_id"), exc)
            return self._heuristic_ai_fallback(candidate)

    def _gemini_generate(self, prompt: str) -> str:
        if self._model is None:
            return "{}"

        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 400,
                "response_mime_type": "application/json",
            },
        )
        return (response.text or "").strip()

    @staticmethod
    def _build_gemini_prompt(candidate: Dict[str, Any]) -> str:
        return (
            "Analizza questo set LEGO per investimento a 12 mesi. "
            "Rispondi SOLO con JSON valido nel formato: "
            '{"score": 1-100, "summary": "max 3 frasi", "predicted_eol_date": "YYYY-MM-DD o null"}.\n\n'
            f"Set ID: {candidate.get('set_id')}\n"
            f"Nome: {candidate.get('set_name')}\n"
            f"Tema: {candidate.get('theme')}\n"
            f"Fonte: {candidate.get('source')}\n"
            f"Prezzo: {candidate.get('current_price')}\n"
            "Criteri: domanda collezionisti, brand power del tema, probabilita rivalutazione, velocita di rotazione."
        )

    def _heuristic_ai_fallback(self, candidate: Dict[str, Any]) -> AIInsight:
        name = str(candidate.get("set_name") or "")
        source = str(candidate.get("source") or "")
        price = float(candidate.get("current_price") or 0.0)

        base = 55
        if source == "lego_retiring":
            base += 18
        if any(key in name.lower() for key in ("star wars", "icons", "technic", "modular")):
            base += 12
        if 30 <= price <= 180:
            base += 6

        score = max(1, min(100, base))
        eol = candidate.get("eol_date_prediction") or (date.today() + timedelta(days=80)).isoformat()
        return AIInsight(
            score=score,
            summary=(
                "Fallback scoring: forte segnale su tema/sorgente; confermare con storico prezzi "
                "prima dell'acquisto definitivo."
            ),
            predicted_eol_date=eol,
        )

    def _estimate_market_demand(self, candidate: Dict[str, Any], ai_score: int) -> int:
        set_id = str(candidate.get("set_id") or "")
        if not set_id:
            return max(1, min(100, ai_score))

        try:
            recent = self.repository.get_recent_market_prices(set_id, days=30)
        except Exception:  # noqa: BLE001
            recent = []

        liquidity_factor = min(35, len(recent) * 3)
        source_bonus = 20 if candidate.get("source") == "lego_retiring" else 8
        final_score = int((ai_score * 0.65) + liquidity_factor + source_bonus)
        return max(1, min(100, final_score))

    @staticmethod
    def _extract_json(raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = JSON_RE.search(text)
            if not match:
                raise
            return json.loads(match.group(0))


__all__ = ["DiscoveryOracle"]
