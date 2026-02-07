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
        self.ai_runtime = {
            "engine": "heuristic",
            "model": "heuristic-v1",
            "mode": "fallback",
        }
        self._last_source_diagnostics: Dict[str, Any] = {
            "source_raw_counts": {"lego_retiring": 0, "amazon_bestsellers": 0},
            "source_dedup_counts": {},
            "source_failures": [],
            "dedup_candidates": 0,
            "anti_bot_alert": False,
            "anti_bot_message": None,
        }
        if self.gemini_api_key and genai is not None:
            genai.configure(api_key=self.gemini_api_key)
            self._model = genai.GenerativeModel(self.gemini_model)
            self.ai_runtime = {
                "engine": "gemini",
                "model": self.gemini_model,
                "mode": "api",
            }
        else:
            if not self.gemini_api_key:
                LOGGER.warning("Gemini API key missing: fallback heuristic scoring will be used")
                self.ai_runtime = {
                    "engine": "heuristic",
                    "model": "heuristic-v1",
                    "mode": "fallback_no_key",
                }
            elif genai is None:
                LOGGER.warning(
                    "google-generativeai package not installed: fallback heuristic scoring will be used"
                )
                self.ai_runtime = {
                    "engine": "heuristic",
                    "model": "heuristic-v1",
                    "mode": "fallback_missing_package",
                }

    async def discover_opportunities(
        self,
        *,
        persist: bool = True,
        top_limit: int = 25,
        include_low_confidence: bool = False,
    ) -> list[Dict[str, Any]]:
        """Scan sources, enrich with AI score, and return opportunities."""
        report = await self.discover_with_diagnostics(
            persist=persist,
            top_limit=top_limit,
            fallback_limit=3,
        )
        if include_low_confidence:
            return report["selected"][:top_limit]
        return report["above_threshold"][:top_limit]

    async def discover_with_diagnostics(
        self,
        *,
        persist: bool = True,
        top_limit: int = 25,
        fallback_limit: int = 3,
    ) -> Dict[str, Any]:
        """Run discovery and return picks plus execution diagnostics."""
        LOGGER.info(
            "Discovery start | persist=%s top_limit=%s fallback_limit=%s threshold=%s",
            persist,
            top_limit,
            fallback_limit,
            self.min_ai_score,
        )
        source_candidates = await self._collect_source_candidates()
        source_diagnostics = self._last_source_diagnostics
        LOGGER.info(
            "Discovery sources | raw=%s dedup=%s failures=%s anti_bot=%s",
            source_diagnostics.get("source_raw_counts"),
            source_diagnostics.get("dedup_candidates"),
            len(source_diagnostics.get("source_failures") or []),
            source_diagnostics.get("anti_bot_alert"),
        )
        ranked = await self._rank_and_persist_candidates(source_candidates, persist=persist)

        ranked.sort(key=lambda row: (row["ai_investment_score"], row["market_demand_score"]), reverse=True)
        above_threshold = [row for row in ranked if row["ai_investment_score"] >= self.min_ai_score]

        selected: list[Dict[str, Any]]
        fallback_used = False

        if above_threshold:
            selected = [
                {
                    **row,
                    "signal_strength": "HIGH_CONFIDENCE",
                }
                for row in above_threshold[:top_limit]
            ]
        else:
            fallback_used = bool(ranked)
            selected = [
                {
                    **row,
                    "signal_strength": "LOW_CONFIDENCE",
                    "risk_note": f"Nessun set sopra soglia {self.min_ai_score}.",
                }
                for row in ranked[:fallback_limit]
            ]

        diagnostics = {
            "threshold": self.min_ai_score,
            "source_raw_counts": source_diagnostics["source_raw_counts"],
            "source_dedup_counts": source_diagnostics["source_dedup_counts"],
            "source_failures": source_diagnostics["source_failures"],
            "dedup_candidates": source_diagnostics["dedup_candidates"],
            "ranked_candidates": len(ranked),
            "above_threshold_count": len(above_threshold),
            "below_threshold_count": len(ranked) - len(above_threshold),
            "max_ai_score": max((row["ai_investment_score"] for row in ranked), default=0),
            "fallback_used": fallback_used,
            "anti_bot_alert": source_diagnostics["anti_bot_alert"],
            "anti_bot_message": source_diagnostics["anti_bot_message"],
            "ai_runtime": self.ai_runtime,
        }

        if ranked:
            top_debug = [
                {
                    "set_id": row.get("set_id"),
                    "source": row.get("source"),
                    "ai": row.get("ai_investment_score"),
                    "demand": row.get("market_demand_score"),
                }
                for row in ranked[:3]
            ]
        else:
            top_debug = []

        LOGGER.info(
            "Discovery summary | ranked=%s above_threshold=%s fallback_used=%s max_ai=%s ai=%s top=%s",
            diagnostics["ranked_candidates"],
            diagnostics["above_threshold_count"],
            diagnostics["fallback_used"],
            diagnostics["max_ai_score"],
            diagnostics["ai_runtime"],
            top_debug,
        )
        if diagnostics["anti_bot_alert"]:
            LOGGER.warning("Discovery anti-bot alert | message=%s", diagnostics["anti_bot_message"])

        return {
            "selected": selected,
            "above_threshold": above_threshold[:top_limit],
            "ranked": ranked,
            "diagnostics": diagnostics,
        }

    async def _rank_and_persist_candidates(
        self,
        source_candidates: list[Dict[str, Any]],
        *,
        persist: bool,
    ) -> list[Dict[str, Any]]:
        if not source_candidates:
            LOGGER.info("Ranking skipped | no source candidates available")
            return []

        ranked: list[Dict[str, Any]] = []
        persisted_opportunities = 0
        persisted_snapshots = 0
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
                persisted_opportunities += 1
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
                    persisted_snapshots += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to persist opportunity %s: %s", candidate.get("set_id"), exc)
        LOGGER.info(
            "Ranking completed | candidates=%s persisted_opportunities=%s persisted_snapshots=%s",
            len(source_candidates),
            persisted_opportunities,
            persisted_snapshots,
        )
        return ranked

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
        candidates, diagnostics = await self._collect_source_candidates_with_diagnostics()
        self._last_source_diagnostics = diagnostics
        return candidates

    async def _collect_source_candidates_with_diagnostics(self) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        async with LegoRetiringScraper() as lego_scraper, AmazonBestsellerScraper() as amazon_scraper:
            lego_task = lego_scraper.fetch_retiring_sets(limit=50)
            amazon_task = amazon_scraper.fetch_bestsellers(limit=50)
            lego_data, amazon_data = await asyncio.gather(lego_task, amazon_task, return_exceptions=True)

        candidates: list[Dict[str, Any]] = []
        source_raw_counts: Dict[str, int] = {"lego_retiring": 0, "amazon_bestsellers": 0}
        source_failures: list[str] = []

        if isinstance(lego_data, Exception):
            LOGGER.warning("Lego source failed: %s", lego_data)
            source_failures.append(f"lego_retiring: {lego_data}")
        else:
            source_raw_counts["lego_retiring"] = len(lego_data)
            candidates.extend(lego_data)

        if isinstance(amazon_data, Exception):
            LOGGER.warning("Amazon source failed: %s", amazon_data)
            source_failures.append(f"amazon_bestsellers: {amazon_data}")
        else:
            source_raw_counts["amazon_bestsellers"] = len(amazon_data)
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

        dedup_values = list(dedup.values())
        source_dedup_counts: Dict[str, int] = {}
        for row in dedup_values:
            source = str(row.get("source") or "unknown")
            source_dedup_counts[source] = source_dedup_counts.get(source, 0) + 1

        anti_bot_alert = (
            source_raw_counts["lego_retiring"] == 0
            and source_raw_counts["amazon_bestsellers"] == 0
            and not source_failures
        )
        anti_bot_message = (
            "Entrambe le fonti discovery hanno restituito 0 risultati: possibile anti-bot o cambio DOM."
            if anti_bot_alert
            else None
        )

        diagnostics = {
            "source_raw_counts": source_raw_counts,
            "source_dedup_counts": source_dedup_counts,
            "source_failures": source_failures,
            "dedup_candidates": len(dedup_values),
            "anti_bot_alert": anti_bot_alert,
            "anti_bot_message": anti_bot_message,
        }
        LOGGER.info(
            "Source collection completed | raw=%s dedup_counts=%s dedup_total=%s failures=%s",
            source_raw_counts,
            source_dedup_counts,
            len(dedup_values),
            source_failures,
        )
        return dedup_values, diagnostics

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
