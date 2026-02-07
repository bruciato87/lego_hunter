from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urljoin

try:
    import google.generativeai as genai
except Exception:  # noqa: BLE001
    genai = None

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

from models import LegoHunterRepository, MarketTimeSeriesRecord, OpportunityRadarRecord
from scrapers import AmazonBestsellerScraper, LegoRetiringScraper, SecondaryMarketValidator

LOGGER = logging.getLogger(__name__)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


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
            "fallback_source_used": source_diagnostics.get("fallback_source_used"),
            "fallback_notes": source_diagnostics.get("fallback_notes"),
            "anti_bot_alert": source_diagnostics["anti_bot_alert"],
            "anti_bot_message": source_diagnostics["anti_bot_message"],
            "root_cause_hint": source_diagnostics.get("root_cause_hint"),
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
        fallback_source_used = False
        fallback_notes: list[str] = []
        root_cause_hint = "primary_scrapers_ok"
        fallback_meta: Dict[str, Any] = {}

        if not dedup_values:
            fallback_source_used = True
            LOGGER.warning("Primary scrapers produced 0 dedup candidates; trying HTTP fallback")
            fallback_candidates, fallback_meta = await asyncio.to_thread(self._collect_http_fallback_candidates)
            LOGGER.info(
                "HTTP fallback result | counts=%s errors=%s signals=%s candidates=%s",
                fallback_meta.get("source_raw_counts"),
                fallback_meta.get("errors"),
                fallback_meta.get("signals"),
                len(fallback_candidates),
            )
            if fallback_candidates:
                LOGGER.warning(
                    "Primary scrapers returned 0 candidates; activating HTTP fallback source with %s candidates",
                    len(fallback_candidates),
                )
                dedup_values = fallback_candidates
                root_cause_hint = "playwright_extraction_issue_or_dynamic_dom_shift"
                for key, value in (fallback_meta.get("source_raw_counts") or {}).items():
                    source_raw_counts[key] = value
                for err in fallback_meta.get("errors") or []:
                    source_failures.append(f"http_fallback: {err}")
            else:
                fallback_notes.append("HTTP fallback returned 0 candidates")
                signals = fallback_meta.get("signals") or {}
                if signals.get("lego_robot_markers") or signals.get("amazon_robot_markers"):
                    root_cause_hint = "anti_bot_or_robot_challenge_detected"
                else:
                    root_cause_hint = "dom_shift_or_empty_upstream_payload"
                for err in fallback_meta.get("errors") or []:
                    source_failures.append(f"http_fallback: {err}")
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
            "fallback_source_used": fallback_source_used,
            "fallback_notes": fallback_notes,
            "root_cause_hint": root_cause_hint,
        }
        LOGGER.info(
            "Source collection completed | raw=%s dedup_counts=%s dedup_total=%s failures=%s",
            source_raw_counts,
            source_dedup_counts,
            len(dedup_values),
            source_failures,
        )
        return dedup_values, diagnostics

    def _collect_http_fallback_candidates(self) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        if requests is None:
            return [], {
                "source_raw_counts": {},
                "errors": ["requests package unavailable"],
                "signals": {},
            }

        raw_counts: Dict[str, int] = {}
        errors: list[str] = []
        signals: Dict[str, Any] = {}
        candidates: list[Dict[str, Any]] = []

        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        headers = {"User-Agent": user_agent, "Accept-Language": "it-IT,it;q=0.9,en;q=0.8"}

        try:
            lego_url = "https://www.lego.com/it-it/categories/retiring-soon"
            lego_html = requests.get(lego_url, headers=headers, timeout=30).text
            lego_lower = lego_html.lower()
            signals["lego_robot_markers"] = any(
                marker in lego_lower for marker in ("robot", "captcha", "cloudflare", "access denied")
            )
            signals["lego_has_next_data"] = "__next_data__" in lego_lower
            signals["lego_product_link_count"] = len(re.findall(r"/product/", lego_html, re.IGNORECASE))
            lego_rows = self._parse_lego_html_fallback(lego_html, base_url="https://www.lego.com", limit=50)
            raw_counts["lego_http_fallback"] = len(lego_rows)
            candidates.extend(lego_rows)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"lego_fallback_failed: {exc}")

        try:
            amazon_url = "https://www.amazon.it/gp/bestsellers/toys/635019031"
            amazon_html = requests.get(amazon_url, headers=headers, timeout=30).text
            amazon_lower = amazon_html.lower()
            signals["amazon_robot_markers"] = any(
                marker in amazon_lower for marker in ("robot check", "captcha", "sorry", "access denied")
            )
            signals["amazon_dp_link_count"] = len(
                re.findall(r"/(?:dp|gp/product)/", amazon_html, re.IGNORECASE)
            )
            amazon_rows = self._parse_amazon_html_fallback(
                amazon_html,
                base_url="https://www.amazon.it",
                limit=50,
            )
            raw_counts["amazon_http_fallback"] = len(amazon_rows)
            candidates.extend(amazon_rows)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"amazon_fallback_failed: {exc}")

        dedup: dict[str, Dict[str, Any]] = {}
        for row in candidates:
            set_id = str(row.get("set_id") or "").strip()
            if not set_id:
                continue
            if set_id not in dedup:
                dedup[set_id] = row
            elif dedup[set_id].get("source") != "lego_http_fallback" and row.get("source") == "lego_http_fallback":
                dedup[set_id] = row

        return list(dedup.values()), {
            "source_raw_counts": raw_counts,
            "errors": errors,
            "signals": signals,
        }

    @classmethod
    def _parse_lego_html_fallback(cls, html_text: str, *, base_url: str, limit: int) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        pattern = re.compile(r'<a[^>]+href="([^"]*/product/[^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
        for href, inner in pattern.findall(html_text):
            full_url = urljoin(base_url, href)
            name = cls._cleanup_html_text(inner)
            if len(name) < 4:
                continue
            set_id = cls._extract_set_id(full_url, name)
            if not set_id:
                continue
            rows.append(
                {
                    "set_id": set_id,
                    "set_name": name,
                    "theme": cls._guess_theme_from_name(name),
                    "source": "lego_http_fallback",
                    "current_price": cls._extract_price_from_text(inner),
                    "eol_date_prediction": (date.today() + timedelta(days=75)).isoformat(),
                    "listing_url": full_url,
                    "metadata": {"fallback": True},
                }
            )
            if len(rows) >= limit:
                break
        return rows

    @classmethod
    def _parse_amazon_html_fallback(cls, html_text: str, *, base_url: str, limit: int) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        pattern = re.compile(
            r'<a[^>]+href="([^"]*(?:/dp/|/gp/product/)[^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        seen_links: set[str] = set()
        for href, inner in pattern.findall(html_text):
            full_url = urljoin(base_url, href)
            if full_url in seen_links:
                continue
            seen_links.add(full_url)

            name = cls._cleanup_html_text(inner)
            if len(name) < 4:
                alt_match = re.search(r'alt="([^"]+)"', inner, re.IGNORECASE)
                if alt_match:
                    name = cls._cleanup_html_text(alt_match.group(1))
            if "lego" not in name.lower():
                continue

            set_id = cls._extract_set_id(name, full_url)
            if not set_id:
                continue

            rows.append(
                {
                    "set_id": set_id,
                    "set_name": name,
                    "theme": cls._guess_theme_from_name(name),
                    "source": "amazon_http_fallback",
                    "current_price": cls._extract_price_from_text(inner),
                    "eol_date_prediction": None,
                    "listing_url": full_url,
                    "metadata": {"fallback": True},
                }
            )
            if len(rows) >= limit:
                break
        return rows

    @staticmethod
    def _cleanup_html_text(raw: str) -> str:
        text = TAG_RE.sub(" ", raw or "")
        text = html.unescape(text)
        return SPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _extract_set_id(*texts: str) -> Optional[str]:
        for text in texts:
            match = re.search(r"\b(\d{4,6})\b", text or "")
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_price_from_text(raw: str) -> Optional[float]:
        text = html.unescape(raw or "")
        match = re.search(
            r"(?:€|eur)\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
            r"|(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?:€|eur)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        raw_num = (match.group(1) or match.group(2) or "").replace(" ", "")
        if "," in raw_num and "." in raw_num:
            if raw_num.rfind(",") > raw_num.rfind("."):
                raw_num = raw_num.replace(".", "").replace(",", ".")
            else:
                raw_num = raw_num.replace(",", "")
        elif "," in raw_num:
            raw_num = raw_num.replace(",", ".")
        try:
            return float(raw_num)
        except ValueError:
            return None

    @staticmethod
    def _guess_theme_from_name(name: str) -> str:
        lowered = (name or "").lower()
        mapping = {
            "star wars": "Star Wars",
            "technic": "Technic",
            "city": "City",
            "icons": "Icons",
            "harry potter": "Harry Potter",
            "marvel": "Marvel",
            "ninjago": "Ninjago",
            "friends": "Friends",
            "architecture": "Architecture",
        }
        for key, value in mapping.items():
            if key in lowered:
                return value
        return "Unknown"

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
