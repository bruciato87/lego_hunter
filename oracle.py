from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, Optional
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
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
MODEL_VERSION_RE = re.compile(r"gemini-(\d+(?:\.\d+)?)", re.IGNORECASE)

DISCOVERY_SOURCE_MODES = {"external_first", "playwright_first", "external_only"}
DEFAULT_DISCOVERY_SOURCE_MODE = "external_first"
DEFAULT_GEMINI_MODEL = "models/gemini-2.0-flash"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


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
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        openrouter_api_key: Optional[str] = None,
        min_ai_score: int = 60,
    ) -> None:
        self.repository = repository
        self.min_ai_score = min_ai_score
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = self._normalize_model_name(os.getenv("GEMINI_MODEL") or gemini_model)
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_api_base = (os.getenv("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE).rstrip("/")
        self.openrouter_model_preference = (os.getenv("OPENROUTER_MODEL") or "").strip()
        requested_mode = (os.getenv("DISCOVERY_SOURCE_MODE") or DEFAULT_DISCOVERY_SOURCE_MODE).strip().lower()
        if requested_mode not in DISCOVERY_SOURCE_MODES:
            LOGGER.warning(
                "Invalid DISCOVERY_SOURCE_MODE='%s'. Falling back to '%s'.",
                requested_mode,
                DEFAULT_DISCOVERY_SOURCE_MODE,
            )
            requested_mode = DEFAULT_DISCOVERY_SOURCE_MODE
        self.discovery_source_mode = requested_mode

        self._model = None
        self._gemini_candidates: list[str] = []
        self._gemini_available_candidates: list[str] = []
        self._gemini_probe_report: list[Dict[str, Any]] = []
        self._gemini_candidate_index: Optional[int] = None
        self._openrouter_model_id: Optional[str] = None
        self._openrouter_candidates: list[str] = []
        self._openrouter_available_candidates: list[str] = []
        self._openrouter_probe_report: list[Dict[str, Any]] = []
        self._openrouter_candidate_index: Optional[int] = None
        self._openrouter_inventory_loaded = False
        self.ai_runtime = {
            "engine": "heuristic",
            "model": "heuristic-ai-v2",
            "mode": "fallback",
        }
        self._last_source_diagnostics: Dict[str, Any] = {
            "source_strategy": self.discovery_source_mode,
            "source_raw_counts": {
                "lego_proxy_reader": 0,
                "amazon_proxy_reader": 0,
                "lego_retiring": 0,
                "amazon_bestsellers": 0,
                "lego_http_fallback": 0,
                "amazon_http_fallback": 0,
            },
            "source_dedup_counts": {},
            "source_failures": [],
            "source_signals": {},
            "dedup_candidates": 0,
            "anti_bot_alert": False,
            "anti_bot_message": None,
        }
        if self.gemini_api_key and genai is not None:
            self._initialize_gemini_runtime()
        elif not self.gemini_api_key:
            LOGGER.warning("Gemini API key missing.")
            self.ai_runtime = {
                "engine": "heuristic",
                "model": "heuristic-ai-v2",
                "mode": "fallback_no_gemini_key",
            }
        else:
            LOGGER.warning("google-generativeai package not installed.")
            self.ai_runtime = {
                "engine": "heuristic",
                "model": "heuristic-ai-v2",
                "mode": "fallback_missing_gemini_package",
            }

        if self._model is None:
            self._initialize_openrouter_runtime()

        if self._model is None and self._openrouter_model_id is None:
            if self.ai_runtime.get("mode") not in {
                "fallback_quota_exhausted",
                "fallback_no_working_model",
                "fallback_no_openrouter_key",
                "fallback_openrouter_unavailable",
            }:
                self.ai_runtime = {
                    "engine": "heuristic",
                    "model": "heuristic-ai-v2",
                    "mode": "fallback_no_external_ai",
                }
        LOGGER.info("Discovery source mode configured: %s", self.discovery_source_mode)

    def _initialize_gemini_runtime(self) -> None:
        if not self.gemini_api_key or genai is None:
            return

        try:
            genai.configure(api_key=self.gemini_api_key)
        except Exception as exc:  # noqa: BLE001
            self._disable_gemini("fallback_gemini_config_error", str(exc))
            return

        discovered = self._discover_available_gemini_models()
        candidates = self._sort_gemini_model_candidates(
            discovered,
            preferred_model=self.gemini_model,
        )
        if not candidates:
            self._disable_gemini("fallback_no_models_listed", "Nessun modello Gemini con generateContent disponibile.")
            return

        self._gemini_candidates = candidates
        probe_report = self._probe_all_gemini_candidates(candidates)
        self._gemini_probe_report = probe_report
        available_models = [row["model"] for row in probe_report if row.get("available")]
        self._gemini_available_candidates = available_models
        LOGGER.info(
            "Gemini inventory complete | total=%s available=%s",
            len(candidates),
            len(available_models),
        )

        if available_models:
            best_model = available_models[0]
            best_idx = candidates.index(best_model)
            self._activate_gemini_model(model_name=best_model, index=best_idx, mode="api_dynamic_inventory")
            if self._model is None:
                self._disable_gemini(
                    "fallback_after_gemini_error",
                    "Attivazione modello Gemini non riuscita dopo inventory scan.",
                )
                return
            self.ai_runtime["inventory_total"] = len(candidates)
            self.ai_runtime["inventory_available"] = len(available_models)
            self.ai_runtime["probe_report"] = probe_report[:8]
            return

        if any(row.get("status") == "quota_exhausted_global" for row in probe_report):
            self._disable_gemini(
                "fallback_quota_exhausted",
                "Quota free-tier Gemini esaurita globalmente (limit: 0).",
            )
            self.ai_runtime["inventory_total"] = len(candidates)
            self.ai_runtime["inventory_available"] = 0
            self.ai_runtime["probe_report"] = probe_report[:8]
            return

        self._disable_gemini(
            "fallback_no_working_model",
            "Nessun modello Gemini disponibile con generateContent + quota.",
        )
        self.ai_runtime["inventory_total"] = len(candidates)
        self.ai_runtime["inventory_available"] = 0
        self.ai_runtime["probe_report"] = probe_report[:8]

    def _discover_available_gemini_models(self) -> list[str]:
        if genai is None:
            return []
        try:
            models = list(genai.list_models())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Gemini list_models failed: %s", exc)
            return []

        available: list[str] = []
        for model in models:
            raw_name = getattr(model, "name", None)
            normalized = self._normalize_model_name(raw_name)
            if not normalized.startswith("models/gemini"):
                continue
            lowered_name = normalized.lower()
            if any(
                token in lowered_name
                for token in (
                    "image",
                    "tts",
                    "transcribe",
                    "speech",
                    "audio",
                )
            ):
                continue
            methods = [str(item).lower() for item in (getattr(model, "supported_generation_methods", None) or [])]
            if "generatecontent" not in methods:
                continue
            available.append(normalized)

        return sorted(set(available))

    @classmethod
    def _sort_gemini_model_candidates(
        cls,
        model_names: Iterable[str],
        *,
        preferred_model: Optional[str] = None,
    ) -> list[str]:
        normalized_unique = sorted({cls._normalize_model_name(name) for name in model_names if name})
        if not normalized_unique:
            return []

        ranked = sorted(
            normalized_unique,
            key=cls._score_gemini_model_name,
            reverse=True,
        )
        preferred = cls._normalize_model_name(preferred_model)
        if preferred and preferred in ranked:
            ranked = [preferred] + [name for name in ranked if name != preferred]
        return ranked

    @classmethod
    def _score_gemini_model_name(cls, model_name: str) -> int:
        lowered = cls._normalize_model_name(model_name).lower()
        version = cls._extract_gemini_version(lowered)
        score = int(version * 100)

        if "ultra" in lowered:
            score += 420
        elif "pro" in lowered:
            score += 360
        elif "thinking" in lowered:
            score += 320
        elif "flash" in lowered and "lite" not in lowered:
            score += 260
        elif "lite" in lowered:
            score += 180
        else:
            score += 120

        if "latest" in lowered:
            score += 12
        if any(token in lowered for token in ("preview", "experimental", "-exp")):
            score -= 18
        return score

    @staticmethod
    def _extract_gemini_version(model_name: str) -> float:
        match = MODEL_VERSION_RE.search(model_name or "")
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    @staticmethod
    def _normalize_model_name(model_name: Optional[str]) -> str:
        cleaned = str(model_name or "").strip()
        if not cleaned:
            return ""
        if cleaned.startswith("models/"):
            return cleaned
        return f"models/{cleaned}"

    def _probe_gemini_model(self, model_name: str) -> tuple[bool, str]:
        if genai is None:
            return False, "google-generativeai unavailable"
        try:
            model = genai.GenerativeModel(model_name)
            model.generate_content(
                "Rispondi con una sola parola: ok",
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 16,
                },
            )
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def _probe_all_gemini_candidates(self, candidates: list[str]) -> list[Dict[str, Any]]:
        report: list[Dict[str, Any]] = []
        global_quota_zero = False

        for model_name in candidates:
            if global_quota_zero:
                report.append(
                    {
                        "model": model_name,
                        "available": False,
                        "status": "quota_exhausted_global",
                        "reason": "Inherited global quota lock (limit: 0).",
                    }
                )
                continue

            ok, reason = self._probe_gemini_model(model_name)
            status = "available" if ok else self._classify_gemini_probe_failure(reason)
            report.append(
                {
                    "model": model_name,
                    "available": ok,
                    "status": status,
                    "reason": reason[:220],
                }
            )
            if ok:
                continue
            LOGGER.warning("Gemini model probe failed | model=%s status=%s reason=%s", model_name, status, reason)
            if status == "quota_exhausted_global":
                global_quota_zero = True

        return report

    @classmethod
    def _classify_gemini_probe_failure(cls, reason: str) -> str:
        text = str(reason or "").lower()
        if cls._is_global_quota_exhausted(text):
            return "quota_exhausted_global"
        if "quota" in text or "resource exhausted" in text or "429" in text or "rate limit" in text:
            return "quota_limited"
        if any(token in text for token in ("not found", "not supported", "permission denied")):
            return "unsupported_or_denied"
        if "deadline" in text or "timeout" in text or "temporarily unavailable" in text:
            return "transient_error"
        return "probe_error"

    def _activate_gemini_model(self, *, model_name: str, index: int, mode: str) -> None:
        if genai is None:
            self._disable_gemini("fallback_missing_package", "google-generativeai non disponibile")
            return

        try:
            self._model = genai.GenerativeModel(model_name)
        except Exception as exc:  # noqa: BLE001
            self._disable_gemini("fallback_after_gemini_error", str(exc))
            return
        self.gemini_model = model_name
        self._gemini_candidate_index = index
        self.ai_runtime = {
            "engine": "gemini",
            "model": model_name,
            "mode": mode,
            "candidate_index": index,
            "candidate_count": len(self._gemini_candidates),
        }
        LOGGER.info(
            "Gemini model activated | model=%s index=%s/%s mode=%s",
            model_name,
            index + 1,
            max(1, len(self._gemini_candidates)),
            mode,
        )

    def _advance_gemini_model(self, *, reason: str) -> bool:
        if not self._gemini_candidates:
            return False

        if self._gemini_candidate_index is not None and 0 <= self._gemini_candidate_index < len(self._gemini_candidates):
            current_name = self._gemini_candidates[self._gemini_candidate_index]
        else:
            current_name = None

        if self._gemini_available_candidates:
            fallback_pool = [name for name in self._gemini_available_candidates if name != current_name]
        else:
            fallback_pool = [name for name in self._gemini_candidates if name != current_name]

        for model_name in fallback_pool:
            idx = self._gemini_candidates.index(model_name)
            ok, probe_reason = self._probe_gemini_model(model_name)
            if not ok:
                LOGGER.warning(
                    "Gemini candidate rejected during failover | model=%s reason=%s",
                    model_name,
                    probe_reason,
                )
                continue

            self._activate_gemini_model(
                model_name=model_name,
                index=idx,
                mode="api_dynamic_failover",
            )
            if self._model is None:
                continue
            LOGGER.warning("Gemini failover completed | previous_reason=%s new_model=%s", reason, model_name)
            return True

        return False

    def _disable_gemini(self, mode: str, reason: str) -> None:
        self._model = None
        self._gemini_candidate_index = None
        self.ai_runtime = {
            "engine": "heuristic",
            "model": "heuristic-ai-v2",
            "mode": mode,
            "reason": reason[:220],
        }
        LOGGER.warning("Gemini disabled | mode=%s reason=%s", mode, reason)

    def _initialize_openrouter_runtime(self) -> None:
        if self._openrouter_inventory_loaded and self._openrouter_model_id is not None:
            return

        self._openrouter_inventory_loaded = True
        if requests is None:
            self._disable_openrouter("fallback_openrouter_unavailable", "requests package unavailable")
            return
        if not self.openrouter_api_key:
            self._disable_openrouter("fallback_no_openrouter_key", "OPENROUTER_API_KEY missing")
            return

        try:
            model_payloads = self._fetch_openrouter_model_payloads()
        except Exception as exc:  # noqa: BLE001
            self._disable_openrouter("fallback_openrouter_unavailable", str(exc))
            return
        candidates = self._sort_openrouter_model_candidates(model_payloads, preferred_model=self.openrouter_model_preference)
        self._openrouter_candidates = candidates
        if not candidates:
            self._disable_openrouter("fallback_no_openrouter_models", "Nessun modello OpenRouter free-tier text-capable.")
            return

        probe_report = self._probe_all_openrouter_candidates(candidates)
        self._openrouter_probe_report = probe_report
        available_models = [row["model"] for row in probe_report if row.get("available")]
        self._openrouter_available_candidates = available_models
        LOGGER.info(
            "OpenRouter inventory complete | total=%s available=%s",
            len(candidates),
            len(available_models),
        )

        if available_models:
            best_model = available_models[0]
            best_idx = candidates.index(best_model)
            self._activate_openrouter_model(
                model_id=best_model,
                index=best_idx,
                mode="api_openrouter_inventory",
                probe_report=probe_report,
            )
            return

        if any(row.get("status") == "quota_exhausted_global" for row in probe_report):
            self._disable_openrouter(
                "fallback_openrouter_quota_exhausted",
                "Quota OpenRouter free-tier non disponibile per i modelli candidati.",
                probe_report=probe_report,
            )
            return

        self._disable_openrouter(
            "fallback_openrouter_no_working_model",
            "Nessun modello OpenRouter free-tier ha superato il probe API.",
            probe_report=probe_report,
        )

    def _fetch_openrouter_model_payloads(self) -> list[Dict[str, Any]]:
        if requests is None:
            return []

        url = f"{self.openrouter_api_base}/models"
        headers = self._openrouter_headers(include_json=False)
        response = requests.get(url, headers=headers, timeout=40)
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter model list error {response.status_code}: {response.text[:220]}")

        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        return []

    @classmethod
    def _sort_openrouter_model_candidates(
        cls,
        model_payloads: list[Dict[str, Any]],
        *,
        preferred_model: Optional[str] = None,
    ) -> list[str]:
        ranked_rows: list[tuple[int, str]] = []
        for row in model_payloads:
            model_id = str(row.get("id") or "").strip()
            if not model_id:
                continue
            if not cls._is_openrouter_text_model(row):
                continue
            if not cls._is_openrouter_free_model(row):
                continue
            ranked_rows.append((cls._score_openrouter_model_payload(row), model_id))

        ranked_rows.sort(key=lambda item: item[0], reverse=True)
        ranked = [item[1] for item in ranked_rows]
        preferred = str(preferred_model or "").strip()
        if preferred and preferred in ranked:
            ranked = [preferred] + [name for name in ranked if name != preferred]
        return ranked

    @classmethod
    def _score_openrouter_model_payload(cls, payload: Dict[str, Any]) -> int:
        model_id = str(payload.get("id") or "").lower()
        score = 0

        if "pro" in model_id:
            score += 360
        elif any(token in model_id for token in ("reason", "thinking", "r1")):
            score += 320
        elif "flash" in model_id:
            score += 250
        elif any(token in model_id for token in ("mini", "small", "nano")):
            score += 170
        else:
            score += 130

        if "preview" in model_id or "beta" in model_id or "exp" in model_id:
            score -= 18

        context_length = payload.get("context_length")
        try:
            context_int = int(context_length or 0)
        except (TypeError, ValueError):
            context_int = 0
        score += min(120, context_int // 4096)
        return score

    @staticmethod
    def _is_openrouter_text_model(payload: Dict[str, Any]) -> bool:
        architecture = payload.get("architecture") if isinstance(payload, dict) else {}
        if not isinstance(architecture, dict):
            architecture = {}
        modality = str(architecture.get("modality") or "").lower()
        if modality:
            left, _, right = modality.partition("->")
            return ("text" in left) and ("text" in right)
        model_id = str(payload.get("id") or "").lower()
        return not any(token in model_id for token in ("image", "tts", "speech", "audio", "vision"))

    @staticmethod
    def _is_openrouter_free_model(payload: Dict[str, Any]) -> bool:
        model_id = str(payload.get("id") or "").lower()
        if model_id.endswith(":free") or ":free" in model_id:
            return True
        pricing = payload.get("pricing")
        if not isinstance(pricing, dict):
            return False

        non_empty_values = []
        for value in pricing.values():
            text = str(value or "").strip()
            if not text:
                continue
            non_empty_values.append(text)
        if not non_empty_values:
            return False

        for text in non_empty_values:
            try:
                if float(text) > 0:
                    return False
            except ValueError:
                return False
        return True

    def _probe_all_openrouter_candidates(self, candidates: list[str]) -> list[Dict[str, Any]]:
        report: list[Dict[str, Any]] = []
        global_quota_zero = False
        for model_id in candidates:
            if global_quota_zero:
                report.append(
                    {
                        "model": model_id,
                        "available": False,
                        "status": "quota_exhausted_global",
                        "reason": "Inherited global quota lock.",
                    }
                )
                continue

            ok, reason = self._probe_openrouter_model(model_id)
            status = "available" if ok else self._classify_openrouter_probe_failure(reason)
            report.append(
                {
                    "model": model_id,
                    "available": ok,
                    "status": status,
                    "reason": reason[:220],
                }
            )
            if ok:
                continue
            LOGGER.warning("OpenRouter model probe failed | model=%s status=%s reason=%s", model_id, status, reason)
            if status == "quota_exhausted_global":
                global_quota_zero = True
        return report

    def _probe_openrouter_model(self, model_id: str) -> tuple[bool, str]:
        if requests is None:
            return False, "requests unavailable"
        try:
            self._openrouter_chat_completion(
                model_id=model_id,
                messages=[{"role": "user", "content": "Rispondi con una sola parola: ok"}],
                max_tokens=8,
                temperature=0.0,
            )
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    @classmethod
    def _classify_openrouter_probe_failure(cls, reason: str) -> str:
        text = str(reason or "").lower()
        if cls._is_global_quota_exhausted(text):
            return "quota_exhausted_global"
        if "quota" in text or "402" in text or "429" in text or "rate limit" in text:
            return "quota_limited"
        if any(token in text for token in ("not found", "not supported", "permission", "401", "403")):
            return "unsupported_or_denied"
        if "timeout" in text or "temporarily unavailable" in text or "502" in text or "503" in text:
            return "transient_error"
        return "probe_error"

    def _activate_openrouter_model(
        self,
        *,
        model_id: str,
        index: int,
        mode: str,
        probe_report: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        self._openrouter_model_id = model_id
        self._openrouter_candidate_index = index
        self.ai_runtime = {
            "engine": "openrouter",
            "provider": "openrouter",
            "model": model_id,
            "mode": mode,
            "candidate_index": index,
            "candidate_count": len(self._openrouter_candidates),
            "inventory_total": len(self._openrouter_candidates),
            "inventory_available": len(self._openrouter_available_candidates),
            "probe_report": (probe_report or self._openrouter_probe_report)[:8],
        }
        LOGGER.info(
            "OpenRouter model activated | model=%s index=%s/%s mode=%s",
            model_id,
            index + 1,
            max(1, len(self._openrouter_candidates)),
            mode,
        )

    def _advance_openrouter_model(self, *, reason: str) -> bool:
        if not self._openrouter_candidates:
            return False
        current = self._openrouter_model_id
        fallback_pool = [name for name in (self._openrouter_available_candidates or self._openrouter_candidates) if name != current]
        for model_id in fallback_pool:
            ok, probe_reason = self._probe_openrouter_model(model_id)
            if not ok:
                LOGGER.warning(
                    "OpenRouter candidate rejected during failover | model=%s reason=%s",
                    model_id,
                    probe_reason,
                )
                continue
            idx = self._openrouter_candidates.index(model_id)
            self._activate_openrouter_model(model_id=model_id, index=idx, mode="api_openrouter_failover")
            LOGGER.warning("OpenRouter failover completed | previous_reason=%s new_model=%s", reason, model_id)
            return True
        return False

    def _disable_openrouter(
        self,
        mode: str,
        reason: str,
        *,
        probe_report: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        self._openrouter_model_id = None
        self._openrouter_candidate_index = None
        if self._model is None:
            self.ai_runtime = {
                "engine": "heuristic",
                "model": "heuristic-ai-v2",
                "mode": mode,
                "reason": reason[:220],
                "inventory_total": len(self._openrouter_candidates),
                "inventory_available": len(self._openrouter_available_candidates),
                "probe_report": (probe_report or self._openrouter_probe_report)[:8],
            }
        LOGGER.warning("OpenRouter disabled | mode=%s reason=%s", mode, reason)

    @staticmethod
    def _should_rotate_openrouter_model(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            token in text
            for token in (
                "quota",
                "rate limit",
                "429",
                "402",
                "not found",
                "not supported",
                "permission",
                "401",
                "403",
            )
        )

    def _openrouter_headers(self, *, include_json: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/bruciato87/lego_hunter",
            "X-Title": "Lego_Hunter",
        }
        if include_json:
            headers["Content-Type"] = "application/json"
        return headers

    def _openrouter_chat_completion(
        self,
        *,
        model_id: str,
        messages: list[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests unavailable")
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")

        url = f"{self.openrouter_api_base}/chat/completions"
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(
            url,
            headers=self._openrouter_headers(include_json=True),
            json=payload,
            timeout=45,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:260]}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("OpenRouter invalid response payload")
        return data

    def _openrouter_generate(self, prompt: str) -> str:
        if not self._openrouter_model_id:
            return "{}"

        data = self._openrouter_chat_completion(
            model_id=self._openrouter_model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=420,
            temperature=0.2,
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("OpenRouter response missing message")
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text")
                    if txt:
                        chunks.append(str(txt))
            if chunks:
                return " ".join(chunks).strip()
        raise RuntimeError("OpenRouter response missing text content")

    @staticmethod
    def _should_rotate_gemini_model(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            token in text
            for token in (
                "not found",
                "not supported",
                "permission denied",
                "resource exhausted",
                "quota",
                "rate limit",
                "429",
            )
        )

    @staticmethod
    def _is_global_quota_exhausted(reason: str) -> bool:
        lowered = str(reason or "").lower()
        return (
            ("quota exceeded" in lowered or "resource exhausted" in lowered)
            and "limit: 0" in lowered
        )

    @staticmethod
    def _ai_runtime_public(runtime: Dict[str, Any]) -> Dict[str, Any]:
        public = dict(runtime or {})
        probe = public.get("probe_report")
        if isinstance(probe, list):
            public["probe_report"] = probe[:5]
        return public

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
            "source_strategy": source_diagnostics.get("source_strategy"),
            "source_order": source_diagnostics.get("source_order", []),
            "selected_source": source_diagnostics.get("selected_source"),
            "source_raw_counts": source_diagnostics["source_raw_counts"],
            "source_dedup_counts": source_diagnostics["source_dedup_counts"],
            "source_failures": source_diagnostics["source_failures"],
            "source_signals": source_diagnostics.get("source_signals", {}),
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
            "ai_runtime": self._ai_runtime_public(self.ai_runtime),
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
        if self.discovery_source_mode == "playwright_first":
            source_order = ("playwright", "external_proxy", "http_fallback")
        elif self.discovery_source_mode == "external_only":
            source_order = ("external_proxy",)
        else:
            source_order = ("external_proxy", "playwright", "http_fallback")

        source_raw_counts: Dict[str, int] = {
            "lego_proxy_reader": 0,
            "amazon_proxy_reader": 0,
            "lego_retiring": 0,
            "amazon_bestsellers": 0,
            "lego_http_fallback": 0,
            "amazon_http_fallback": 0,
        }
        source_failures: list[str] = []
        source_signals: Dict[str, Any] = {}
        fallback_notes: list[str] = []
        fallback_source_used = False
        selected_source = None
        dedup_values: list[Dict[str, Any]] = []
        root_cause_hint = "no_candidates"

        for idx, source_name in enumerate(source_order):
            if source_name == "external_proxy":
                candidates, meta = await asyncio.to_thread(self._collect_external_proxy_candidates)
            elif source_name == "playwright":
                candidates, meta = await self._collect_playwright_candidates()
            else:
                candidates, meta = await asyncio.to_thread(self._collect_http_fallback_candidates)

            self._merge_source_meta(
                source_name=source_name,
                source_raw_counts=source_raw_counts,
                source_failures=source_failures,
                source_signals=source_signals,
                meta=meta,
            )
            LOGGER.info(
                "Discovery source '%s' completed | raw=%s errors=%s signals=%s candidates=%s",
                source_name,
                meta.get("source_raw_counts"),
                meta.get("errors"),
                meta.get("signals"),
                len(candidates),
            )

            if not candidates:
                fallback_notes.append(f"{source_name} returned 0 candidates")
                continue

            dedup_values = self._dedup_candidates(candidates)
            if dedup_values:
                selected_source = source_name
                root_cause_hint = f"{source_name}_success"
                if idx > 0:
                    fallback_source_used = True
                    fallback_notes.append(
                        f"{source_name} activated after '{source_order[0]}' returned 0 candidates"
                    )
                break

            fallback_notes.append(f"{source_name} produced candidates but none were usable after dedup")

        if not dedup_values:
            block_flags = [
                key
                for key, value in source_signals.items()
                if value and self._is_blocking_signal_key(key)
            ]
            if block_flags:
                root_cause_hint = "anti_bot_or_robot_challenge_detected"
            elif source_failures:
                root_cause_hint = "source_errors_or_timeouts"
            else:
                root_cause_hint = "dom_shift_or_empty_upstream_payload"

        source_dedup_counts: Dict[str, int] = {}
        for row in dedup_values:
            source = str(row.get("source") or "unknown")
            source_dedup_counts[source] = source_dedup_counts.get(source, 0) + 1

        anti_bot_alert = bool(
            not dedup_values
            and any(
                value and self._is_blocking_signal_key(key)
                for key, value in source_signals.items()
            )
        )
        anti_bot_message = (
            "Discovery bloccata da anti-bot/challenge: fonti cloud primarie senza risultati utili."
            if anti_bot_alert
            else None
        )

        diagnostics = {
            "source_strategy": self.discovery_source_mode,
            "source_order": list(source_order),
            "selected_source": selected_source,
            "source_raw_counts": source_raw_counts,
            "source_dedup_counts": source_dedup_counts,
            "source_failures": source_failures,
            "source_signals": source_signals,
            "dedup_candidates": len(dedup_values),
            "anti_bot_alert": anti_bot_alert,
            "anti_bot_message": anti_bot_message,
            "fallback_source_used": fallback_source_used,
            "fallback_notes": fallback_notes,
            "root_cause_hint": root_cause_hint,
        }
        LOGGER.info(
            "Source collection completed | strategy=%s selected=%s raw=%s dedup_counts=%s dedup_total=%s failures=%s root=%s",
            self.discovery_source_mode,
            selected_source,
            source_raw_counts,
            source_dedup_counts,
            len(dedup_values),
            source_failures,
            root_cause_hint,
        )
        return dedup_values, diagnostics

    async def _collect_playwright_candidates(self) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        source_raw_counts: Dict[str, int] = {"lego_retiring": 0, "amazon_bestsellers": 0}
        errors: list[str] = []
        candidates: list[Dict[str, Any]] = []

        try:
            async with LegoRetiringScraper() as lego_scraper, AmazonBestsellerScraper() as amazon_scraper:
                lego_task = lego_scraper.fetch_retiring_sets(limit=50)
                amazon_task = amazon_scraper.fetch_bestsellers(limit=50)
                lego_data, amazon_data = await asyncio.gather(lego_task, amazon_task, return_exceptions=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"playwright_bootstrap_failed: {exc}")
            return [], {
                "source_raw_counts": source_raw_counts,
                "errors": errors,
                "signals": {},
            }

        if isinstance(lego_data, Exception):
            LOGGER.warning("Lego source failed: %s", lego_data)
            errors.append(f"lego_retiring: {lego_data}")
        else:
            source_raw_counts["lego_retiring"] = len(lego_data)
            candidates.extend(lego_data)

        if isinstance(amazon_data, Exception):
            LOGGER.warning("Amazon source failed: %s", amazon_data)
            errors.append(f"amazon_bestsellers: {amazon_data}")
        else:
            source_raw_counts["amazon_bestsellers"] = len(amazon_data)
            candidates.extend(amazon_data)

        return candidates, {
            "source_raw_counts": source_raw_counts,
            "errors": errors,
            "signals": {},
        }

    def _collect_external_proxy_candidates(self) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
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
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
        }

        try:
            lego_reader_url = "https://r.jina.ai/http://https://www.lego.com/it-it/categories/retiring-soon"
            lego_response = requests.get(lego_reader_url, headers=headers, timeout=40)
            lego_md = lego_response.text
            lego_lower = lego_md.lower()
            signals["lego_proxy_status_ok"] = lego_response.ok
            signals["lego_proxy_blocked"] = any(
                marker in lego_lower
                for marker in ("captcha", "cloudflare", "access denied", "robot check", "are you human")
            )
            signals["lego_proxy_product_link_count"] = len(re.findall(r"/product/", lego_md, re.IGNORECASE))
            lego_rows = self._parse_lego_proxy_markdown(lego_md, limit=60)
            raw_counts["lego_proxy_reader"] = len(lego_rows)
            candidates.extend(lego_rows)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"lego_proxy_failed: {exc}")

        try:
            amazon_reader_url = "https://r.jina.ai/http://https://www.amazon.it/gp/bestsellers/toys/635019031"
            amazon_response = requests.get(amazon_reader_url, headers=headers, timeout=40)
            amazon_md = amazon_response.text
            amazon_lower = amazon_md.lower()
            signals["amazon_proxy_status_ok"] = amazon_response.ok
            signals["amazon_proxy_blocked"] = any(
                marker in amazon_lower
                for marker in ("captcha", "robot check", "access denied", "are you a robot")
            )
            signals["amazon_proxy_cookie_wall"] = "cookies and advertising choices" in amazon_lower
            signals["amazon_proxy_dp_link_count"] = len(
                re.findall(r"/(?:dp|gp/product)/", amazon_md, re.IGNORECASE)
            )
            amazon_rows = self._parse_amazon_proxy_markdown(amazon_md, limit=60)
            raw_counts["amazon_proxy_reader"] = len(amazon_rows)
            candidates.extend(amazon_rows)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"amazon_proxy_failed: {exc}")

        return self._dedup_candidates(candidates), {
            "source_raw_counts": raw_counts,
            "errors": errors,
            "signals": signals,
        }

    @staticmethod
    def _merge_source_meta(
        *,
        source_name: str,
        source_raw_counts: Dict[str, int],
        source_failures: list[str],
        source_signals: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        for key, value in (meta.get("source_raw_counts") or {}).items():
            source_raw_counts[key] = int(value or 0)
        for err in meta.get("errors") or []:
            source_failures.append(f"{source_name}: {err}")
        for key, value in (meta.get("signals") or {}).items():
            source_signals[key] = value

    @classmethod
    def _dedup_candidates(cls, candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        dedup: Dict[str, Dict[str, Any]] = {}
        for row in candidates:
            set_id = str(row.get("set_id") or "").strip()
            if not set_id:
                continue
            current = dedup.get(set_id)
            if current is None:
                dedup[set_id] = row
                continue

            current_rank = cls._source_priority(current.get("source"))
            incoming_rank = cls._source_priority(row.get("source"))
            if incoming_rank > current_rank:
                dedup[set_id] = row
                continue
            if incoming_rank == current_rank and current.get("current_price") is None and row.get("current_price") is not None:
                dedup[set_id] = row

        return list(dedup.values())

    @staticmethod
    def _source_priority(source: Any) -> int:
        order = {
            "lego_proxy_reader": 120,
            "lego_retiring": 110,
            "lego_http_fallback": 90,
            "amazon_proxy_reader": 80,
            "amazon_bestsellers": 70,
            "amazon_http_fallback": 60,
        }
        return order.get(str(source or "unknown"), 10)

    @staticmethod
    def _is_blocking_signal_key(signal_key: str) -> bool:
        lowered = (signal_key or "").lower()
        return any(token in lowered for token in ("robot", "captcha", "blocked", "cookie_wall"))

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

    @classmethod
    def _parse_lego_proxy_markdown(cls, markdown_text: str, *, limit: int) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        headline_pattern = re.compile(
            r"###\s+\[(?P<name>[^\]]+)\]\((?P<url>https?://www\.lego\.com[^\)]*/product/[^\)]*)\)",
            re.IGNORECASE,
        )
        matches = list(headline_pattern.finditer(markdown_text))
        for idx, match in enumerate(matches):
            name = cls._cleanup_html_text(match.group("name"))
            url = match.group("url")
            set_id = cls._extract_set_id(url, name)
            if not set_id:
                continue

            end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(markdown_text), match.end() + 500)
            snippet = markdown_text[match.end() : end]
            price = cls._extract_price_from_text(snippet)
            rows.append(
                {
                    "set_id": set_id,
                    "set_name": name,
                    "theme": cls._guess_theme_from_name(name),
                    "source": "lego_proxy_reader",
                    "current_price": price,
                    "eol_date_prediction": (date.today() + timedelta(days=75)).isoformat(),
                    "listing_url": url,
                    "metadata": {"proxy_reader": True},
                }
            )
            if len(rows) >= limit:
                break
        return rows

    @classmethod
    def _parse_amazon_proxy_markdown(cls, markdown_text: str, *, limit: int) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        matches = list(MARKDOWN_LINK_RE.finditer(markdown_text))
        seen_urls: set[str] = set()

        for idx, match in enumerate(matches):
            name = cls._cleanup_html_text(match.group(1))
            url = match.group(2)
            lowered_url = url.lower()
            if "amazon.it" not in lowered_url:
                continue
            if "/dp/" not in lowered_url and "/gp/product/" not in lowered_url:
                continue
            if "lego" not in name.lower():
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            set_id = cls._extract_set_id(name, url)
            if not set_id:
                continue

            end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(markdown_text), match.end() + 300)
            snippet = markdown_text[match.start() : end]
            rows.append(
                {
                    "set_id": set_id,
                    "set_name": name,
                    "theme": cls._guess_theme_from_name(name),
                    "source": "amazon_proxy_reader",
                    "current_price": cls._extract_price_from_text(snippet),
                    "eol_date_prediction": None,
                    "listing_url": url,
                    "metadata": {"proxy_reader": True},
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
        prompt = self._build_gemini_prompt(candidate)
        if self._model is not None:
            try:
                text = await asyncio.to_thread(self._gemini_generate, prompt)
                payload = self._extract_json(text)
                return self._payload_to_ai_insight(payload, candidate)
            except Exception as exc:  # noqa: BLE001
                if self._should_rotate_gemini_model(exc) and self._advance_gemini_model(reason=str(exc)):
                    try:
                        text = await asyncio.to_thread(self._gemini_generate, prompt)
                        payload = self._extract_json(text)
                        return self._payload_to_ai_insight(payload, candidate)
                    except Exception as exc_after_switch:  # noqa: BLE001
                        self._disable_gemini("fallback_after_gemini_error", str(exc_after_switch))
                        LOGGER.warning(
                            "Gemini scoring failed after failover for %s: %s",
                            candidate.get("set_id"),
                            exc_after_switch,
                        )
                elif self._should_rotate_gemini_model(exc):
                    self._disable_gemini("fallback_after_gemini_error", str(exc))
                else:
                    LOGGER.warning("Gemini scoring failed for %s: %s", candidate.get("set_id"), exc)

        if self._openrouter_model_id is None and not self._openrouter_inventory_loaded:
            self._initialize_openrouter_runtime()

        if self._openrouter_model_id is not None:
            try:
                text = await asyncio.to_thread(self._openrouter_generate, prompt)
                payload = self._extract_json(text)
                return self._payload_to_ai_insight(payload, candidate)
            except Exception as exc:  # noqa: BLE001
                if self._should_rotate_openrouter_model(exc) and self._advance_openrouter_model(reason=str(exc)):
                    try:
                        text = await asyncio.to_thread(self._openrouter_generate, prompt)
                        payload = self._extract_json(text)
                        return self._payload_to_ai_insight(payload, candidate)
                    except Exception as exc_after_switch:  # noqa: BLE001
                        self._disable_openrouter("fallback_after_openrouter_error", str(exc_after_switch))
                        LOGGER.warning(
                            "OpenRouter scoring failed after failover for %s: %s",
                            candidate.get("set_id"),
                            exc_after_switch,
                        )
                elif self._should_rotate_openrouter_model(exc):
                    self._disable_openrouter("fallback_after_openrouter_error", str(exc))
                else:
                    LOGGER.warning("OpenRouter scoring failed for %s: %s", candidate.get("set_id"), exc)

        return self._heuristic_ai_fallback(candidate)

    @staticmethod
    def _payload_to_ai_insight(payload: Dict[str, Any], candidate: Dict[str, Any]) -> AIInsight:
        score = int(payload.get("score", 50))
        score = max(1, min(100, score))
        summary = str(payload.get("summary") or "No summary")[:1200]
        predicted_eol_date = payload.get("predicted_eol_date") or candidate.get("eol_date_prediction")
        return AIInsight(score=score, summary=summary, predicted_eol_date=predicted_eol_date)

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

        if self.ai_runtime.get("engine") != "gemini":
            self.ai_runtime.setdefault("model", "heuristic-ai-v2")

        base = 55
        if source in {"lego_retiring", "lego_proxy_reader", "lego_http_fallback"}:
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
        source = str(candidate.get("source") or "")
        source_bonus = 20 if source in {"lego_retiring", "lego_proxy_reader", "lego_http_fallback"} else 8
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
