from __future__ import annotations

import asyncio
import csv
import math
import html
import json
import logging
import os
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urljoin, urlparse

try:
    import google.generativeai as genai
except Exception:  # noqa: BLE001
    genai = None

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

from backtest import OpportunityBacktester
from forecast import ForecastInsight, InvestmentForecaster
from models import LegoHunterRepository, MarketTimeSeriesRecord, OpportunityRadarRecord
from scrapers import AmazonBestsellerScraper, LegoRetiringScraper, SecondaryMarketValidator

LOGGER = logging.getLogger(__name__)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
MODEL_VERSION_RE = re.compile(r"gemini-(\d+(?:\.\d+)?)", re.IGNORECASE)

DISCOVERY_SOURCE_MODES = {"external_first", "playwright_first", "external_only"}
DEFAULT_DISCOVERY_SOURCE_MODE = "external_first"
DEFAULT_GEMINI_MODEL = "models/gemini-2.0-flash"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_GEMINI_FREE_TIER_ONLY = True
DEFAULT_OPENROUTER_FREE_TIER_ONLY = True
DEFAULT_AI_PROBE_MAX_CANDIDATES = 8
DEFAULT_AI_PROBE_BATCH_SIZE = 3
DEFAULT_AI_PROBE_BUDGET_SEC = 90.0
DEFAULT_AI_PROBE_TIMEOUT_SEC = 12.0
DEFAULT_AI_GENERATION_TIMEOUT_SEC = 20.0
DEFAULT_OPENROUTER_MODELS_TIMEOUT_SEC = 15.0
DEFAULT_OPENROUTER_JSON_REPAIR_PROBE_TIMEOUT_SEC = 4.0
DEFAULT_AI_SCORING_CONCURRENCY = 4
DEFAULT_AI_RANK_MAX_CANDIDATES = 12
DEFAULT_AI_CACHE_TTL_SEC = 10800.0
DEFAULT_AI_PERSISTED_CACHE_TTL_SEC = 129600.0
DEFAULT_AI_CACHE_MAX_ITEMS = 4000
DEFAULT_OPENROUTER_MALFORMED_LIMIT = 3
DEFAULT_AI_SCORING_HARD_BUDGET_SEC = 60.0
DEFAULT_AI_SCORING_ITEM_TIMEOUT_SEC = 18.0
DEFAULT_AI_SCORING_TIMEOUT_RETRIES = 1
DEFAULT_AI_SCORING_RETRY_TIMEOUT_SEC = 7.0
DEFAULT_AI_TIMEOUT_RECOVERY_PROBES = 1
DEFAULT_AI_TIMEOUT_RECOVERY_PROBE_TIMEOUT_SEC = 5.0
DEFAULT_AI_FAST_FAIL_ENABLED = True
DEFAULT_AI_BATCH_SCORING_ENABLED = True
DEFAULT_AI_BATCH_MIN_CANDIDATES = 3
DEFAULT_AI_BATCH_MAX_CANDIDATES = 12
DEFAULT_AI_BATCH_TIMEOUT_SEC = 26.0
DEFAULT_AI_SINGLE_CALL_SCORING_ENABLED = False
DEFAULT_AI_SINGLE_CALL_ALLOW_REPAIR_CALLS = False
DEFAULT_AI_SINGLE_CALL_MAX_CANDIDATES = 12
DEFAULT_AI_SINGLE_CALL_MISSING_RESCUE_ENABLED = True
DEFAULT_AI_SINGLE_CALL_MISSING_RESCUE_MAX_CANDIDATES = 3
DEFAULT_AI_SINGLE_CALL_MISSING_RESCUE_TIMEOUT_SEC = 10.0
DEFAULT_AI_SCORE_GUARDRAIL_ENABLED = True
DEFAULT_AI_SCORE_SOFT_CAP = 95
DEFAULT_AI_SCORE_SOFT_CAP_FACTOR = 0.35
DEFAULT_AI_LOW_CONFIDENCE_SCORE_CAP = 90
DEFAULT_AI_NON_JSON_SCORE_CAP = 85
DEFAULT_BOOTSTRAP_THRESHOLDS_ENABLED = False
DEFAULT_BOOTSTRAP_MIN_HISTORY_POINTS = 45
DEFAULT_BOOTSTRAP_MIN_UPSIDE_PROBABILITY = 0.56
DEFAULT_BOOTSTRAP_MIN_CONFIDENCE_SCORE = 55
DEFAULT_HISTORICAL_HIGH_CONF_REQUIRED = True
DEFAULT_HISTORICAL_HIGH_CONF_MIN_SAMPLES = 24
DEFAULT_HISTORICAL_HIGH_CONF_MIN_WIN_RATE_PCT = 56.0
DEFAULT_HISTORICAL_HIGH_CONF_MIN_SUPPORT_CONFIDENCE = 50
DEFAULT_HISTORICAL_HIGH_CONF_MIN_PRIOR_SCORE = 60
DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLDS_ENABLED = True
DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLD_MIN_CASES = 30
DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLD_MIN_THEMES = 4
DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLD_QUANTILE = 0.35
DEFAULT_HISTORICAL_CONTEXTUAL_GATE_ENABLED = True
DEFAULT_HISTORICAL_CONTEXT_STRONG_PATTERN_MIN_SCORE = 75
DEFAULT_HISTORICAL_CONTEXT_MAX_WIN_RATE_RELAX_PCT = 10.0
DEFAULT_HISTORICAL_CONTEXT_MAX_SUPPORT_RELAX = 6
DEFAULT_HISTORICAL_CONTEXT_MAX_PRIOR_RELAX = 10
DEFAULT_HISTORICAL_QUALITY_GUARD_ENABLED = True
DEFAULT_HISTORICAL_QUALITY_SOFT_GATE_ENABLED = True
DEFAULT_HISTORICAL_QUALITY_MAX_MEDIAN_AGE_YEARS = 4
DEFAULT_HISTORICAL_QUALITY_MAX_TOP_THEME_SHARE = 0.26
DEFAULT_HISTORICAL_QUALITY_MAX_GENERAL_TAG_SHARE = 0.70
DEFAULT_HISTORICAL_QUALITY_MIN_THEME_COUNT = 12
DEFAULT_HISTORICAL_DEGRADED_GATE_RELAX_ENABLED = True
DEFAULT_HISTORICAL_DEGRADED_MIN_SAMPLES = 8
DEFAULT_HISTORICAL_DEGRADED_MIN_WIN_RATE_PCT = 10.0
DEFAULT_HISTORICAL_DEGRADED_WIN_RATE_MARGIN_PCT = 3.0
DEFAULT_HISTORICAL_DEGRADED_MIN_SUPPORT_CONFIDENCE = 40
DEFAULT_HISTORICAL_DEGRADED_MIN_PRIOR_SCORE = 35
DEFAULT_AI_MODEL_BAN_SEC = 1800.0
DEFAULT_AI_MODEL_BAN_FAILURES = 2
DEFAULT_AI_MODEL_FAILURE_PENALTY = 22
DEFAULT_AI_MODEL_SUCCESS_REWARD = 3
DEFAULT_AI_STRICT_PROBE_VALIDATION = True
DEFAULT_OPENROUTER_OPPORTUNISTIC_ENABLED = True
DEFAULT_OPENROUTER_OPPORTUNISTIC_ATTEMPTS = 3
DEFAULT_OPENROUTER_OPPORTUNISTIC_TIMEOUT_SEC = 8.0
DEFAULT_AI_DYNAMIC_SHORTLIST_ENABLED = True
DEFAULT_AI_DYNAMIC_SHORTLIST_FLOOR = 4
DEFAULT_AI_DYNAMIC_SHORTLIST_MULTI_MODEL_FLOOR = 5
DEFAULT_AI_DYNAMIC_SHORTLIST_PER_MODEL = 2
DEFAULT_AI_DYNAMIC_SHORTLIST_BONUS = 1
DEFAULT_AI_TOP_PICK_RESCUE_ENABLED = True
DEFAULT_AI_TOP_PICK_RESCUE_COUNT = 3
DEFAULT_AI_TOP_PICK_RESCUE_TIMEOUT_SEC = 9.0
DEFAULT_AI_FINAL_PICK_GUARANTEE_COUNT = 3
DEFAULT_AI_FINAL_PICK_GUARANTEE_ROUNDS = 2
DEFAULT_HISTORICAL_REFERENCE_CASES_PATH = "data/historical_seed/historical_reference_cases.csv"
DEFAULT_HISTORICAL_REFERENCE_ENABLED = True
DEFAULT_HISTORICAL_REFERENCE_MIN_SAMPLES = 24
DEFAULT_HISTORICAL_PRIOR_WEIGHT = 0.10
DEFAULT_HISTORICAL_PRICE_BAND_TOLERANCE = 0.45
DEFAULT_HISTORICAL_ALLOWED_COUNTRIES = "IT"
DEFAULT_HISTORICAL_ALLOWED_REGIONS = "EU"
DEFAULT_HISTORICAL_INCLUDE_UNKNOWN_MARKET = True
DEFAULT_HISTORICAL_RECENCY_HALFLIFE_DAYS = 900
DEFAULT_HISTORICAL_RECENCY_MIN_WEIGHT = 0.20
DEFAULT_HISTORY_WINDOW_DAYS = 180
DEFAULT_BACKTEST_LOOKBACK_DAYS = 365
DEFAULT_BACKTEST_HORIZON_DAYS = 180

CULT_MOVIE_KEYWORDS = (
    "fast & furious",
    "paul walker",
    "lord of the rings",
    "lotr",
    "dune",
    "jurassic",
    "batman",
    "ghostbusters",
    "back to the future",
)
BLOCKBUSTER_FRANCHISE_KEYWORDS = (
    "star wars",
    "marvel",
    "harry potter",
    "ninjago",
    "disney",
    "pokemon",
)
EXCLUSIVE_MINIFIGURE_KEYWORDS = (
    "exclusive minifigure",
    "minifigure esclusiva",
    "solo minifigure",
    "only minifigure",
    "personaggio esclusivo",
    "paul walker",
)
SERIES_COLLECTION_KEYWORDS = (
    "series",
    "collezione",
    "collection",
    "helmet",
    "casco",
    "modular",
    "diorama",
    "botanical collection",
    "speed champions",
)
ADULT_DISPLAY_KEYWORDS = (
    "18+",
    "for adults",
    "adults",
    "ideas",
    "art",
    "van gogh",
    "display model",
    "botanical",
)
FLAGSHIP_COLLECTOR_KEYWORDS = (
    "ucs",
    "ultimate collector",
    "collector",
    "iconic",
    "technic",
    "flagship",
)
MODULAR_COMPLETIST_KEYWORDS = (
    "modular",
    "town square",
    "medieval",
    "castle",
    "city center",
)
VEHICLE_NOSTALGIA_KEYWORDS = (
    "skyline",
    "charger",
    "mustang",
    "camaro",
    "aston martin",
    "batmobile",
    "x-jet",
    "rover",
    "van",
)
SCARCITY_KEYWORDS = (
    "limited edition",
    "gwp",
    "gift with purchase",
    "edizione limitata",
    "retiring soon",
    "last chance",
)
SPACE_STEM_KEYWORDS = (
    "nasa",
    "apollo",
    "space",
    "moon",
    "lunar",
    "rover",
    "astronaut",
)
ART_DISPLAY_THEMES = {"Art", "Ideas", "Icons", "Botanicals", "Architecture"}
LEGO_PRIMARY_SOURCES = {"lego_retiring", "lego_proxy_reader", "lego_http_fallback"}
EUROPE_MARKET_COUNTRY_CODES = {
    "IT", "DE", "FR", "ES", "PT", "NL", "BE", "LU", "AT", "IE", "PL", "CZ", "SK", "HU", "SI", "HR", "RO",
    "BG", "GR", "CY", "MT", "DK", "SE", "FI", "EE", "LV", "LT",
    "NO", "IS", "CH", "GB", "UK",
}
HISTORICAL_THEME_ALIASES = {
    "marvel": ("super heroes", "batman"),
    "super heroes": ("marvel", "batman"),
    "dc": ("super heroes", "batman"),
    "icons": ("advanced models", "architecture", "ideas"),
    "botanicals": ("advanced models", "ideas"),
    "animal crossing": ("miscellaneous", "city", "friends"),
    "ideas": ("advanced models", "architecture"),
    "creator expert": ("advanced models", "creator"),
}


@dataclass
class AIInsight:
    score: int
    summary: str
    predicted_eol_date: Optional[str] = None
    fallback_used: bool = False
    confidence: str = "HIGH_CONFIDENCE"
    risk_note: Optional[str] = None
    model_raw_score: Optional[int] = None


@dataclass
class PatternEvaluation:
    score: int
    confidence_score: int
    summary: str
    signals: list[Dict[str, Any]]
    features: Dict[str, Any]


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
        self.min_composite_score = self._safe_env_int(
            "MIN_COMPOSITE_SCORE",
            default=min_ai_score,
            minimum=1,
            maximum=100,
        )
        self.min_upside_probability = self._safe_env_float(
            "MIN_UPSIDE_PROBABILITY",
            default=0.60,
            minimum=0.05,
            maximum=0.99,
        )
        self.min_confidence_score = self._safe_env_int(
            "MIN_CONFIDENCE_SCORE",
            default=68,
            minimum=1,
            maximum=100,
        )
        self.history_window_days = self._safe_env_int(
            "HISTORY_WINDOW_DAYS",
            default=DEFAULT_HISTORY_WINDOW_DAYS,
            minimum=30,
            maximum=365,
        )
        self.target_roi_pct = self._safe_env_float(
            "TARGET_ROI_PCT",
            default=30.0,
            minimum=5.0,
            maximum=150.0,
        )
        self.forecaster = InvestmentForecaster(target_roi_pct=self.target_roi_pct)
        self.backtest_lookback_days = self._safe_env_int(
            "BACKTEST_LOOKBACK_DAYS",
            default=DEFAULT_BACKTEST_LOOKBACK_DAYS,
            minimum=90,
            maximum=1095,
        )
        self.backtest_horizon_days = self._safe_env_int(
            "BACKTEST_HORIZON_DAYS",
            default=DEFAULT_BACKTEST_HORIZON_DAYS,
            minimum=30,
            maximum=540,
        )
        self.backtest_min_selected = self._safe_env_int(
            "BACKTEST_MIN_SELECTED",
            default=15,
            minimum=5,
            maximum=200,
        )
        self.auto_tune_thresholds = self._safe_env_bool("AUTO_TUNE_THRESHOLDS", default=False)
        self.backtester = OpportunityBacktester(
            target_roi_pct=self.target_roi_pct,
            horizon_days=self.backtest_horizon_days,
            top_k=3,
        )
        self.threshold_profile = {
            "source": "static_env",
            "composite": self.min_composite_score,
            "probability": self.min_upside_probability,
            "confidence": self.min_confidence_score,
        }
        self.backtest_runtime: Dict[str, Any] = {}
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = self._normalize_model_name(os.getenv("GEMINI_MODEL") or gemini_model)
        self.gemini_free_tier_only = self._safe_env_bool(
            "GEMINI_FREE_TIER_ONLY",
            default=DEFAULT_GEMINI_FREE_TIER_ONLY,
        )
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_api_base = (os.getenv("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE).rstrip("/")
        self.openrouter_free_tier_only = self._safe_env_bool(
            "OPENROUTER_FREE_TIER_ONLY",
            default=DEFAULT_OPENROUTER_FREE_TIER_ONLY,
        )
        self.openrouter_model_preference = (os.getenv("OPENROUTER_MODEL") or "").strip()
        self.openrouter_json_repair_model_preference = (os.getenv("OPENROUTER_JSON_REPAIR_MODEL") or "").strip()
        self.ai_probe_max_candidates = self._safe_env_int(
            "AI_PROBE_MAX_CANDIDATES",
            default=DEFAULT_AI_PROBE_MAX_CANDIDATES,
            minimum=1,
            maximum=30,
        )
        self.ai_probe_batch_size = self._safe_env_int(
            "AI_PROBE_BATCH_SIZE",
            default=DEFAULT_AI_PROBE_BATCH_SIZE,
            minimum=1,
            maximum=10,
        )
        self.ai_probe_early_successes = self._safe_env_int(
            "AI_PROBE_EARLY_SUCCESSES",
            default=1,
            minimum=1,
            maximum=5,
        )
        self.ai_probe_budget_sec = self._safe_env_float(
            "AI_PROBE_BUDGET_SEC",
            default=DEFAULT_AI_PROBE_BUDGET_SEC,
            minimum=15.0,
            maximum=240.0,
        )
        self.ai_probe_timeout_sec = self._safe_env_float(
            "AI_PROBE_TIMEOUT_SEC",
            default=DEFAULT_AI_PROBE_TIMEOUT_SEC,
            minimum=4.0,
            maximum=60.0,
        )
        self.ai_generation_timeout_sec = self._safe_env_float(
            "AI_GENERATION_TIMEOUT_SEC",
            default=DEFAULT_AI_GENERATION_TIMEOUT_SEC,
            minimum=8.0,
            maximum=120.0,
        )
        self.openrouter_models_timeout_sec = self._safe_env_float(
            "OPENROUTER_MODELS_TIMEOUT_SEC",
            default=DEFAULT_OPENROUTER_MODELS_TIMEOUT_SEC,
            minimum=5.0,
            maximum=90.0,
        )
        self.openrouter_json_repair_probe_timeout_sec = self._safe_env_float(
            "OPENROUTER_JSON_REPAIR_PROBE_TIMEOUT_SEC",
            default=DEFAULT_OPENROUTER_JSON_REPAIR_PROBE_TIMEOUT_SEC,
            minimum=2.0,
            maximum=12.0,
        )
        self.ai_scoring_concurrency = self._safe_env_int(
            "AI_SCORING_CONCURRENCY",
            default=DEFAULT_AI_SCORING_CONCURRENCY,
            minimum=1,
            maximum=12,
        )
        self.ai_rank_max_candidates = self._safe_env_int(
            "AI_RANK_MAX_CANDIDATES",
            default=DEFAULT_AI_RANK_MAX_CANDIDATES,
            minimum=3,
            maximum=120,
        )
        self.ai_dynamic_shortlist_enabled = self._safe_env_bool(
            "AI_DYNAMIC_SHORTLIST_ENABLED",
            default=DEFAULT_AI_DYNAMIC_SHORTLIST_ENABLED,
        )
        self.ai_dynamic_shortlist_floor = self._safe_env_int(
            "AI_DYNAMIC_SHORTLIST_FLOOR",
            default=DEFAULT_AI_DYNAMIC_SHORTLIST_FLOOR,
            minimum=2,
            maximum=20,
        )
        self.ai_dynamic_shortlist_multi_model_floor = self._safe_env_int(
            "AI_DYNAMIC_SHORTLIST_MULTI_MODEL_FLOOR",
            default=DEFAULT_AI_DYNAMIC_SHORTLIST_MULTI_MODEL_FLOOR,
            minimum=3,
            maximum=20,
        )
        self.ai_dynamic_shortlist_per_model = self._safe_env_int(
            "AI_DYNAMIC_SHORTLIST_PER_MODEL",
            default=DEFAULT_AI_DYNAMIC_SHORTLIST_PER_MODEL,
            minimum=1,
            maximum=8,
        )
        self.ai_dynamic_shortlist_bonus = self._safe_env_int(
            "AI_DYNAMIC_SHORTLIST_BONUS",
            default=DEFAULT_AI_DYNAMIC_SHORTLIST_BONUS,
            minimum=0,
            maximum=8,
        )
        self.ai_cache_ttl_sec = self._safe_env_float(
            "AI_INSIGHT_CACHE_TTL_SEC",
            default=DEFAULT_AI_CACHE_TTL_SEC,
            minimum=0.0,
            maximum=86400.0,
        )
        self.ai_persisted_cache_ttl_sec = self._safe_env_float(
            "AI_PERSISTED_CACHE_TTL_SEC",
            default=DEFAULT_AI_PERSISTED_CACHE_TTL_SEC,
            minimum=0.0,
            maximum=172800.0,
        )
        self.ai_cache_max_items = self._safe_env_int(
            "AI_INSIGHT_CACHE_MAX_ITEMS",
            default=DEFAULT_AI_CACHE_MAX_ITEMS,
            minimum=100,
            maximum=20000,
        )
        self.openrouter_malformed_limit = self._safe_env_int(
            "OPENROUTER_MALFORMED_LIMIT",
            default=DEFAULT_OPENROUTER_MALFORMED_LIMIT,
            minimum=1,
            maximum=20,
        )
        self.ai_scoring_hard_budget_sec = self._safe_env_float(
            "AI_SCORING_HARD_BUDGET_SEC",
            default=DEFAULT_AI_SCORING_HARD_BUDGET_SEC,
            minimum=10.0,
            maximum=300.0,
        )
        self.ai_scoring_item_timeout_sec = self._safe_env_float(
            "AI_SCORING_ITEM_TIMEOUT_SEC",
            default=DEFAULT_AI_SCORING_ITEM_TIMEOUT_SEC,
            minimum=4.0,
            maximum=120.0,
        )
        self.ai_scoring_timeout_retries = self._safe_env_int(
            "AI_SCORING_TIMEOUT_RETRIES",
            default=DEFAULT_AI_SCORING_TIMEOUT_RETRIES,
            minimum=0,
            maximum=2,
        )
        self.ai_scoring_retry_timeout_sec = self._safe_env_float(
            "AI_SCORING_RETRY_TIMEOUT_SEC",
            default=DEFAULT_AI_SCORING_RETRY_TIMEOUT_SEC,
            minimum=2.0,
            maximum=60.0,
        )
        self.ai_timeout_recovery_probes = self._safe_env_int(
            "AI_TIMEOUT_RECOVERY_PROBES",
            default=DEFAULT_AI_TIMEOUT_RECOVERY_PROBES,
            minimum=0,
            maximum=3,
        )
        self.ai_timeout_recovery_probe_timeout_sec = self._safe_env_float(
            "AI_TIMEOUT_RECOVERY_PROBE_TIMEOUT_SEC",
            default=DEFAULT_AI_TIMEOUT_RECOVERY_PROBE_TIMEOUT_SEC,
            minimum=2.0,
            maximum=12.0,
        )
        self.ai_fast_fail_enabled = self._safe_env_bool(
            "AI_FAST_FAIL_ENABLED",
            default=DEFAULT_AI_FAST_FAIL_ENABLED,
        )
        self.ai_batch_scoring_enabled = self._safe_env_bool(
            "AI_BATCH_SCORING_ENABLED",
            default=DEFAULT_AI_BATCH_SCORING_ENABLED,
        )
        self.ai_batch_min_candidates = self._safe_env_int(
            "AI_BATCH_MIN_CANDIDATES",
            default=DEFAULT_AI_BATCH_MIN_CANDIDATES,
            minimum=2,
            maximum=20,
        )
        self.ai_batch_max_candidates = self._safe_env_int(
            "AI_BATCH_MAX_CANDIDATES",
            default=DEFAULT_AI_BATCH_MAX_CANDIDATES,
            minimum=2,
            maximum=40,
        )
        self.ai_batch_timeout_sec = self._safe_env_float(
            "AI_BATCH_TIMEOUT_SEC",
            default=DEFAULT_AI_BATCH_TIMEOUT_SEC,
            minimum=6.0,
            maximum=60.0,
        )
        self.ai_single_call_scoring_enabled = self._safe_env_bool(
            "AI_SINGLE_CALL_SCORING_ENABLED",
            default=DEFAULT_AI_SINGLE_CALL_SCORING_ENABLED,
        )
        self.ai_single_call_allow_repair_calls = self._safe_env_bool(
            "AI_SINGLE_CALL_ALLOW_REPAIR_CALLS",
            default=DEFAULT_AI_SINGLE_CALL_ALLOW_REPAIR_CALLS,
        )
        self.ai_single_call_max_candidates = self._safe_env_int(
            "AI_SINGLE_CALL_MAX_CANDIDATES",
            default=DEFAULT_AI_SINGLE_CALL_MAX_CANDIDATES,
            minimum=3,
            maximum=40,
        )
        self.ai_single_call_missing_rescue_enabled = self._safe_env_bool(
            "AI_SINGLE_CALL_MISSING_RESCUE_ENABLED",
            default=DEFAULT_AI_SINGLE_CALL_MISSING_RESCUE_ENABLED,
        )
        self.ai_single_call_missing_rescue_max_candidates = self._safe_env_int(
            "AI_SINGLE_CALL_MISSING_RESCUE_MAX_CANDIDATES",
            default=DEFAULT_AI_SINGLE_CALL_MISSING_RESCUE_MAX_CANDIDATES,
            minimum=1,
            maximum=8,
        )
        self.ai_single_call_missing_rescue_timeout_sec = self._safe_env_float(
            "AI_SINGLE_CALL_MISSING_RESCUE_TIMEOUT_SEC",
            default=DEFAULT_AI_SINGLE_CALL_MISSING_RESCUE_TIMEOUT_SEC,
            minimum=6.0,
            maximum=25.0,
        )
        self.ai_score_guardrail_enabled = self._safe_env_bool(
            "AI_SCORE_GUARDRAIL_ENABLED",
            default=DEFAULT_AI_SCORE_GUARDRAIL_ENABLED,
        )
        self.ai_score_soft_cap = self._safe_env_int(
            "AI_SCORE_SOFT_CAP",
            default=DEFAULT_AI_SCORE_SOFT_CAP,
            minimum=70,
            maximum=99,
        )
        self.ai_score_soft_cap_factor = self._safe_env_float(
            "AI_SCORE_SOFT_CAP_FACTOR",
            default=DEFAULT_AI_SCORE_SOFT_CAP_FACTOR,
            minimum=0.05,
            maximum=0.95,
        )
        self.ai_low_confidence_score_cap = self._safe_env_int(
            "AI_LOW_CONFIDENCE_SCORE_CAP",
            default=DEFAULT_AI_LOW_CONFIDENCE_SCORE_CAP,
            minimum=50,
            maximum=100,
        )
        self.ai_non_json_score_cap = self._safe_env_int(
            "AI_NON_JSON_SCORE_CAP",
            default=DEFAULT_AI_NON_JSON_SCORE_CAP,
            minimum=40,
            maximum=100,
        )
        self.ai_top_pick_rescue_enabled = self._safe_env_bool(
            "AI_TOP_PICK_RESCUE_ENABLED",
            default=DEFAULT_AI_TOP_PICK_RESCUE_ENABLED,
        )
        self.ai_top_pick_rescue_count = self._safe_env_int(
            "AI_TOP_PICK_RESCUE_COUNT",
            default=DEFAULT_AI_TOP_PICK_RESCUE_COUNT,
            minimum=1,
            maximum=6,
        )
        self.ai_top_pick_rescue_timeout_sec = self._safe_env_float(
            "AI_TOP_PICK_RESCUE_TIMEOUT_SEC",
            default=DEFAULT_AI_TOP_PICK_RESCUE_TIMEOUT_SEC,
            minimum=2.0,
            maximum=20.0,
        )
        self.ai_final_pick_guarantee_count = self._safe_env_int(
            "AI_FINAL_PICK_GUARANTEE_COUNT",
            default=DEFAULT_AI_FINAL_PICK_GUARANTEE_COUNT,
            minimum=1,
            maximum=5,
        )
        self.ai_final_pick_guarantee_rounds = self._safe_env_int(
            "AI_FINAL_PICK_GUARANTEE_ROUNDS",
            default=DEFAULT_AI_FINAL_PICK_GUARANTEE_ROUNDS,
            minimum=1,
            maximum=4,
        )
        self.historical_reference_enabled = self._safe_env_bool(
            "HISTORICAL_REFERENCE_ENABLED",
            default=DEFAULT_HISTORICAL_REFERENCE_ENABLED,
        )
        reference_paths_raw = (
            os.getenv("HISTORICAL_REFERENCE_CASES_PATH")
            or DEFAULT_HISTORICAL_REFERENCE_CASES_PATH
        ).strip()
        self.historical_reference_paths = self._split_env_csv(reference_paths_raw)
        if not self.historical_reference_paths:
            self.historical_reference_paths = [DEFAULT_HISTORICAL_REFERENCE_CASES_PATH]
        self.historical_allowed_countries = self._split_env_csv(
            os.getenv("HISTORICAL_ALLOWED_COUNTRIES") or DEFAULT_HISTORICAL_ALLOWED_COUNTRIES,
            upper=True,
        )
        self.historical_allowed_regions = self._split_env_csv(
            os.getenv("HISTORICAL_ALLOWED_REGIONS") or DEFAULT_HISTORICAL_ALLOWED_REGIONS,
            upper=True,
        )
        self.historical_include_unknown_market = self._safe_env_bool(
            "HISTORICAL_INCLUDE_UNKNOWN_MARKET",
            default=DEFAULT_HISTORICAL_INCLUDE_UNKNOWN_MARKET,
        )
        self.historical_recency_halflife_days = self._safe_env_int(
            "HISTORICAL_RECENCY_HALFLIFE_DAYS",
            default=DEFAULT_HISTORICAL_RECENCY_HALFLIFE_DAYS,
            minimum=30,
            maximum=3650,
        )
        self.historical_recency_min_weight = self._safe_env_float(
            "HISTORICAL_RECENCY_MIN_WEIGHT",
            default=DEFAULT_HISTORICAL_RECENCY_MIN_WEIGHT,
            minimum=0.01,
            maximum=1.0,
        )
        self.historical_reference_min_samples = self._safe_env_int(
            "HISTORICAL_REFERENCE_MIN_SAMPLES",
            default=DEFAULT_HISTORICAL_REFERENCE_MIN_SAMPLES,
            minimum=5,
            maximum=250,
        )
        self.historical_prior_weight = self._safe_env_float(
            "HISTORICAL_PRIOR_WEIGHT",
            default=DEFAULT_HISTORICAL_PRIOR_WEIGHT,
            minimum=0.0,
            maximum=0.35,
        )
        self.historical_price_band_tolerance = self._safe_env_float(
            "HISTORICAL_PRICE_BAND_TOLERANCE",
            default=DEFAULT_HISTORICAL_PRICE_BAND_TOLERANCE,
            minimum=0.10,
            maximum=1.50,
        )
        self.bootstrap_thresholds_enabled = self._safe_env_bool(
            "BOOTSTRAP_THRESHOLDS_ENABLED",
            default=DEFAULT_BOOTSTRAP_THRESHOLDS_ENABLED,
        )
        self.bootstrap_min_history_points = self._safe_env_int(
            "BOOTSTRAP_MIN_HISTORY_POINTS",
            default=DEFAULT_BOOTSTRAP_MIN_HISTORY_POINTS,
            minimum=5,
            maximum=365,
        )
        self.bootstrap_min_upside_probability = self._safe_env_float(
            "BOOTSTRAP_MIN_UPSIDE_PROBABILITY",
            default=DEFAULT_BOOTSTRAP_MIN_UPSIDE_PROBABILITY,
            minimum=0.05,
            maximum=0.99,
        )
        self.bootstrap_min_confidence_score = self._safe_env_int(
            "BOOTSTRAP_MIN_CONFIDENCE_SCORE",
            default=DEFAULT_BOOTSTRAP_MIN_CONFIDENCE_SCORE,
            minimum=1,
            maximum=100,
        )
        self.historical_high_conf_required = self._safe_env_bool(
            "HISTORICAL_HIGH_CONF_REQUIRED",
            default=DEFAULT_HISTORICAL_HIGH_CONF_REQUIRED,
        )
        self.historical_high_conf_min_samples = self._safe_env_int(
            "HISTORICAL_HIGH_CONF_MIN_SAMPLES",
            default=DEFAULT_HISTORICAL_HIGH_CONF_MIN_SAMPLES,
            minimum=5,
            maximum=250,
        )
        self.historical_high_conf_min_win_rate_pct = self._safe_env_float(
            "HISTORICAL_HIGH_CONF_MIN_WIN_RATE_PCT",
            default=DEFAULT_HISTORICAL_HIGH_CONF_MIN_WIN_RATE_PCT,
            minimum=1.0,
            maximum=100.0,
        )
        self.historical_high_conf_min_support_confidence = self._safe_env_int(
            "HISTORICAL_HIGH_CONF_MIN_SUPPORT_CONFIDENCE",
            default=DEFAULT_HISTORICAL_HIGH_CONF_MIN_SUPPORT_CONFIDENCE,
            minimum=1,
            maximum=100,
        )
        self.historical_high_conf_min_prior_score = self._safe_env_int(
            "HISTORICAL_HIGH_CONF_MIN_PRIOR_SCORE",
            default=DEFAULT_HISTORICAL_HIGH_CONF_MIN_PRIOR_SCORE,
            minimum=1,
            maximum=100,
        )
        self.adaptive_historical_thresholds_enabled = self._safe_env_bool(
            "ADAPTIVE_HISTORICAL_THRESHOLDS_ENABLED",
            default=DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLDS_ENABLED,
        )
        self.adaptive_historical_threshold_min_cases = self._safe_env_int(
            "ADAPTIVE_HISTORICAL_THRESHOLD_MIN_CASES",
            default=DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLD_MIN_CASES,
            minimum=10,
            maximum=1000,
        )
        self.adaptive_historical_threshold_min_themes = self._safe_env_int(
            "ADAPTIVE_HISTORICAL_THRESHOLD_MIN_THEMES",
            default=DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLD_MIN_THEMES,
            minimum=2,
            maximum=50,
        )
        self.adaptive_historical_threshold_quantile = self._safe_env_float(
            "ADAPTIVE_HISTORICAL_THRESHOLD_QUANTILE",
            default=DEFAULT_ADAPTIVE_HISTORICAL_THRESHOLD_QUANTILE,
            minimum=0.05,
            maximum=0.95,
        )
        self.historical_contextual_gate_enabled = self._safe_env_bool(
            "HISTORICAL_CONTEXTUAL_GATE_ENABLED",
            default=DEFAULT_HISTORICAL_CONTEXTUAL_GATE_ENABLED,
        )
        self.historical_context_strong_pattern_min_score = self._safe_env_int(
            "HISTORICAL_CONTEXT_STRONG_PATTERN_MIN_SCORE",
            default=DEFAULT_HISTORICAL_CONTEXT_STRONG_PATTERN_MIN_SCORE,
            minimum=55,
            maximum=95,
        )
        self.historical_context_max_win_rate_relax_pct = self._safe_env_float(
            "HISTORICAL_CONTEXT_MAX_WIN_RATE_RELAX_PCT",
            default=DEFAULT_HISTORICAL_CONTEXT_MAX_WIN_RATE_RELAX_PCT,
            minimum=0.0,
            maximum=20.0,
        )
        self.historical_context_max_support_relax = self._safe_env_int(
            "HISTORICAL_CONTEXT_MAX_SUPPORT_RELAX",
            default=DEFAULT_HISTORICAL_CONTEXT_MAX_SUPPORT_RELAX,
            minimum=0,
            maximum=25,
        )
        self.historical_context_max_prior_relax = self._safe_env_int(
            "HISTORICAL_CONTEXT_MAX_PRIOR_RELAX",
            default=DEFAULT_HISTORICAL_CONTEXT_MAX_PRIOR_RELAX,
            minimum=0,
            maximum=30,
        )
        self.historical_quality_guard_enabled = self._safe_env_bool(
            "HISTORICAL_QUALITY_GUARD_ENABLED",
            default=DEFAULT_HISTORICAL_QUALITY_GUARD_ENABLED,
        )
        self.historical_quality_soft_gate_enabled = self._safe_env_bool(
            "HISTORICAL_QUALITY_SOFT_GATE_ENABLED",
            default=DEFAULT_HISTORICAL_QUALITY_SOFT_GATE_ENABLED,
        )
        self.historical_quality_max_median_age_years = self._safe_env_int(
            "HISTORICAL_QUALITY_MAX_MEDIAN_AGE_YEARS",
            default=DEFAULT_HISTORICAL_QUALITY_MAX_MEDIAN_AGE_YEARS,
            minimum=1,
            maximum=20,
        )
        self.historical_quality_max_top_theme_share = self._safe_env_float(
            "HISTORICAL_QUALITY_MAX_TOP_THEME_SHARE",
            default=DEFAULT_HISTORICAL_QUALITY_MAX_TOP_THEME_SHARE,
            minimum=0.10,
            maximum=0.95,
        )
        self.historical_quality_max_general_tag_share = self._safe_env_float(
            "HISTORICAL_QUALITY_MAX_GENERAL_TAG_SHARE",
            default=DEFAULT_HISTORICAL_QUALITY_MAX_GENERAL_TAG_SHARE,
            minimum=0.10,
            maximum=1.0,
        )
        self.historical_quality_min_theme_count = self._safe_env_int(
            "HISTORICAL_QUALITY_MIN_THEME_COUNT",
            default=DEFAULT_HISTORICAL_QUALITY_MIN_THEME_COUNT,
            minimum=3,
            maximum=100,
        )
        self.historical_degraded_gate_relax_enabled = self._safe_env_bool(
            "HISTORICAL_DEGRADED_GATE_RELAX_ENABLED",
            default=DEFAULT_HISTORICAL_DEGRADED_GATE_RELAX_ENABLED,
        )
        self.historical_degraded_min_samples = self._safe_env_int(
            "HISTORICAL_DEGRADED_MIN_SAMPLES",
            default=DEFAULT_HISTORICAL_DEGRADED_MIN_SAMPLES,
            minimum=5,
            maximum=40,
        )
        self.historical_degraded_min_win_rate_pct = self._safe_env_float(
            "HISTORICAL_DEGRADED_MIN_WIN_RATE_PCT",
            default=DEFAULT_HISTORICAL_DEGRADED_MIN_WIN_RATE_PCT,
            minimum=1.0,
            maximum=80.0,
        )
        self.historical_degraded_win_rate_margin_pct = self._safe_env_float(
            "HISTORICAL_DEGRADED_WIN_RATE_MARGIN_PCT",
            default=DEFAULT_HISTORICAL_DEGRADED_WIN_RATE_MARGIN_PCT,
            minimum=0.0,
            maximum=20.0,
        )
        self.historical_degraded_min_support_confidence = self._safe_env_int(
            "HISTORICAL_DEGRADED_MIN_SUPPORT_CONFIDENCE",
            default=DEFAULT_HISTORICAL_DEGRADED_MIN_SUPPORT_CONFIDENCE,
            minimum=1,
            maximum=100,
        )
        self.historical_degraded_min_prior_score = self._safe_env_int(
            "HISTORICAL_DEGRADED_MIN_PRIOR_SCORE",
            default=DEFAULT_HISTORICAL_DEGRADED_MIN_PRIOR_SCORE,
            minimum=1,
            maximum=100,
        )
        self.openrouter_opportunistic_enabled = self._safe_env_bool(
            "OPENROUTER_OPPORTUNISTIC_ENABLED",
            default=DEFAULT_OPENROUTER_OPPORTUNISTIC_ENABLED,
        )
        self.openrouter_opportunistic_attempts = self._safe_env_int(
            "OPENROUTER_OPPORTUNISTIC_ATTEMPTS",
            default=DEFAULT_OPENROUTER_OPPORTUNISTIC_ATTEMPTS,
            minimum=1,
            maximum=6,
        )
        self.openrouter_opportunistic_timeout_sec = self._safe_env_float(
            "OPENROUTER_OPPORTUNISTIC_TIMEOUT_SEC",
            default=DEFAULT_OPENROUTER_OPPORTUNISTIC_TIMEOUT_SEC,
            minimum=3.0,
            maximum=30.0,
        )
        self.strict_ai_probe_validation = self._safe_env_bool(
            "AI_STRICT_PROBE_VALIDATION",
            default=DEFAULT_AI_STRICT_PROBE_VALIDATION,
        )
        self.ai_model_ban_sec = self._safe_env_float(
            "AI_MODEL_BAN_SEC",
            default=DEFAULT_AI_MODEL_BAN_SEC,
            minimum=60.0,
            maximum=86400.0,
        )
        self.ai_model_ban_failures = self._safe_env_int(
            "AI_MODEL_BAN_FAILURES",
            default=DEFAULT_AI_MODEL_BAN_FAILURES,
            minimum=1,
            maximum=20,
        )
        self.ai_model_failure_penalty = self._safe_env_int(
            "AI_MODEL_FAILURE_PENALTY",
            default=DEFAULT_AI_MODEL_FAILURE_PENALTY,
            minimum=1,
            maximum=80,
        )
        self.ai_model_success_reward = self._safe_env_int(
            "AI_MODEL_SUCCESS_REWARD",
            default=DEFAULT_AI_MODEL_SUCCESS_REWARD,
            minimum=1,
            maximum=30,
        )
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
        self._openrouter_recovery_attempted = False
        self._openrouter_repair_probe_fail_until: Dict[str, float] = {}
        self._openrouter_malformed_errors = 0
        self._model_health: Dict[str, Dict[str, Dict[str, Any]]] = {
            "gemini": {},
            "openrouter": {},
        }
        self._ai_insight_cache: Dict[str, tuple[float, AIInsight]] = {}
        self._ai_failover_lock: Optional[asyncio.Lock] = None
        self._last_ranking_diagnostics: Dict[str, Any] = {}
        self._historical_reference_cases: list[Dict[str, Any]] = []
        self._historical_quality_profile: Dict[str, Any] = {}
        self._adaptive_historical_thresholds: Dict[str, Any] = {}
        self._historical_market_filter_stats: Dict[str, Any] = {}
        self.ai_runtime = {
            "engine": "local_ai",
            "provider": "local",
            "model": "local-quant-ai-v1",
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
                "engine": "local_ai",
                "provider": "local",
                "model": "local-quant-ai-v1",
                "mode": "fallback_no_gemini_key",
            }
        else:
            LOGGER.warning("google-generativeai package not installed.")
            self.ai_runtime = {
                "engine": "local_ai",
                "provider": "local",
                "model": "local-quant-ai-v1",
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
                    "engine": "local_ai",
                    "provider": "local",
                    "model": "local-quant-ai-v1",
                    "mode": "fallback_no_external_ai",
                }

        self._historical_reference_cases = self._load_historical_reference_cases()
        self._historical_quality_profile = self._evaluate_historical_reference_quality(
            self._historical_reference_cases
        )
        self._adaptive_historical_thresholds = self._compute_adaptive_historical_thresholds()

        if self.auto_tune_thresholds:
            self._apply_auto_tuned_thresholds()

        LOGGER.info(
            "AI probe tuning | max_candidates=%s batch=%s early_successes=%s budget_sec=%.1f probe_timeout_sec=%.1f gen_timeout_sec=%.1f strict_json=%s",
            self.ai_probe_max_candidates,
            self.ai_probe_batch_size,
            self.ai_probe_early_successes,
            self.ai_probe_budget_sec,
            self.ai_probe_timeout_sec,
            self.ai_generation_timeout_sec,
            self.strict_ai_probe_validation,
        )
        LOGGER.info(
            "AI free-tier policy | gemini_free_only=%s openrouter_free_only=%s",
            self.gemini_free_tier_only,
            self.openrouter_free_tier_only,
        )
        LOGGER.info(
            "Ranking tuning | ai_concurrency=%s ai_rank_max=%s ai_dynamic_shortlist=%s floor=%s floor_multi_model=%s per_model=%s bonus=%s cache_ttl_sec=%.0f persisted_cache_ttl_sec=%.0f cache_max=%s openrouter_malformed_limit=%s ai_hard_budget_sec=%.1f ai_item_timeout_sec=%.1f ai_timeout_retries=%s ai_retry_timeout_sec=%.1f ai_timeout_recovery_probes=%s ai_timeout_recovery_probe_timeout_sec=%.1f ai_fast_fail_enabled=%s ai_batch_enabled=%s ai_batch_min=%s ai_batch_max=%s ai_batch_timeout_sec=%.1f ai_single_call=%s ai_single_call_allow_repair=%s ai_single_call_max_candidates=%s ai_single_call_missing_rescue_enabled=%s ai_single_call_missing_rescue_max=%s ai_single_call_missing_rescue_timeout_sec=%.1f ai_guardrail_enabled=%s ai_soft_cap=%s ai_soft_cap_factor=%.2f ai_low_conf_cap=%s ai_non_json_cap=%s ai_top_pick_rescue_enabled=%s ai_top_pick_rescue_count=%s ai_top_pick_rescue_timeout_sec=%.1f ai_final_pick_guarantee_count=%s ai_final_pick_guarantee_rounds=%s openrouter_json_repair_probe_timeout_sec=%.1f model_ban_sec=%.0f model_ban_failures=%s openrouter_opp_enabled=%s openrouter_opp_attempts=%s openrouter_opp_timeout_sec=%.1f",
            self.ai_scoring_concurrency,
            self.ai_rank_max_candidates,
            self.ai_dynamic_shortlist_enabled,
            self.ai_dynamic_shortlist_floor,
            self.ai_dynamic_shortlist_multi_model_floor,
            self.ai_dynamic_shortlist_per_model,
            self.ai_dynamic_shortlist_bonus,
            self.ai_cache_ttl_sec,
            self.ai_persisted_cache_ttl_sec,
            self.ai_cache_max_items,
            self.openrouter_malformed_limit,
            self.ai_scoring_hard_budget_sec,
            self.ai_scoring_item_timeout_sec,
            self.ai_scoring_timeout_retries,
            self.ai_scoring_retry_timeout_sec,
            self.ai_timeout_recovery_probes,
            self.ai_timeout_recovery_probe_timeout_sec,
            self.ai_fast_fail_enabled,
            self.ai_batch_scoring_enabled,
            self.ai_batch_min_candidates,
            self.ai_batch_max_candidates,
            self.ai_batch_timeout_sec,
            self.ai_single_call_scoring_enabled,
            self.ai_single_call_allow_repair_calls,
            self.ai_single_call_max_candidates,
            self.ai_single_call_missing_rescue_enabled,
            self.ai_single_call_missing_rescue_max_candidates,
            self.ai_single_call_missing_rescue_timeout_sec,
            self.ai_score_guardrail_enabled,
            self.ai_score_soft_cap,
            self.ai_score_soft_cap_factor,
            self.ai_low_confidence_score_cap,
            self.ai_non_json_score_cap,
            self.ai_top_pick_rescue_enabled,
            self.ai_top_pick_rescue_count,
            self.ai_top_pick_rescue_timeout_sec,
            self.ai_final_pick_guarantee_count,
            self.ai_final_pick_guarantee_rounds,
            self.openrouter_json_repair_probe_timeout_sec,
            self.ai_model_ban_sec,
            self.ai_model_ban_failures,
            self.openrouter_opportunistic_enabled,
            self.openrouter_opportunistic_attempts,
            self.openrouter_opportunistic_timeout_sec,
        )
        LOGGER.info(
            "Predictive tuning | min_composite=%s min_prob=%.2f min_confidence=%s history_days=%s target_roi=%.1f bootstrap_enabled=%s bootstrap_min_history=%s bootstrap_min_prob=%.2f bootstrap_min_confidence=%s historical_gate_required=%s historical_min_samples=%s historical_min_win_rate_pct=%.1f historical_min_support_confidence=%s historical_min_prior_score=%s adaptive_hist_thresholds=%s adaptive_quantile=%.2f contextual_hist_gate=%s contextual_pattern_min=%s contextual_win_relax=%.1f contextual_support_relax=%s contextual_prior_relax=%s",
            self.min_composite_score,
            self.min_upside_probability,
            self.min_confidence_score,
            self.history_window_days,
            self.target_roi_pct,
            self.bootstrap_thresholds_enabled,
            self.bootstrap_min_history_points,
            self.bootstrap_min_upside_probability,
            self.bootstrap_min_confidence_score,
            self.historical_high_conf_required,
            self.historical_high_conf_min_samples,
            self.historical_high_conf_min_win_rate_pct,
            self.historical_high_conf_min_support_confidence,
            self.historical_high_conf_min_prior_score,
            self.adaptive_historical_thresholds_enabled,
            self.adaptive_historical_threshold_quantile,
            self.historical_contextual_gate_enabled,
            self.historical_context_strong_pattern_min_score,
            self.historical_context_max_win_rate_relax_pct,
            self.historical_context_max_support_relax,
            self.historical_context_max_prior_relax,
        )
        LOGGER.info(
            "Backtest tuning | enabled=%s lookback_days=%s horizon_days=%s min_selected=%s profile=%s",
            self.auto_tune_thresholds,
            self.backtest_lookback_days,
            self.backtest_horizon_days,
            self.backtest_min_selected,
            self.threshold_profile,
        )
        LOGGER.info(
            "Historical prior tuning | enabled=%s cases=%s paths=%s market_countries=%s market_regions=%s include_unknown=%s recency_halflife_days=%s recency_min_weight=%.2f min_samples=%s weight=%.2f price_tolerance=%.2f high_conf_required=%s high_conf_min_samples=%s high_conf_min_win_rate_pct=%.1f high_conf_min_support_confidence=%s high_conf_min_prior_score=%s adaptive_active=%s adaptive_effective={samples:%s,win_rate:%.1f,support:%s,prior:%s}",
            self.historical_reference_enabled,
            len(self._historical_reference_cases),
            ",".join(self.historical_reference_paths),
            ",".join(self.historical_allowed_countries) or "*",
            ",".join(self.historical_allowed_regions) or "*",
            self.historical_include_unknown_market,
            self.historical_recency_halflife_days,
            self.historical_recency_min_weight,
            self.historical_reference_min_samples,
            self.historical_prior_weight,
            self.historical_price_band_tolerance,
            self.historical_high_conf_required,
            self.historical_high_conf_min_samples,
            self.historical_high_conf_min_win_rate_pct,
            self.historical_high_conf_min_support_confidence,
            self.historical_high_conf_min_prior_score,
            bool(self._adaptive_historical_thresholds.get("active")),
            int(self._adaptive_historical_thresholds.get("min_samples") or self.historical_high_conf_min_samples),
            float(self._adaptive_historical_thresholds.get("min_win_rate_pct") or self.historical_high_conf_min_win_rate_pct),
            int(
                self._adaptive_historical_thresholds.get("min_support_confidence")
                or self.historical_high_conf_min_support_confidence
            ),
            int(self._adaptive_historical_thresholds.get("min_prior_score") or self.historical_high_conf_min_prior_score),
        )
        LOGGER.info(
            "Historical quality | guard_enabled=%s soft_gate=%s tier=%s degraded=%s median_age_years=%s latest_year=%s themes=%s top_theme_share=%.2f general_tag_share=%.2f issues=%s",
            self.historical_quality_guard_enabled,
            self.historical_quality_soft_gate_enabled,
            self._historical_quality_profile.get("tier"),
            self._historical_quality_profile.get("degraded"),
            self._historical_quality_profile.get("median_age_years"),
            self._historical_quality_profile.get("latest_end_year"),
            self._historical_quality_profile.get("theme_count"),
            float(self._historical_quality_profile.get("top_theme_share") or 0.0),
            float(self._historical_quality_profile.get("general_tag_share") or 0.0),
            self._historical_quality_profile.get("issues") or [],
        )
        LOGGER.info(
            "Historical degraded relax | enabled=%s min_samples=%s min_win_rate_pct=%.1f win_margin_pct=%.1f min_support=%s min_prior=%s",
            self.historical_degraded_gate_relax_enabled,
            self.historical_degraded_min_samples,
            self.historical_degraded_min_win_rate_pct,
            self.historical_degraded_win_rate_margin_pct,
            self.historical_degraded_min_support_confidence,
            self.historical_degraded_min_prior_score,
        )
        if self.historical_quality_guard_enabled and self._historical_quality_profile.get("degraded"):
            LOGGER.warning(
                "Historical seed degraded: thresholds will use quality-aware softening. details=%s",
                self._historical_quality_profile.get("issues") or [],
            )
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
            ordered = self._rank_candidate_models("gemini", available_models, allow_forced_retry=False) or available_models
            best_model = ordered[0]
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

        if self.strict_ai_probe_validation:
            self._disable_gemini(
                "fallback_no_working_model",
                "Nessun modello Gemini ha superato il probe reale JSON.",
            )
            self.ai_runtime["inventory_total"] = len(candidates)
            self.ai_runtime["inventory_available"] = 0
            self.ai_runtime["probe_report"] = probe_report[:8]
            return

        optimistic_model = next(
            (
                row.get("model")
                for row in probe_report
                if row.get("status") in {"not_probed_budget_exhausted", "not_probed_early_stop", "not_probed"}
            ),
            None,
        )
        if optimistic_model and optimistic_model in candidates:
            optimistic_idx = candidates.index(optimistic_model)
            self._activate_gemini_model(
                model_name=optimistic_model,
                index=optimistic_idx,
                mode="api_dynamic_optimistic",
            )
            if self._model is not None:
                self.ai_runtime["inventory_total"] = len(candidates)
                self.ai_runtime["inventory_available"] = 0
                self.ai_runtime["probe_report"] = probe_report[:8]
                LOGGER.warning(
                    "Gemini optimistic activation | model=%s reason=no_probed_available",
                    optimistic_model,
                )
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
            if self.gemini_free_tier_only and not self._is_gemini_free_tier_model_name(normalized):
                continue
            available.append(normalized)

        return sorted(set(available))

    @staticmethod
    def _is_gemini_free_tier_model_name(model_name: str) -> bool:
        lowered = str(model_name or "").strip().lower()
        if not lowered.startswith("models/gemini"):
            return False
        if any(token in lowered for token in ("pro", "ultra")):
            return False
        return any(token in lowered for token in ("flash", "lite"))

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

    @staticmethod
    def _safe_env_int(
        name: str,
        *,
        default: int,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> int:
        raw = os.getenv(name)
        if raw is None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                LOGGER.warning("Invalid int env %s=%r. Using default=%s.", name, raw, default)
                value = default

        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    @staticmethod
    def _safe_env_float(
        name: str,
        *,
        default: float,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        raw = os.getenv(name)
        if raw is None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                LOGGER.warning("Invalid float env %s=%r. Using default=%.2f.", name, raw, default)
                value = default

        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    @staticmethod
    def _safe_env_bool(name: str, *, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        lowered = str(raw).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        LOGGER.warning("Invalid bool env %s=%r. Using default=%s.", name, raw, default)
        return default

    @staticmethod
    def _split_env_csv(raw_value: Optional[str], *, upper: bool = False) -> list[str]:
        if raw_value is None:
            return []
        values = []
        for chunk in str(raw_value).split(","):
            value = chunk.strip()
            if not value:
                continue
            values.append(value.upper() if upper else value)
        return values

    @staticmethod
    def _provider_health_key(provider: str) -> str:
        lowered = str(provider or "").strip().lower()
        if "openrouter" in lowered:
            return "openrouter"
        return "gemini"

    def _get_model_health(self, provider: str, model_name: str) -> Dict[str, Any]:
        provider_key = self._provider_health_key(provider)
        model_key = str(model_name or "").strip()
        bucket = self._model_health.setdefault(provider_key, {})
        row = bucket.get(model_key)
        if row is None:
            row = {
                "score": 100,
                "consecutive_failures": 0,
                "total_failures": 0,
                "total_successes": 0,
                "banned_until": 0.0,
                "last_error": None,
                "last_event_at": None,
            }
            bucket[model_key] = row
        return row

    def _ban_remaining_sec(self, provider: str, model_name: str) -> float:
        row = self._get_model_health(provider, model_name)
        banned_until = float(row.get("banned_until") or 0.0)
        return max(0.0, banned_until - time.time())

    def _is_model_temporarily_banned(self, provider: str, model_name: str) -> bool:
        return self._ban_remaining_sec(provider, model_name) > 0.0

    @staticmethod
    def _is_severe_model_failure_reason(reason: str) -> bool:
        text = str(reason or "").lower()
        severe_tokens = (
            "524",
            "timeout",
            "timed out",
            "missing choices",
            "missing message",
            "missing text content",
            "invalid response payload",
            "invalid json payload",
            "invalid ai payload",
            "payload error",
            "502",
            "503",
            "504",
        )
        return any(token in text for token in severe_tokens)

    def _record_model_success(self, provider: str, model_name: str, *, phase: str) -> None:
        row = self._get_model_health(provider, model_name)
        row["score"] = min(100, int(row.get("score") or 100) + self.ai_model_success_reward)
        row["consecutive_failures"] = 0
        row["total_successes"] = int(row.get("total_successes") or 0) + 1
        row["banned_until"] = 0.0
        row["last_event_at"] = datetime.now(timezone.utc).isoformat()
        if phase == "probe":
            LOGGER.info(
                "AI model health success | provider=%s model=%s score=%s",
                self._provider_health_key(provider),
                model_name,
                row["score"],
            )

    def _record_model_failure(self, provider: str, model_name: str, reason: str, *, phase: str) -> bool:
        row = self._get_model_health(provider, model_name)
        severe = self._is_severe_model_failure_reason(reason)
        penalty = self.ai_model_failure_penalty + (10 if severe else 0)
        row["score"] = max(1, int(row.get("score") or 100) - penalty)
        row["consecutive_failures"] = int(row.get("consecutive_failures") or 0) + 1
        row["total_failures"] = int(row.get("total_failures") or 0) + 1
        row["last_error"] = str(reason)[:240]
        row["last_event_at"] = datetime.now(timezone.utc).isoformat()

        should_ban = severe or int(row.get("consecutive_failures") or 0) >= self.ai_model_ban_failures
        if should_ban:
            ban_multiplier = min(3, int(row.get("consecutive_failures") or 1))
            ban_seconds = float(self.ai_model_ban_sec) * float(ban_multiplier)
            row["banned_until"] = max(float(row.get("banned_until") or 0.0), time.time() + ban_seconds)
            LOGGER.warning(
                "AI model temporarily banned | provider=%s model=%s phase=%s ban_sec=%.0f score=%s failures=%s reason=%s",
                self._provider_health_key(provider),
                model_name,
                phase,
                ban_seconds,
                row["score"],
                row["consecutive_failures"],
                str(reason)[:220],
            )
        else:
            LOGGER.warning(
                "AI model failure | provider=%s model=%s phase=%s score=%s failures=%s reason=%s",
                self._provider_health_key(provider),
                model_name,
                phase,
                row["score"],
                row["consecutive_failures"],
                str(reason)[:220],
            )
        return should_ban

    def _rank_candidate_models(
        self,
        provider: str,
        pool: list[str],
        *,
        allow_forced_retry: bool = True,
    ) -> list[str]:
        if not pool:
            return []

        indexed = {model: idx for idx, model in enumerate(pool)}
        eligible = [model for model in pool if not self._is_model_temporarily_banned(provider, model)]
        if eligible:
            return sorted(
                eligible,
                key=lambda model: (
                    -int(self._get_model_health(provider, model).get("score") or 100),
                    indexed[model],
                ),
            )

        if not allow_forced_retry:
            return []
        forced = min(pool, key=lambda model: self._ban_remaining_sec(provider, model))
        LOGGER.warning(
            "All models banned for provider=%s; forcing retry on model=%s (remaining_ban_sec=%.1f)",
            self._provider_health_key(provider),
            forced,
            self._ban_remaining_sec(provider, forced),
        )
        return [forced]

    @staticmethod
    def _build_ai_probe_prompt() -> str:
        return (
            "Analizza un set LEGO TEST e rispondi SOLO con JSON valido nel formato: "
            '{"score": 1-100, "summary": "max 20 parole", "predicted_eol_date": "YYYY-MM-DD o null"}. '
            "Set ID: 99999, Nome: Probe Set, Tema: Star Wars, Prezzo: 99.99, Fonte: test."
        )

    @staticmethod
    def _select_probe_candidates(candidates: list[str], limit: int) -> list[str]:
        if not candidates or limit <= 0:
            return []
        if limit >= len(candidates):
            return list(candidates)

        # Keep most-capable head models and add sparse tail coverage for diversity.
        head_count = max(1, int(math.ceil(limit * 0.7)))
        head = candidates[:head_count]
        remaining = limit - len(head)
        if remaining <= 0:
            return head

        tail = candidates[head_count:]
        if not tail:
            return head

        if remaining >= len(tail):
            return head + tail

        step = len(tail) / float(remaining)
        selected_tail: list[str] = []
        used_indices: set[int] = set()
        for idx in range(remaining):
            pos = int(round(idx * step))
            pos = min(pos, len(tail) - 1)
            while pos in used_indices and pos + 1 < len(tail):
                pos += 1
            used_indices.add(pos)
            selected_tail.append(tail[pos])
        return head + selected_tail

    def _probe_candidates_with_budget(
        self,
        *,
        provider: str,
        candidates: list[str],
        probe_fn,
        classify_fn,
    ) -> list[Dict[str, Any]]:
        if not candidates:
            return []

        provider_key = self._provider_health_key(provider)
        selected = self._select_probe_candidates(candidates, self.ai_probe_max_candidates)
        selected_set = set(selected)
        probe_limit = max(1, self.ai_probe_batch_size)
        report_by_model: Dict[str, Dict[str, Any]] = {}
        available_count = 0
        global_quota_zero = False
        budget_exhausted = False
        started = time.monotonic()

        LOGGER.info(
            "%s probe start | candidates=%s selected=%s budget_sec=%.1f batch=%s",
            provider,
            len(candidates),
            len(selected),
            self.ai_probe_budget_sec,
            probe_limit,
        )

        for offset in range(0, len(selected), probe_limit):
            elapsed = time.monotonic() - started
            if elapsed >= self.ai_probe_budget_sec:
                budget_exhausted = True
                break
            if global_quota_zero or available_count >= self.ai_probe_early_successes:
                break

            batch = selected[offset : offset + probe_limit]
            LOGGER.info(
                "%s probe batch | idx=%s size=%s elapsed=%.2fs",
                provider,
                (offset // probe_limit) + 1,
                len(batch),
                elapsed,
            )
            allowed_batch: list[str] = []
            for model_name in batch:
                if self._is_model_temporarily_banned(provider_key, model_name):
                    report_by_model[model_name] = {
                        "model": model_name,
                        "available": False,
                        "status": "temporarily_banned",
                        "reason": (
                            f"Model in cooldown ({self._ban_remaining_sec(provider_key, model_name):.1f}s remaining)."
                        ),
                    }
                    continue
                allowed_batch.append(model_name)

            if not allowed_batch:
                continue

            executor = ThreadPoolExecutor(max_workers=min(len(allowed_batch), probe_limit))
            try:
                futures = {executor.submit(probe_fn, model_name): model_name for model_name in allowed_batch}
                remaining_budget = max(0.1, self.ai_probe_budget_sec - (time.monotonic() - started))
                batch_timeout = max(
                    1.0,
                    min(
                        remaining_budget,
                        (self.ai_probe_timeout_sec * max(1, len(allowed_batch))) + 1.0,
                    ),
                )
                done, pending = wait(
                    set(futures.keys()),
                    timeout=batch_timeout,
                    return_when=ALL_COMPLETED,
                )

                for future in done:
                    model_name = futures[future]
                    try:
                        ok, reason = future.result()
                    except Exception as exc:  # noqa: BLE001
                        ok, reason = False, str(exc)
                    status = "available" if ok else classify_fn(reason)
                    if ok:
                        self._record_model_success(provider_key, model_name, phase="probe")
                    else:
                        self._record_model_failure(provider_key, model_name, str(reason), phase="probe")
                    report_by_model[model_name] = {
                        "model": model_name,
                        "available": ok,
                        "status": status,
                        "reason": str(reason)[:220],
                    }
                    if ok:
                        available_count += 1
                    else:
                        LOGGER.warning(
                            "%s model probe failed | model=%s status=%s reason=%s",
                            provider,
                            model_name,
                            status,
                            reason,
                        )
                        if status == "quota_exhausted_global":
                            global_quota_zero = True

                if pending:
                    for future in pending:
                        model_name = futures[future]
                        future.cancel()
                        timeout_reason = f"probe timeout after {batch_timeout:.1f}s"
                        status = classify_fn(timeout_reason)
                        self._record_model_failure(provider_key, model_name, timeout_reason, phase="probe")
                        report_by_model[model_name] = {
                            "model": model_name,
                            "available": False,
                            "status": status,
                            "reason": timeout_reason,
                        }
                        LOGGER.warning(
                            "%s model probe timeout | model=%s timeout=%.1fs",
                            provider,
                            model_name,
                            batch_timeout,
                        )
                    if (time.monotonic() - started) >= self.ai_probe_budget_sec:
                        budget_exhausted = True
                        break
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        report: list[Dict[str, Any]] = []
        for model_name in candidates:
            entry = report_by_model.get(model_name)
            if entry:
                report.append(entry)
                continue

            if model_name not in selected_set:
                report.append(
                    {
                        "model": model_name,
                        "available": False,
                        "status": "skipped_low_priority",
                        "reason": "Skipped by budgeted probe strategy.",
                    }
                )
                continue

            if global_quota_zero:
                report.append(
                    {
                        "model": model_name,
                        "available": False,
                        "status": "quota_exhausted_global",
                        "reason": "Inherited global quota lock.",
                    }
                )
                continue

            if budget_exhausted:
                report.append(
                    {
                        "model": model_name,
                        "available": False,
                        "status": "not_probed_budget_exhausted",
                        "reason": "AI probe budget exhausted before testing this model.",
                    }
                )
                continue

            if available_count >= self.ai_probe_early_successes:
                report.append(
                    {
                        "model": model_name,
                        "available": False,
                        "status": "not_probed_early_stop",
                        "reason": "Early-stop after finding available model.",
                    }
                )
                continue

            report.append(
                {
                    "model": model_name,
                    "available": False,
                    "status": "not_probed",
                    "reason": "Model not reached in probe loop.",
                }
            )

        LOGGER.info(
            "%s probe summary | selected=%s probed=%s available=%s elapsed=%.2fs budget_exhausted=%s early_stop=%s quota_zero=%s",
            provider,
            len(selected),
            len(report_by_model),
            available_count,
            time.monotonic() - started,
            budget_exhausted,
            available_count >= self.ai_probe_early_successes,
            global_quota_zero,
        )
        return report

    def _probe_gemini_model(self, model_name: str) -> tuple[bool, str]:
        if genai is None:
            return False, "google-generativeai unavailable"
        try:
            model = genai.GenerativeModel(model_name)
            payload = {
                "temperature": 0.0,
                "max_output_tokens": 96,
                "response_mime_type": "application/json",
            }
            prompt = self._build_ai_probe_prompt()
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=payload,
                    request_options={"timeout": self.ai_probe_timeout_sec},
                )
            except TypeError:
                response = model.generate_content(
                    prompt,
                    generation_config=payload,
                )
            if self.strict_ai_probe_validation:
                text = str(getattr(response, "text", "") or "").strip()
                parsed = self._extract_json(text)
                self._validate_ai_payload(parsed, candidate=None)
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def _probe_all_gemini_candidates(self, candidates: list[str]) -> list[Dict[str, Any]]:
        return self._probe_candidates_with_budget(
            provider="Gemini",
            candidates=candidates,
            probe_fn=self._probe_gemini_model,
            classify_fn=self._classify_gemini_probe_failure,
        )

    @classmethod
    def _classify_gemini_probe_failure(cls, reason: str) -> str:
        text = str(reason or "").lower()
        if cls._is_global_quota_exhausted(text):
            return "quota_exhausted_global"
        if "invalid ai payload" in text or "json" in text:
            return "invalid_output"
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
            pool = [name for name in self._gemini_available_candidates if name != current_name]
        else:
            pool = [name for name in self._gemini_candidates if name != current_name]
        fallback_pool = self._rank_candidate_models("gemini", pool, allow_forced_retry=True)

        for model_name in fallback_pool:
            idx = self._gemini_candidates.index(model_name)
            ok, probe_reason = self._probe_gemini_model(model_name)
            if not ok:
                self._record_model_failure("gemini", model_name, probe_reason, phase="failover_probe")
                LOGGER.warning(
                    "Gemini candidate rejected during failover | model=%s reason=%s",
                    model_name,
                    probe_reason,
                )
                continue

            self._record_model_success("gemini", model_name, phase="failover_probe")
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
            "engine": "local_ai",
            "provider": "local",
            "model": "local-quant-ai-v1",
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
        candidates = self._sort_openrouter_model_candidates(
            model_payloads,
            preferred_model=self.openrouter_model_preference,
            require_suffix_free=self.openrouter_free_tier_only,
        )
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
            ordered = self._rank_candidate_models("openrouter", available_models, allow_forced_retry=False) or available_models
            best_model = ordered[0]
            best_idx = candidates.index(best_model)
            self._activate_openrouter_model(
                model_id=best_model,
                index=best_idx,
                mode="api_openrouter_inventory",
                probe_report=probe_report,
            )
            return

        # Best-effort activation: if strict probe did not confirm availability, keep one
        # candidate active when signals suggest temporary/provider drift conditions.
        optimistic_model = next(
            (
                row.get("model")
                for row in probe_report
                if row.get("status")
                in {
                    "not_probed_budget_exhausted",
                    "not_probed_early_stop",
                    "not_probed",
                    "quota_limited",
                    "transient_error",
                    "invalid_output",
                }
            ),
            None,
        )
        if optimistic_model and optimistic_model in candidates:
            optimistic_idx = candidates.index(optimistic_model)
            self._activate_openrouter_model(
                model_id=optimistic_model,
                index=optimistic_idx,
                mode="api_openrouter_best_effort",
                probe_report=probe_report,
            )
            LOGGER.warning(
                "OpenRouter best-effort activation | model=%s reason=no_confirmed_available",
                optimistic_model,
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
        response = requests.get(url, headers=headers, timeout=self.openrouter_models_timeout_sec)
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
        require_suffix_free: bool = True,
    ) -> list[str]:
        ranked_rows: list[tuple[int, str]] = []
        for row in model_payloads:
            model_id = str(row.get("id") or "").strip()
            if not model_id:
                continue
            if not cls._is_openrouter_text_model(row):
                continue
            if not cls._is_openrouter_free_model(row, require_suffix_free=require_suffix_free):
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
    def _is_openrouter_free_model(payload: Dict[str, Any], *, require_suffix_free: bool = True) -> bool:
        model_id = str(payload.get("id") or "").lower()
        has_suffix_free = model_id.endswith(":free") or ":free" in model_id
        if has_suffix_free:
            return True
        if require_suffix_free:
            return False
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
        return self._probe_candidates_with_budget(
            provider="OpenRouter",
            candidates=candidates,
            probe_fn=self._probe_openrouter_model,
            classify_fn=self._classify_openrouter_probe_failure,
        )

    def _probe_openrouter_model(
        self,
        model_id: str,
        *,
        timeout_sec: Optional[float] = None,
    ) -> tuple[bool, str]:
        if requests is None:
            return False, "requests unavailable"
        try:
            response = self._openrouter_chat_completion(
                model_id=model_id,
                messages=[{"role": "user", "content": self._build_ai_probe_prompt()}],
                max_tokens=96,
                temperature=0.0,
                request_timeout=timeout_sec or self.ai_probe_timeout_sec,
            )
            text = self._extract_openrouter_text(response)
            if self.strict_ai_probe_validation:
                try:
                    parsed = self._extract_json(text)
                    self._validate_ai_payload(parsed, candidate=None)
                    return True, "ok_json"
                except Exception:
                    # Accept non-JSON outputs if they still expose a usable score:
                    # this keeps at least one AI model active under provider drift.
                    if self._extract_unstructured_score(text) is not None:
                        return True, "ok_text_non_json"
                    raise
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    @classmethod
    def _classify_openrouter_probe_failure(cls, reason: str) -> str:
        text = str(reason or "").lower()
        if cls._is_global_quota_exhausted(text):
            return "quota_exhausted_global"
        if "invalid ai payload" in text or "json" in text:
            return "invalid_output"
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
        self._openrouter_malformed_errors = 0
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
        pool = [name for name in (self._openrouter_available_candidates or self._openrouter_candidates) if name != current]
        fallback_pool = self._rank_candidate_models("openrouter", pool, allow_forced_retry=True)
        for model_id in fallback_pool:
            ok, probe_reason = self._probe_openrouter_model(model_id)
            if not ok:
                self._record_model_failure("openrouter", model_id, probe_reason, phase="failover_probe")
                LOGGER.warning(
                    "OpenRouter candidate rejected during failover | model=%s reason=%s",
                    model_id,
                    probe_reason,
                )
                continue
            self._record_model_success("openrouter", model_id, phase="failover_probe")
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
        self._openrouter_recovery_attempted = False
        self._openrouter_malformed_errors = 0
        if self._model is None:
            self.ai_runtime = {
                "engine": "local_ai",
                "provider": "local",
                "model": "local-quant-ai-v1",
                "mode": mode,
                "reason": reason[:220],
                "inventory_total": len(self._openrouter_candidates),
                "inventory_available": len(self._openrouter_available_candidates),
                "probe_report": (probe_report or self._openrouter_probe_report)[:8],
            }
        LOGGER.warning("OpenRouter disabled | mode=%s reason=%s", mode, reason)

    async def _advance_gemini_model_locked(self, *, reason: str) -> bool:
        if self._ai_failover_lock is None:
            self._ai_failover_lock = asyncio.Lock()
        async with self._ai_failover_lock:
            return self._advance_gemini_model(reason=reason)

    async def _advance_openrouter_model_locked(self, *, reason: str) -> bool:
        if self._ai_failover_lock is None:
            self._ai_failover_lock = asyncio.Lock()
        async with self._ai_failover_lock:
            return self._advance_openrouter_model(reason=reason)

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
                "missing choices",
                "missing message",
                "missing text content",
                "invalid response payload",
                "invalid json payload",
                "timeout",
                "timed out",
                "524",
                "temporarily unavailable",
            )
        )

    @staticmethod
    def _is_openrouter_malformed_response_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            token in text
            for token in (
                "missing choices",
                "missing message",
                "missing text content",
                "invalid response payload",
                "invalid json payload",
            )
        )

    @classmethod
    def _is_openrouter_rate_limited_error(cls, exc: Exception) -> bool:
        status = cls._classify_openrouter_probe_failure(str(exc or ""))
        return status in {"quota_limited", "quota_exhausted_global"}

    def _register_openrouter_malformed_failure(self, *, set_id: Any, reason: str) -> bool:
        self._openrouter_malformed_errors += 1
        LOGGER.warning(
            "OpenRouter malformed counter | model=%s set_id=%s count=%s/%s reason=%s",
            self._openrouter_model_id,
            set_id,
            self._openrouter_malformed_errors,
            self.openrouter_malformed_limit,
            str(reason)[:220],
        )
        if self._openrouter_malformed_errors >= self.openrouter_malformed_limit:
            self._disable_openrouter(
                "fallback_openrouter_malformed_payload",
                (
                    "Soglia payload OpenRouter malformati raggiunta "
                    f"({self._openrouter_malformed_errors}/{self.openrouter_malformed_limit})."
                ),
            )
            return True
        return False

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
        request_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests unavailable")
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")
        if self.openrouter_free_tier_only and ":free" not in str(model_id or "").lower():
            raise RuntimeError(f"OpenRouter free-tier policy violation: {model_id}")

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
            timeout=request_timeout or self.ai_generation_timeout_sec,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:260]}")
        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenRouter invalid JSON payload: {response.text[:180]}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("OpenRouter invalid response payload")
        error_payload = data.get("error")
        if isinstance(error_payload, dict):
            err_message = str(error_payload.get("message") or "unknown_error")
            err_code = error_payload.get("code")
            raise RuntimeError(f"OpenRouter payload error {err_code}: {err_message[:220]}")
        return data

    def _openrouter_generate(
        self,
        prompt: str,
        *,
        request_timeout: Optional[float] = None,
        model_id_override: Optional[str] = None,
    ) -> str:
        model_id = str(model_id_override or self._openrouter_model_id or "").strip()
        if not model_id:
            raise RuntimeError("OpenRouter model not active")

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            data = self._openrouter_chat_completion(
                model_id=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=220,
                temperature=0.2,
                request_timeout=request_timeout,
            )
            try:
                return self._extract_openrouter_text(data)
            except RuntimeError as exc:
                if self._is_openrouter_malformed_response_error(exc) and attempt < max_attempts:
                    delay = 0.45 * attempt
                    LOGGER.warning(
                        "OpenRouter malformed response | model=%s attempt=%s/%s retry_in=%.2fs reason=%s",
                        model_id,
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("OpenRouter response missing text content")

    @staticmethod
    def _build_openrouter_json_repair_prompt(raw_text: str, candidate: Dict[str, Any]) -> str:
        clipped = str(raw_text or "").strip()[:2000]
        return (
            "Trasforma il testo seguente in JSON valido, senza alcun testo extra. "
            "Output SOLO JSON con schema esatto: "
            '{"score": 1-100, "summary": "max 3 frasi", "predicted_eol_date": "YYYY-MM-DD o null"}.\n'
            f"Set ID: {candidate.get('set_id')}\n"
            f"Nome: {candidate.get('set_name')}\n"
            f"Tema: {candidate.get('theme')}\n"
            f"Prezzo: {candidate.get('current_price')}\n"
            "Testo da convertire:\n"
            f"{clipped}"
        )

    @staticmethod
    def _build_openrouter_batch_json_repair_prompt(raw_text: str, candidates: list[Dict[str, Any]]) -> str:
        clipped = str(raw_text or "").strip()[:3500]
        lines = [
            "Trasforma il testo seguente in JSON valido, senza alcun testo extra.",
            'Output SOLO JSON con schema: {"results":[{"set_id":"...", "score":1-100, "summary":"max 2 frasi", "predicted_eol_date":"YYYY-MM-DD o null"}]}',
            "Mantieni solo i set_id presenti nella lista.",
            "SET CONSENTITI:",
        ]
        for row in candidates:
            lines.append(f"- {row.get('set_id')} | {row.get('set_name')} | tema={row.get('theme')} | prezzo={row.get('current_price')}")
        lines.append("Testo da convertire:")
        lines.append(clipped)
        return "\n".join(lines)

    def _resolve_openrouter_json_repair_model(self, *, current_model: str) -> str:
        pool: list[str] = []
        preferred = str(self.openrouter_json_repair_model_preference or "").strip()
        if preferred and preferred != current_model and preferred in self._openrouter_candidates:
            pool.append(preferred)

        pool.extend(
            model
            for model in self._openrouter_available_candidates
            if model and model != current_model and model not in pool
        )
        ranked = self._rank_candidate_models(
            "openrouter",
            [model for model in self._openrouter_candidates if model != current_model],
            allow_forced_retry=False,
        )
        pool.extend(model for model in ranked if model not in pool)

        if not pool:
            return current_model

        now = time.time()
        probe_timeout = float(self.openrouter_json_repair_probe_timeout_sec)
        for model_id in pool[:5]:
            ban_until = float(self._openrouter_repair_probe_fail_until.get(model_id) or 0.0)
            if ban_until > now:
                continue

            ok, reason = self._probe_openrouter_model(model_id, timeout_sec=probe_timeout)
            if ok:
                self._record_model_success("openrouter", model_id, phase="json_repair_probe")
                if model_id not in self._openrouter_available_candidates:
                    self._openrouter_available_candidates.append(model_id)
                return model_id

            self._record_model_failure("openrouter", model_id, reason, phase="json_repair_probe")
            self._openrouter_repair_probe_fail_until[model_id] = now + 600.0
            LOGGER.warning(
                "OpenRouter JSON repair model probe failed | model=%s reason=%s",
                model_id,
                str(reason)[:220],
            )
        return current_model

    async def _repair_openrouter_non_json_output(
        self,
        *,
        raw_text: str,
        candidate: Dict[str, Any],
        timeout_sec: float,
    ) -> Optional[AIInsight]:
        model_id = str(self._openrouter_model_id or "").strip()
        if not model_id:
            return None

        repair_prompt = self._build_openrouter_json_repair_prompt(raw_text, candidate)
        repair_timeout = max(3.0, min(8.0, float(timeout_sec)))
        repair_model = await asyncio.to_thread(
            self._resolve_openrouter_json_repair_model,
            current_model=model_id,
        )
        if repair_model != model_id:
            LOGGER.info(
                "OpenRouter JSON repair switching model | from=%s to=%s set_id=%s",
                model_id,
                repair_model,
                candidate.get("set_id"),
            )
        try:
            repaired_text = await asyncio.to_thread(
                self._openrouter_generate,
                repair_prompt,
                request_timeout=repair_timeout,
                model_id_override=repair_model,
            )
            payload = self._extract_json(repaired_text)
            insight = self._payload_to_ai_insight(payload, candidate)
            self._record_model_success("openrouter", repair_model, phase="scoring_json_repair")
            LOGGER.info(
                "OpenRouter non-JSON repaired to JSON | model=%s set_id=%s",
                repair_model,
                candidate.get("set_id"),
            )
            return insight
        except Exception as exc:  # noqa: BLE001
            self._record_model_failure("openrouter", repair_model, str(exc), phase="scoring_json_repair")
            LOGGER.warning(
                "OpenRouter JSON repair failed | model=%s set_id=%s error=%s",
                repair_model,
                candidate.get("set_id"),
                str(exc)[:220],
            )
            return None

    async def _repair_openrouter_non_json_batch_output(
        self,
        *,
        raw_text: str,
        candidates: list[Dict[str, Any]],
        timeout_sec: float,
    ) -> Optional[Dict[str, AIInsight]]:
        model_id = str(self._openrouter_model_id or "").strip()
        if not model_id or not candidates:
            return None

        repair_prompt = self._build_openrouter_batch_json_repair_prompt(raw_text, candidates)
        repair_timeout = max(4.0, min(10.0, float(timeout_sec)))
        repair_model = await asyncio.to_thread(
            self._resolve_openrouter_json_repair_model,
            current_model=model_id,
        )
        if repair_model != model_id:
            LOGGER.info(
                "OpenRouter batch JSON repair switching model | from=%s to=%s candidates=%s",
                model_id,
                repair_model,
                len(candidates),
            )
        try:
            repaired_text = await asyncio.to_thread(
                self._openrouter_generate,
                repair_prompt,
                request_timeout=repair_timeout,
                model_id_override=repair_model,
            )
            payload = self._extract_json(repaired_text)
            insights = self._batch_payload_to_ai_insights(payload, candidates)
            if not insights:
                raise RuntimeError("batch_json_repair_no_valid_rows")
            self._record_model_success("openrouter", repair_model, phase="batch_json_repair")
            LOGGER.info(
                "OpenRouter batch non-JSON repaired to JSON | model=%s candidates=%s scored=%s",
                repair_model,
                len(candidates),
                len(insights),
            )
            return insights
        except Exception as exc:  # noqa: BLE001
            self._record_model_failure("openrouter", repair_model, str(exc), phase="batch_json_repair")
            LOGGER.warning(
                "OpenRouter batch JSON repair failed | model=%s candidates=%s error=%s",
                repair_model,
                len(candidates),
                str(exc)[:220],
            )
            return None

    async def _recover_openrouter_after_timeout(self, *, set_id: str) -> bool:
        if self._openrouter_model_id is None:
            return False

        # First try regular failover path (available candidates + health ranking).
        rotated = await self._advance_openrouter_model_locked(reason=f"scoring_timeout:{set_id}:failover")
        if rotated:
            return True

        max_probes = max(0, int(self.ai_timeout_recovery_probes))
        if max_probes <= 0:
            return False

        if self._ai_failover_lock is None:
            self._ai_failover_lock = asyncio.Lock()
        async with self._ai_failover_lock:
            current_model = str(self._openrouter_model_id or "")
            pool = [name for name in self._openrouter_candidates if name != current_model]
            ranked_pool = self._rank_candidate_models("openrouter", pool, allow_forced_retry=True)
            if not ranked_pool:
                return False

            quick_timeout = float(self.ai_timeout_recovery_probe_timeout_sec)
            for model_id in ranked_pool[:max_probes]:
                ok, reason = self._probe_openrouter_model(model_id, timeout_sec=quick_timeout)
                if ok:
                    idx = self._openrouter_candidates.index(model_id)
                    self._record_model_success("openrouter", model_id, phase="timeout_recovery_probe")
                    if model_id not in self._openrouter_available_candidates:
                        self._openrouter_available_candidates.append(model_id)
                    self._activate_openrouter_model(
                        model_id=model_id,
                        index=idx,
                        mode="api_openrouter_timeout_recovery",
                    )
                    LOGGER.warning(
                        "OpenRouter timeout recovery activated | set_id=%s model=%s",
                        set_id,
                        model_id,
                    )
                    return True

                self._record_model_failure("openrouter", model_id, reason, phase="timeout_recovery_probe")
                LOGGER.warning(
                    "OpenRouter timeout recovery probe failed | set_id=%s model=%s reason=%s",
                    set_id,
                    model_id,
                    str(reason)[:220],
                )
        return False

    @staticmethod
    def _extract_openrouter_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    if isinstance(content, dict):
                        for key in ("text", "content", "value"):
                            value = content.get(key)
                            if isinstance(value, str) and value.strip():
                                return value.strip()
                    if isinstance(content, list):
                        chunks: list[str] = []
                        for part in content:
                            if isinstance(part, dict):
                                txt = part.get("text")
                                if txt:
                                    chunks.append(str(txt))
                                elif isinstance(part.get("content"), str):
                                    chunks.append(str(part.get("content")))
                                elif isinstance(part.get("value"), str):
                                    chunks.append(str(part.get("value")))
                            elif isinstance(part, str) and part.strip():
                                chunks.append(part.strip())
                        if chunks:
                            return " ".join(chunks).strip()

                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list):
                        arg_chunks: list[str] = []
                        for call in tool_calls:
                            if not isinstance(call, dict):
                                continue
                            function = call.get("function")
                            if isinstance(function, dict):
                                arguments = function.get("arguments")
                                if isinstance(arguments, str) and arguments.strip():
                                    arg_chunks.append(arguments.strip())
                            arguments = call.get("arguments")
                            if isinstance(arguments, str) and arguments.strip():
                                arg_chunks.append(arguments.strip())
                        if arg_chunks:
                            return " ".join(arg_chunks).strip()
                    raise RuntimeError("OpenRouter response missing text content")

                text_value = first.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return text_value.strip()
                raise RuntimeError("OpenRouter response missing message")

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = data.get("output")
        if isinstance(output, list):
            chunks: list[str] = []
            for part in output:
                if isinstance(part, dict):
                    content = part.get("content")
                    if isinstance(content, list):
                        for chunk in content:
                            if isinstance(chunk, dict):
                                txt = chunk.get("text")
                                if txt:
                                    chunks.append(str(txt))
                elif isinstance(part, str) and part.strip():
                    chunks.append(part.strip())
            if chunks:
                return " ".join(chunks).strip()

        raise RuntimeError("OpenRouter response missing choices")

    @staticmethod
    def _extract_unstructured_score(raw_text: str) -> Optional[int]:
        text = str(raw_text or "").strip()
        if not text:
            return None

        patterns = [
            r"(?i)(?:investment[_\s-]?score|score|punteggio|rating|valutazione)\s*[:=]?\s*([1-9]\d?|100)\b",
            r"(?i)\b([1-9]\d?|100)\s*/\s*100\b",
            r"(?i)\b([1-9]\d?|100)\s*(?:su|out of)\s*100\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                return max(1, min(100, value))

        generic = re.search(r"(?<![-\d])([1-9]\d?|100)(?![-\d])", text)
        if generic:
            value = int(generic.group(1))
            return max(1, min(100, value))
        return None

    @classmethod
    def _ai_insight_from_unstructured_text(
        cls,
        text: str,
        candidate: Dict[str, Any],
    ) -> Optional[AIInsight]:
        score = cls._extract_unstructured_score(text)
        if score is None:
            return None

        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return None

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        summary = " ".join(part for part in sentences[:3] if part).strip()
        if not summary:
            summary = cleaned[:320]
        summary = summary[:1200]

        predicted = cls._extract_first_date(cleaned) or candidate.get("eol_date_prediction")
        return AIInsight(
            score=score,
            summary=summary,
            predicted_eol_date=predicted,
            fallback_used=False,
            confidence="LOW_CONFIDENCE",
            risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
        )

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

    def _model_health_public(self, *, per_provider: int = 4) -> Dict[str, list[Dict[str, Any]]]:
        snapshot: Dict[str, list[Dict[str, Any]]] = {}
        now_ts = time.time()
        for provider, rows in (self._model_health or {}).items():
            normalized_provider = self._provider_health_key(provider)
            scored: list[tuple[int, float, str, Dict[str, Any]]] = []
            for model_name, row in rows.items():
                model_row = dict(row or {})
                score = int(model_row.get("score") or 0)
                remaining = max(0.0, float(model_row.get("banned_until") or 0.0) - now_ts)
                scored.append((score, remaining, model_name, model_row))

            scored.sort(key=lambda item: (item[1] > 0.0, -item[0], item[2]))
            preview: list[Dict[str, Any]] = []
            for score, remaining, model_name, model_row in scored[: max(1, per_provider)]:
                preview.append(
                    {
                        "model": model_name,
                        "score": score,
                        "consecutive_failures": int(model_row.get("consecutive_failures") or 0),
                        "total_failures": int(model_row.get("total_failures") or 0),
                        "total_successes": int(model_row.get("total_successes") or 0),
                        "banned_remaining_sec": round(remaining, 1),
                    }
                )
            snapshot[normalized_provider] = preview
        return snapshot

    @staticmethod
    def _backtest_runtime_public(runtime: Dict[str, Any]) -> Dict[str, Any]:
        public = dict(runtime or {})
        if "error" in public:
            public["error"] = str(public.get("error"))[:220]
        return public

    def _apply_auto_tuned_thresholds(self) -> None:
        try:
            report = self.backtester.run(
                repository=self.repository,
                lookback_days=self.backtest_lookback_days,
                max_opportunities=2500,
            )
        except Exception as exc:  # noqa: BLE001
            self.backtest_runtime = {
                "status": "error",
                "error": str(exc),
            }
            LOGGER.warning("Backtest auto-tuning failed while loading data: %s", exc)
            return

        if report.sample_size < max(20, self.backtest_min_selected):
            self.backtest_runtime = {
                "status": "insufficient_data",
                "sample_size": report.sample_size,
                "required": max(20, self.backtest_min_selected),
                "lookback_days": report.lookback_days,
                "horizon_days": report.horizon_days,
            }
            return

        suggestion = self.backtester.tune_thresholds(
            observations=report.observations,
            current_composite=self.min_composite_score,
            current_probability=self.min_upside_probability,
            current_confidence=self.min_confidence_score,
            min_selected=self.backtest_min_selected,
        )

        metrics = suggestion.metrics
        self.backtest_runtime = {
            "status": "ok",
            "sample_size": metrics.sample_size,
            "selected_count": metrics.selected_count,
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "precision_at_k": round(metrics.precision_at_k, 4),
            "coverage": round(metrics.coverage, 4),
            "brier_score": round(metrics.brier_score, 4),
            "avg_realized_roi_selected": round(metrics.avg_realized_roi_selected, 2),
            "objective_score": round(suggestion.objective_score, 4),
            "target_roi_pct": report.target_roi_pct,
            "lookback_days": report.lookback_days,
            "horizon_days": report.horizon_days,
        }

        if suggestion.changed:
            self.min_composite_score = suggestion.composite_score
            self.min_upside_probability = suggestion.probability_upside
            self.min_confidence_score = suggestion.confidence_score
            self.threshold_profile = {
                "source": "auto_tuned_backtest",
                "composite": self.min_composite_score,
                "probability": self.min_upside_probability,
                "confidence": self.min_confidence_score,
            }
            LOGGER.info("Backtest auto-tuning applied | profile=%s metrics=%s", self.threshold_profile, self.backtest_runtime)
        else:
            self.threshold_profile = {
                "source": "auto_tuned_backtest_unchanged",
                "composite": self.min_composite_score,
                "probability": self.min_upside_probability,
                "confidence": self.min_confidence_score,
            }
            LOGGER.info("Backtest auto-tuning kept current profile | profile=%s metrics=%s", self.threshold_profile, self.backtest_runtime)

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
        self._openrouter_recovery_attempted = False
        LOGGER.info(
            "Discovery start | persist=%s top_limit=%s fallback_limit=%s threshold=%s",
            persist,
            top_limit,
            fallback_limit,
            self.min_composite_score,
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

        ranked.sort(
            key=lambda row: (
                int(row.get("composite_score") or row.get("ai_investment_score") or 0),
                int(row.get("forecast_score") or 0),
                int(row.get("market_demand_score") or 0),
            ),
            reverse=True,
        )
        above_threshold = [
            row
            for row in ranked
            if int(row.get("composite_score") or row.get("ai_investment_score") or 0) >= self.min_composite_score
        ]
        above_threshold_with_strength = [
            (row, self._high_confidence_signal_strength(row))
            for row in above_threshold
        ]
        above_threshold_high_conf = [
            (row, strength)
            for row, strength in above_threshold_with_strength
            if str(strength).startswith("HIGH_CONFIDENCE")
        ]
        above_threshold_high_conf_strict = [
            (row, strength)
            for row, strength in above_threshold_with_strength
            if str(strength) in {"HIGH_CONFIDENCE", "HIGH_CONFIDENCE_STRICT"}
        ]
        above_threshold_high_conf_bootstrap = [
            (row, strength)
            for row, strength in above_threshold_with_strength
            if str(strength) == "HIGH_CONFIDENCE_BOOTSTRAP"
        ]
        above_threshold_low_conf = [
            row
            for row, strength in above_threshold_with_strength
            if str(strength) == "LOW_CONFIDENCE"
        ]

        selected: list[Dict[str, Any]]
        fallback_used = False

        if above_threshold_high_conf:
            selected = [
                {
                    **row,
                    "signal_strength": str(strength),
                }
                for row, strength in above_threshold_high_conf[:top_limit]
            ]
        elif above_threshold_low_conf:
            fallback_used = True
            selected = [
                {
                    **row,
                    "signal_strength": "LOW_CONFIDENCE",
                    "risk_note": self._build_low_confidence_note(row),
                }
                for row in above_threshold_low_conf[:fallback_limit]
            ]
        else:
            fallback_used = bool(ranked)
            selected = [
                {
                    **row,
                    "signal_strength": "LOW_CONFIDENCE",
                    "risk_note": f"Nessun set sopra soglia composita {self.min_composite_score}.",
                }
                for row in ranked[:fallback_limit]
            ]

        bootstrap_rows_count = sum(
            1
            for row in ranked
            if self._effective_high_confidence_thresholds(row)[2]
        )
        historical_gate_blocked_count = sum(
            1
            for row in above_threshold
            if not self._historical_high_confidence_status(row)[0]
        )
        (
            effective_hist_min_samples,
            effective_hist_min_win_rate_pct,
            effective_hist_min_support_confidence,
            effective_hist_min_prior_score,
            adaptive_hist_active,
        ) = self._effective_historical_high_confidence_thresholds()
        ai_shortlisted_rows = [row for row in ranked if bool(row.get("ai_shortlisted"))]
        ai_shortlist_count = len(ai_shortlisted_rows)
        ai_fallback_total_count = sum(1 for row in ranked if bool(row.get("ai_fallback_used")))

        def _is_non_json_row(row: Dict[str, Any]) -> bool:
            return self._is_non_json_ai_note(
                str(row.get("risk_note") or row.get("metadata", {}).get("ai_risk_note") or "")
            )

        ai_non_json_total_count = sum(1 for row in ranked if _is_non_json_row(row))
        ai_shortlist_fallback_count = sum(1 for row in ai_shortlisted_rows if bool(row.get("ai_fallback_used")))
        ai_shortlist_non_json_count = sum(1 for row in ai_shortlisted_rows if _is_non_json_row(row))
        ai_shortlist_strict_pass_count = sum(
            1
            for row in ai_shortlisted_rows
            if (not bool(row.get("ai_fallback_used")) and not _is_non_json_row(row))
        )

        diagnostics = {
            "threshold": self.min_composite_score,
            "ai_threshold": self.min_ai_score,
            "min_probability_high_confidence": self.min_upside_probability,
            "min_confidence_score_high_confidence": self.min_confidence_score,
            "bootstrap_thresholds_enabled": self.bootstrap_thresholds_enabled,
            "bootstrap_min_history_points": self.bootstrap_min_history_points,
            "bootstrap_min_probability_high_confidence": self.bootstrap_min_upside_probability,
            "bootstrap_min_confidence_score_high_confidence": self.bootstrap_min_confidence_score,
            "bootstrap_rows_count": bootstrap_rows_count,
            "historical_high_conf_required": self.historical_high_conf_required,
            "historical_high_conf_min_samples": self.historical_high_conf_min_samples,
            "historical_high_conf_min_win_rate_pct": self.historical_high_conf_min_win_rate_pct,
            "historical_high_conf_min_support_confidence": self.historical_high_conf_min_support_confidence,
            "historical_high_conf_min_prior_score": self.historical_high_conf_min_prior_score,
            "historical_high_conf_effective_min_samples": effective_hist_min_samples,
            "historical_high_conf_effective_min_win_rate_pct": effective_hist_min_win_rate_pct,
            "historical_high_conf_effective_min_support_confidence": effective_hist_min_support_confidence,
            "historical_high_conf_effective_min_prior_score": effective_hist_min_prior_score,
            "adaptive_historical_thresholds_enabled": self.adaptive_historical_thresholds_enabled,
            "adaptive_historical_thresholds_active": adaptive_hist_active,
            "adaptive_historical_thresholds": dict(self._adaptive_historical_thresholds or {}),
            "historical_gate_blocked_count": historical_gate_blocked_count,
            "historical_quality": dict(self._historical_quality_profile or {}),
            "historical_quality_guard_enabled": self.historical_quality_guard_enabled,
            "historical_quality_soft_gate_enabled": self.historical_quality_soft_gate_enabled,
            "historical_allowed_countries": list(self.historical_allowed_countries),
            "historical_allowed_regions": list(self.historical_allowed_regions),
            "historical_include_unknown_market": self.historical_include_unknown_market,
            "historical_market_filter": dict(self._historical_market_filter_stats or {}),
            "threshold_profile": dict(self.threshold_profile),
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
            "above_threshold_high_confidence_count": len(above_threshold_high_conf),
            "above_threshold_high_confidence_strict_count": len(above_threshold_high_conf_strict),
            "above_threshold_high_confidence_bootstrap_count": len(above_threshold_high_conf_bootstrap),
            "above_threshold_low_confidence_count": len(above_threshold_low_conf),
            "fallback_scored_count": ai_fallback_total_count,
            "ai_shortlist_effective_count": ai_shortlist_count,
            "ai_shortlist_fallback_count": ai_shortlist_fallback_count,
            "ai_shortlist_non_json_count": ai_shortlist_non_json_count,
            "ai_shortlist_strict_pass_count": ai_shortlist_strict_pass_count,
            "below_threshold_count": len(ranked) - len(above_threshold),
            "max_ai_score": max(
                (
                    int(
                        row.get("ai_raw_score")
                        or row.get("metadata", {}).get("ai_raw_score")
                        or row.get("ai_investment_score")
                        or 0
                    )
                    for row in ranked
                ),
                default=0,
            ),
            "max_ai_model_raw_score": max(
                (
                    int(
                        row.get("ai_model_raw_score")
                        or row.get("metadata", {}).get("ai_model_raw_score")
                        or row.get("ai_raw_score")
                        or row.get("metadata", {}).get("ai_raw_score")
                        or row.get("ai_investment_score")
                        or 0
                    )
                    for row in ranked
                ),
                default=0,
            ),
            "ai_guardrail_applied_count": sum(
                1
                for row in ranked
                if bool(
                    row.get("ai_score_guardrail_applied")
                    or row.get("metadata", {}).get("ai_score_guardrail_applied")
                )
            ),
            "max_composite_score": max((int(row.get("composite_score") or 0) for row in ranked), default=0),
            "max_probability_upside_12m": max(
                (float(row.get("forecast_probability_upside_12m") or 0.0) for row in ranked),
                default=0.0,
            ),
            "fallback_used": fallback_used,
            "fallback_source_used": source_diagnostics.get("fallback_source_used"),
            "fallback_notes": source_diagnostics.get("fallback_notes"),
            "anti_bot_alert": source_diagnostics["anti_bot_alert"],
            "anti_bot_message": source_diagnostics["anti_bot_message"],
            "root_cause_hint": source_diagnostics.get("root_cause_hint"),
            "ai_runtime": self._ai_runtime_public(self.ai_runtime),
            "model_health": self._model_health_public(),
            "backtest_runtime": self._backtest_runtime_public(self.backtest_runtime),
            "ranking": dict(self._last_ranking_diagnostics or {}),
            "ai_non_json_count": ai_non_json_total_count,
        }

        ranked_count = max(1, len(ranked))
        shortlist_count_for_rate = max(1, ai_shortlist_count)
        diagnostics["fallback_rate_total"] = round(float(diagnostics["fallback_scored_count"]) / ranked_count, 4)
        diagnostics["non_json_rate_total"] = round(float(diagnostics["ai_non_json_count"]) / ranked_count, 4)
        diagnostics["fallback_rate_shortlist"] = round(
            float(diagnostics["ai_shortlist_fallback_count"]) / shortlist_count_for_rate,
            4,
        )
        diagnostics["non_json_rate_shortlist"] = round(
            float(diagnostics["ai_shortlist_non_json_count"]) / shortlist_count_for_rate,
            4,
        )
        diagnostics["strict_pass_rate_shortlist"] = round(
            float(diagnostics["ai_shortlist_strict_pass_count"]) / shortlist_count_for_rate,
            4,
        )
        diagnostics["high_conf_strict_rate"] = round(
            float(diagnostics["above_threshold_high_confidence_strict_count"]) / max(1, int(diagnostics["above_threshold_count"])),
            4,
        )
        diagnostics["fallback_rate"] = diagnostics["fallback_rate_shortlist"]
        diagnostics["non_json_rate"] = diagnostics["non_json_rate_shortlist"]
        diagnostics["strict_pass_rate"] = diagnostics["strict_pass_rate_shortlist"]

        if ranked:
            top_debug = [
                {
                    "set_id": row.get("set_id"),
                    "source": row.get("source"),
                    "composite": row.get("composite_score"),
                    "ai": row.get("ai_raw_score"),
                    "quant": row.get("forecast_score"),
                    "demand": row.get("market_demand_score"),
                    "hist": row.get("historical_prior_score"),
                    "prob12m": row.get("forecast_probability_upside_12m"),
                }
                for row in ranked[:3]
            ]
        else:
            top_debug = []

        LOGGER.info(
            "Discovery summary | ranked=%s above_threshold=%s high_conf=%s fallback_used=%s max_ai=%s max_ai_raw=%s ai_guardrails=%s max_composite=%s max_prob=%.1f profile=%s ai=%s ranking=%s top=%s",
            diagnostics["ranked_candidates"],
            diagnostics["above_threshold_count"],
            diagnostics["above_threshold_high_confidence_count"],
            diagnostics["fallback_used"],
            diagnostics["max_ai_score"],
            diagnostics["max_ai_model_raw_score"],
            diagnostics["ai_guardrail_applied_count"],
            diagnostics["max_composite_score"],
            diagnostics["max_probability_upside_12m"],
            diagnostics["threshold_profile"],
            diagnostics["ai_runtime"],
            diagnostics["ranking"],
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
        rerank_attempt: int = 0,
    ) -> list[Dict[str, Any]]:
        if not source_candidates:
            LOGGER.info("Ranking skipped | no source candidates available")
            self._last_ranking_diagnostics = {
                "input_candidates": 0,
                "prepared_candidates": 0,
                "ai_shortlist_count": 0,
                "ai_prefilter_skipped_count": 0,
                "ai_scored_count": 0,
                "ai_batch_scored_count": 0,
                "ai_cache_hits": 0,
                "ai_cache_misses": 0,
                "ai_persisted_cache_hits": 0,
                "ai_errors": 0,
                "ai_budget_exhausted": 0,
                "ai_timeout_count": 0,
                "ai_top_pick_rescue_attempts": 0,
                "ai_top_pick_rescue_successes": 0,
                "ai_top_pick_rescue_failures": 0,
                "ai_top_pick_rescue_timeouts": 0,
                "ai_top_pick_rescue_cache_hits": 0,
                "ai_final_pick_guarantee_rounds": 0,
                "ai_final_pick_guarantee_rescue_sets": 0,
                "ai_final_pick_guarantee_pending_after_rounds": 0,
                "historical_prior_applied_count": 0,
                "quant_prep_sec": 0.0,
                "ai_scoring_sec": 0.0,
                "persistence_sec": 0.0,
                "total_sec": 0.0,
            }
            return []

        started = time.monotonic()
        prep_started = time.monotonic()
        prepared = self._prepare_quantitative_context(source_candidates)
        prep_duration = time.monotonic() - prep_started
        shortlist, skipped = self._select_ai_shortlist(prepared)
        skipped_set_ids = {row["set_id"]: row for row in skipped}

        ai_started = time.monotonic()
        ai_results, ai_stats = await self._score_ai_shortlist(shortlist)

        ranked, opportunities = self._build_ranked_payloads(
            prepared=prepared,
            ai_results=ai_results,
            skipped_set_ids=skipped_set_ids,
            shortlist_count=len(shortlist),
        )

        rescue_stats = {
            "ai_top_pick_rescue_attempts": 0,
            "ai_top_pick_rescue_successes": 0,
            "ai_top_pick_rescue_failures": 0,
            "ai_top_pick_rescue_timeouts": 0,
            "ai_top_pick_rescue_cache_hits": 0,
            "ai_final_pick_guarantee_rounds": 0,
            "ai_final_pick_guarantee_rescue_sets": 0,
            "ai_final_pick_guarantee_pending_after_rounds": 0,
        }
        if self.ai_top_pick_rescue_enabled and ranked:
            initial_rescue_stats = await self._rescue_top_pick_ai_scores(
                prepared=prepared,
                ranked=ranked,
                ai_results=ai_results,
            )
            for key in rescue_stats:
                rescue_stats[key] += int(initial_rescue_stats.get(key, 0))

            if int(initial_rescue_stats.get("ai_top_pick_rescue_attempts") or 0) > 0:
                ranked, opportunities = self._build_ranked_payloads(
                    prepared=prepared,
                    ai_results=ai_results,
                    skipped_set_ids=skipped_set_ids,
                    shortlist_count=len(shortlist),
                )

            # Always guarantee at least a top-3 external AI attempt before accepting fallback.
            guarantee_count = max(3, int(self.ai_final_pick_guarantee_count))
            guarantee_rounds = max(1, int(self.ai_final_pick_guarantee_rounds))
            for _ in range(guarantee_rounds):
                top_rows = sorted(
                    ranked,
                    key=lambda row: (
                        int(row.get("composite_score") or row.get("ai_investment_score") or 0),
                        int(row.get("forecast_score") or 0),
                        int(row.get("market_demand_score") or 0),
                    ),
                    reverse=True,
                )[:guarantee_count]
                pending_set_ids = []
                for row in top_rows:
                    set_id = str(row.get("set_id") or "").strip()
                    if not set_id:
                        continue
                    current_ai = ai_results.get(set_id)
                    if current_ai is None or bool(current_ai.fallback_used):
                        pending_set_ids.append(set_id)

                if not pending_set_ids:
                    rescue_stats["ai_final_pick_guarantee_pending_after_rounds"] = 0
                    break

                rescue_stats["ai_final_pick_guarantee_rounds"] += 1
                rescue_stats["ai_final_pick_guarantee_rescue_sets"] += len(pending_set_ids)
                guarantee_rescue_stats = await self._rescue_top_pick_ai_scores(
                    prepared=prepared,
                    ranked=ranked,
                    ai_results=ai_results,
                    force_set_ids=pending_set_ids,
                    top_k_override=0,
                )
                for key in (
                    "ai_top_pick_rescue_attempts",
                    "ai_top_pick_rescue_successes",
                    "ai_top_pick_rescue_failures",
                    "ai_top_pick_rescue_timeouts",
                    "ai_top_pick_rescue_cache_hits",
                ):
                    rescue_stats[key] += int(guarantee_rescue_stats.get(key, 0))

                if int(guarantee_rescue_stats.get("ai_top_pick_rescue_attempts") or 0) == 0:
                    rescue_stats["ai_final_pick_guarantee_pending_after_rounds"] = len(pending_set_ids)
                    break

                ranked, opportunities = self._build_ranked_payloads(
                    prepared=prepared,
                    ai_results=ai_results,
                    skipped_set_ids=skipped_set_ids,
                    shortlist_count=len(shortlist),
                )
            else:
                top_rows = sorted(
                    ranked,
                    key=lambda row: (
                        int(row.get("composite_score") or row.get("ai_investment_score") or 0),
                        int(row.get("forecast_score") or 0),
                        int(row.get("market_demand_score") or 0),
                    ),
                    reverse=True,
                )[:guarantee_count]
                rescue_stats["ai_final_pick_guarantee_pending_after_rounds"] = sum(
                    1
                    for row in top_rows
                    if (
                        str(row.get("set_id") or "").strip()
                        and (
                            ai_results.get(str(row.get("set_id") or "").strip()) is None
                            or bool(ai_results.get(str(row.get("set_id") or "").strip()).fallback_used)
                        )
                    )
                )

        ai_duration = time.monotonic() - ai_started

        if rerank_attempt == 0 and self._is_ai_score_collapse(ranked):
            switched = False
            if self._openrouter_model_id:
                switched = self._advance_openrouter_model(reason="score_collapse_guard")
            elif self._model is not None:
                switched = self._advance_gemini_model(reason="score_collapse_guard")

            if switched:
                LOGGER.warning(
                    "AI score collapse detected | candidates=%s spread=%s switching_model=%s rerank_attempt=%s",
                    len(ranked),
                    max(int(row.get("ai_raw_score") or 0) for row in ranked)
                    - min(int(row.get("ai_raw_score") or 0) for row in ranked),
                    self.ai_runtime.get("model"),
                    rerank_attempt + 1,
                )
                return await self._rank_and_persist_candidates(
                    source_candidates,
                    persist=persist,
                    rerank_attempt=rerank_attempt + 1,
                )

        persist_started = time.monotonic()
        persisted_opportunities = 0
        persisted_snapshots = 0
        if persist:
            for opportunity, candidate in opportunities:
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
        persist_duration = time.monotonic() - persist_started
        total_duration = time.monotonic() - started
        historical_prior_applied = sum(1 for row in ranked if row.get("historical_prior_score") is not None)
        self._last_ranking_diagnostics = {
            "input_candidates": len(source_candidates),
            "prepared_candidates": len(prepared),
            "ai_shortlist_count": len(shortlist),
            "ai_prefilter_skipped_count": len(skipped),
            "ai_scored_count": int(ai_stats.get("ai_scored_count", 0)),
            "ai_batch_scored_count": int(ai_stats.get("ai_batch_scored_count", 0)),
            "ai_cache_hits": int(ai_stats.get("ai_cache_hits", 0)),
            "ai_cache_misses": int(ai_stats.get("ai_cache_misses", 0)),
            "ai_persisted_cache_hits": int(ai_stats.get("ai_persisted_cache_hits", 0)),
            "ai_errors": int(ai_stats.get("ai_errors", 0)),
            "ai_budget_exhausted": int(ai_stats.get("ai_budget_exhausted", 0)),
            "ai_timeout_count": int(ai_stats.get("ai_timeout_count", 0)),
            "ai_top_pick_rescue_attempts": int(rescue_stats.get("ai_top_pick_rescue_attempts", 0)),
            "ai_top_pick_rescue_successes": int(rescue_stats.get("ai_top_pick_rescue_successes", 0)),
            "ai_top_pick_rescue_failures": int(rescue_stats.get("ai_top_pick_rescue_failures", 0)),
            "ai_top_pick_rescue_timeouts": int(rescue_stats.get("ai_top_pick_rescue_timeouts", 0)),
            "ai_top_pick_rescue_cache_hits": int(rescue_stats.get("ai_top_pick_rescue_cache_hits", 0)),
            "ai_final_pick_guarantee_rounds": int(rescue_stats.get("ai_final_pick_guarantee_rounds", 0)),
            "ai_final_pick_guarantee_rescue_sets": int(rescue_stats.get("ai_final_pick_guarantee_rescue_sets", 0)),
            "ai_final_pick_guarantee_pending_after_rounds": int(
                rescue_stats.get("ai_final_pick_guarantee_pending_after_rounds", 0)
            ),
            "historical_prior_applied_count": historical_prior_applied,
            "quant_prep_sec": round(prep_duration, 2),
            "ai_scoring_sec": round(ai_duration, 2),
            "persistence_sec": round(persist_duration, 2),
            "total_sec": round(total_duration, 2),
        }
        LOGGER.info(
            "Ranking completed | candidates=%s shortlisted=%s ai_scored=%s cache_hits=%s persisted_cache_hits=%s rescue_attempts=%s rescue_successes=%s final_pick_guarantee_rounds=%s final_pick_pending=%s historical_prior_applied=%s persisted_opportunities=%s persisted_snapshots=%s durations={prep:%.2fs ai:%.2fs persist:%.2fs total:%.2fs}",
            len(source_candidates),
            len(shortlist),
            self._last_ranking_diagnostics["ai_scored_count"],
            self._last_ranking_diagnostics["ai_cache_hits"],
            self._last_ranking_diagnostics["ai_persisted_cache_hits"],
            self._last_ranking_diagnostics["ai_top_pick_rescue_attempts"],
            self._last_ranking_diagnostics["ai_top_pick_rescue_successes"],
            self._last_ranking_diagnostics["ai_final_pick_guarantee_rounds"],
            self._last_ranking_diagnostics["ai_final_pick_guarantee_pending_after_rounds"],
            self._last_ranking_diagnostics["historical_prior_applied_count"],
            persisted_opportunities,
            persisted_snapshots,
            prep_duration,
            ai_duration,
            persist_duration,
            total_duration,
        )
        return ranked

    def _build_ranked_payloads(
        self,
        *,
        prepared: list[Dict[str, Any]],
        ai_results: Dict[str, AIInsight],
        skipped_set_ids: Dict[str, Dict[str, Any]],
        shortlist_count: int,
    ) -> tuple[list[Dict[str, Any]], list[tuple[OpportunityRadarRecord, Dict[str, Any]]]]:
        ranked: list[Dict[str, Any]] = []
        opportunities: list[tuple[OpportunityRadarRecord, Dict[str, Any]]] = []

        for row in prepared:
            candidate = row["candidate"]
            set_id = row["set_id"]
            theme = row["theme"]
            forecast = row["forecast"]
            history_30 = row["history_30"]

            ai = ai_results.get(set_id)
            if ai is None:
                ai = self._heuristic_ai_fallback(candidate)
                if set_id in skipped_set_ids:
                    pre_rank = int(skipped_set_ids[set_id].get("prefilter_rank") or 0)
                    rescue_attempted = bool(row.get("ai_rescue_attempted"))
                    rescue_failed = bool(row.get("ai_rescue_failed"))
                    rescue_reason = str(row.get("ai_rescue_reason") or "").strip().lower()
                    if rescue_attempted and rescue_failed:
                        if rescue_reason == "timeout":
                            rescue_note = (
                                "Tentativo AI esterno sui top pick non riuscito (timeout provider): "
                                "ranking calcolato con fallback euristico."
                            )
                        elif rescue_reason in {"provider_error", "provider_fallback"}:
                            rescue_note = (
                                "Tentativo AI esterno sui top pick non riuscito (errore provider): "
                                "ranking calcolato con fallback euristico."
                            )
                        else:
                            rescue_note = (
                                "Tentativo AI esterno sui top pick non riuscito nel ciclo corrente: "
                                "ranking calcolato con fallback euristico."
                            )
                    ai = AIInsight(
                        score=ai.score,
                        summary=(
                            "Tentativo AI esterno effettuato sui top pick ma non riuscito; "
                            "applicato fallback quantitativo del ciclo."
                            if (rescue_attempted and rescue_failed)
                            else
                            "AI scoring saltato per ottimizzazione costi/latency; "
                            "set fuori shortlist quantitativa del ciclo."
                        ),
                        predicted_eol_date=ai.predicted_eol_date or candidate.get("eol_date_prediction"),
                        fallback_used=True,
                        confidence="LOW_CONFIDENCE",
                        risk_note=(
                            rescue_note
                            if (rescue_attempted and rescue_failed)
                            else f"AI non eseguita: pre-filter rank #{pre_rank} oltre top {shortlist_count}."
                        ),
                    )

            pattern_eval = self._evaluate_success_patterns(candidate)
            effective_pattern_score = self._effective_pattern_score(
                pattern_eval=pattern_eval,
                ai_fallback_used=bool(ai.fallback_used),
            )
            historical_prior = self._historical_prior_for_candidate(candidate)
            effective_confidence_score = self._effective_confidence_score(
                forecast=forecast,
                historical_prior=historical_prior,
                ai=ai,
                pattern_eval=pattern_eval,
            )
            demand = self._estimate_market_demand(
                candidate,
                ai.score,
                forecast=forecast,
                recent_prices=history_30,
            )
            historical_score = (
                int(historical_prior.get("prior_score"))
                if historical_prior is not None and historical_prior.get("prior_score") is not None
                else None
            )
            composite_score = self._calculate_composite_score(
                ai_score=ai.score,
                demand_score=demand,
                forecast_score=forecast.forecast_score,
                pattern_score=effective_pattern_score,
                ai_fallback_used=bool(ai.fallback_used),
                historical_score=historical_score,
            )

            opportunity = OpportunityRadarRecord(
                set_id=set_id,
                set_name=candidate["set_name"],
                theme=theme,
                source=candidate.get("source", "unknown"),
                eol_date_prediction=ai.predicted_eol_date or candidate.get("eol_date_prediction"),
                market_demand_score=demand,
                ai_investment_score=composite_score,
                ai_analysis_summary=ai.summary,
                current_price=candidate.get("current_price"),
                metadata={
                    "listing_url": candidate.get("listing_url"),
                    "source_metadata": candidate.get("metadata", {}),
                    "ai_fallback_used": bool(ai.fallback_used),
                    "ai_confidence": str(ai.confidence or "HIGH_CONFIDENCE"),
                    "ai_risk_note": ai.risk_note,
                    "ai_raw_score": int(ai.score),
                    "ai_model_raw_score": (
                        int(ai.model_raw_score)
                        if ai.model_raw_score is not None
                        else int(ai.score)
                    ),
                    "ai_score_guardrail_applied": bool(
                        ai.model_raw_score is not None and int(ai.model_raw_score) != int(ai.score)
                    ),
                    "forecast_score": int(forecast.forecast_score),
                    "forecast_probability_upside_12m": round(float(forecast.probability_upside_12m) * 100.0, 2),
                    "expected_roi_12m_pct": round(float(forecast.expected_roi_12m_pct), 2),
                    "forecast_interval_low_pct": round(float(forecast.interval_low_pct), 2),
                    "forecast_interval_high_pct": round(float(forecast.interval_high_pct), 2),
                    "forecast_confidence_score": int(effective_confidence_score),
                    "forecast_confidence_score_base": int(forecast.confidence_score),
                    "forecast_data_points": int(forecast.data_points),
                    "forecast_target_roi_pct": round(float(forecast.target_roi_pct), 2),
                    "forecast_estimated_months_to_target": forecast.estimated_months_to_target,
                    "forecast_rationale": forecast.rationale,
                    "composite_score": int(composite_score),
                    "success_pattern_score": int(effective_pattern_score),
                    "success_pattern_score_raw": int(pattern_eval.score),
                    "success_pattern_confidence": int(pattern_eval.confidence_score),
                    "success_pattern_summary": pattern_eval.summary,
                    "success_patterns": pattern_eval.signals,
                    "success_pattern_features": pattern_eval.features,
                    "prefilter_score": int(row.get("prefilter_score") or 0),
                    "prefilter_rank": int(row.get("prefilter_rank") or 0),
                    "ai_shortlisted": bool(row.get("ai_shortlisted")),
                    "historical_prior_score": historical_score,
                    "historical_sample_size": int(historical_prior.get("sample_size") or 0) if historical_prior else 0,
                    "historical_effective_sample_size": (
                        round(float(historical_prior.get("effective_sample_size") or 0.0), 2)
                        if historical_prior
                        else 0.0
                    ),
                    "historical_avg_roi_12m_pct": (
                        round(float(historical_prior.get("avg_roi_12m_pct") or 0.0), 2)
                        if historical_prior
                        else None
                    ),
                    "historical_win_rate_12m_pct": (
                        round(float(historical_prior.get("win_rate_12m") or 0.0) * 100.0, 2)
                        if historical_prior
                        else None
                    ),
                    "historical_support_confidence": (
                        int(historical_prior.get("support_confidence") or 0)
                        if historical_prior
                        else 0
                    ),
                    "historical_source": historical_prior.get("source") if historical_prior else None,
                },
            )

            payload = opportunity.__dict__.copy()
            payload["market_demand_score"] = demand
            payload["ai_investment_score"] = composite_score
            payload["ai_raw_score"] = ai.score
            payload["ai_model_raw_score"] = (
                int(ai.model_raw_score)
                if ai.model_raw_score is not None
                else int(ai.score)
            )
            payload["ai_score_guardrail_applied"] = bool(
                ai.model_raw_score is not None and int(ai.model_raw_score) != int(ai.score)
            )
            payload["forecast_score"] = forecast.forecast_score
            payload["forecast_probability_upside_12m"] = round(float(forecast.probability_upside_12m) * 100.0, 2)
            payload["expected_roi_12m_pct"] = round(float(forecast.expected_roi_12m_pct), 2)
            payload["forecast_interval_low_pct"] = round(float(forecast.interval_low_pct), 2)
            payload["forecast_interval_high_pct"] = round(float(forecast.interval_high_pct), 2)
            payload["confidence_score"] = int(effective_confidence_score)
            payload["confidence_score_base"] = int(forecast.confidence_score)
            payload["estimated_months_to_target"] = forecast.estimated_months_to_target
            payload["composite_score"] = composite_score
            payload["pattern_score"] = int(effective_pattern_score)
            payload["pattern_score_raw"] = int(pattern_eval.score)
            payload["pattern_confidence_score"] = int(pattern_eval.confidence_score)
            payload["pattern_summary"] = pattern_eval.summary
            payload["pattern_signals"] = pattern_eval.signals
            payload["ai_fallback_used"] = bool(ai.fallback_used)
            payload["ai_confidence"] = str(ai.confidence or "HIGH_CONFIDENCE")
            payload["forecast_rationale"] = forecast.rationale
            payload["prefilter_score"] = int(row.get("prefilter_score") or 0)
            payload["prefilter_rank"] = int(row.get("prefilter_rank") or 0)
            payload["ai_shortlisted"] = bool(row.get("ai_shortlisted"))
            payload["historical_prior_score"] = historical_score
            payload["historical_sample_size"] = int(historical_prior.get("sample_size") or 0) if historical_prior else 0
            payload["historical_effective_sample_size"] = (
                round(float(historical_prior.get("effective_sample_size") or 0.0), 2) if historical_prior else 0.0
            )
            payload["historical_avg_roi_12m_pct"] = (
                round(float(historical_prior.get("avg_roi_12m_pct") or 0.0), 2) if historical_prior else None
            )
            payload["historical_win_rate_12m_pct"] = (
                round(float(historical_prior.get("win_rate_12m") or 0.0) * 100.0, 2) if historical_prior else None
            )
            payload["historical_support_confidence"] = (
                int(historical_prior.get("support_confidence") or 0) if historical_prior else 0
            )
            if ai.risk_note:
                payload["risk_note"] = ai.risk_note
            ranked.append(payload)
            opportunities.append((opportunity, candidate))

        return ranked, opportunities

    def _external_ai_available(self) -> bool:
        if self._model is not None:
            return True
        if self._openrouter_model_id is not None:
            return True
        return bool(self.openrouter_api_key)

    async def _rescue_top_pick_ai_scores(
        self,
        *,
        prepared: list[Dict[str, Any]],
        ranked: list[Dict[str, Any]],
        ai_results: Dict[str, AIInsight],
        force_set_ids: Optional[list[str]] = None,
        top_k_override: Optional[int] = None,
    ) -> Dict[str, int]:
        stats = {
            "ai_top_pick_rescue_attempts": 0,
            "ai_top_pick_rescue_successes": 0,
            "ai_top_pick_rescue_failures": 0,
            "ai_top_pick_rescue_timeouts": 0,
            "ai_top_pick_rescue_cache_hits": 0,
        }
        if not ranked:
            return stats

        prepared_by_set = {str(row.get("set_id")): row for row in prepared}
        requested_set_ids: list[str] = []
        if force_set_ids:
            for raw_set_id in force_set_ids:
                set_id = str(raw_set_id or "").strip()
                if set_id and set_id not in requested_set_ids:
                    requested_set_ids.append(set_id)

        top_k = int(self.ai_top_pick_rescue_count) if top_k_override is None else int(top_k_override)
        top_k = max(0, top_k)
        if top_k > 0:
            top_rows = sorted(
                ranked,
                key=lambda row: (
                    int(row.get("composite_score") or row.get("ai_investment_score") or 0),
                    int(row.get("forecast_score") or 0),
                    int(row.get("market_demand_score") or 0),
                ),
                reverse=True,
            )[:top_k]
            for row in top_rows:
                set_id = str(row.get("set_id") or "").strip()
                if set_id and set_id not in requested_set_ids:
                    requested_set_ids.append(set_id)

        rescue_candidates: list[Dict[str, Any]] = []

        for set_id in requested_set_ids:
            if not set_id:
                continue
            current_ai = ai_results.get(set_id)
            if current_ai is not None and not current_ai.fallback_used:
                continue
            prepared_row = prepared_by_set.get(set_id)
            if prepared_row is None:
                continue
            prepared_row["ai_rescue_attempted"] = True
            rescue_candidates.append(prepared_row["candidate"])

        if not rescue_candidates:
            return stats

        stats["ai_top_pick_rescue_cache_hits"] = self._prime_ai_cache_from_repository(rescue_candidates)
        external_available = self._external_ai_available()
        if self.ai_single_call_scoring_enabled:
            unresolved: list[Dict[str, Any]] = []
            for candidate in rescue_candidates:
                set_id = str(candidate.get("set_id") or "").strip()
                if not set_id:
                    continue
                cached = self._get_cached_ai_insight(candidate)
                if cached is not None and not cached.fallback_used:
                    ai_results[set_id] = cached
                    prepared_row = prepared_by_set.get(set_id)
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = False
                        prepared_row["ai_rescue_reason"] = "cache_hit"
                    continue
                if not external_available:
                    prepared_row = prepared_by_set.get(set_id)
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = True
                        prepared_row["ai_rescue_reason"] = "no_external_ai"
                    continue
                unresolved.append(candidate)

            if unresolved and external_available:
                stats["ai_top_pick_rescue_attempts"] += 1
                entries = [
                    {
                        "set_id": str(candidate.get("set_id") or "").strip(),
                        "candidate": candidate,
                    }
                    for candidate in unresolved
                    if str(candidate.get("set_id") or "").strip()
                ]
                batch_deadline = time.monotonic() + max(6.0, float(self.ai_top_pick_rescue_timeout_sec) + 1.0)
                batch_results, batch_error = await self._score_ai_shortlist_batch(
                    entries,
                    deadline=batch_deadline,
                    allow_repair_calls=bool(self.ai_single_call_allow_repair_calls),
                )
                if batch_error:
                    LOGGER.warning(
                        "Top pick AI rescue batch failed | candidates=%s error=%s",
                        len(entries),
                        str(batch_error)[:220],
                    )

                for candidate in unresolved:
                    set_id = str(candidate.get("set_id") or "").strip()
                    if not set_id:
                        continue
                    prepared_row = prepared_by_set.get(set_id)
                    insight = batch_results.get(set_id)
                    if insight is None:
                        stats["ai_top_pick_rescue_failures"] += 1
                        if "timeout" in str(batch_error or "").lower():
                            stats["ai_top_pick_rescue_timeouts"] += 1
                        if prepared_row is not None:
                            prepared_row["ai_rescue_failed"] = True
                            prepared_row["ai_rescue_reason"] = "batch_no_result"
                        continue
                    ai_results[set_id] = insight
                    if insight.fallback_used:
                        stats["ai_top_pick_rescue_failures"] += 1
                        if prepared_row is not None:
                            prepared_row["ai_rescue_failed"] = True
                            prepared_row["ai_rescue_reason"] = "provider_fallback"
                        continue
                    self._set_cached_ai_insight(candidate, insight)
                    stats["ai_top_pick_rescue_successes"] += 1
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = False
                        prepared_row["ai_rescue_reason"] = "success"
        else:
            for candidate in rescue_candidates:
                set_id = str(candidate.get("set_id") or "").strip()
                if not set_id:
                    continue
                cached = self._get_cached_ai_insight(candidate)
                if cached is not None and not cached.fallback_used:
                    ai_results[set_id] = cached
                    prepared_row = prepared_by_set.get(set_id)
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = False
                        prepared_row["ai_rescue_reason"] = "cache_hit"
                    continue
                if not external_available:
                    prepared_row = prepared_by_set.get(set_id)
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = True
                        prepared_row["ai_rescue_reason"] = "no_external_ai"
                    continue

                stats["ai_top_pick_rescue_attempts"] += 1
                try:
                    insight = await asyncio.wait_for(
                        self._get_ai_insight(candidate),
                        timeout=float(self.ai_top_pick_rescue_timeout_sec),
                    )
                except asyncio.TimeoutError:
                    stats["ai_top_pick_rescue_failures"] += 1
                    stats["ai_top_pick_rescue_timeouts"] += 1
                    prepared_row = prepared_by_set.get(set_id)
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = True
                        prepared_row["ai_rescue_reason"] = "timeout"
                    LOGGER.warning(
                        "Top pick AI rescue timeout | set_id=%s timeout_sec=%.1f",
                        set_id,
                        self.ai_top_pick_rescue_timeout_sec,
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    stats["ai_top_pick_rescue_failures"] += 1
                    prepared_row = prepared_by_set.get(set_id)
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = True
                        prepared_row["ai_rescue_reason"] = "provider_error"
                    LOGGER.warning("Top pick AI rescue failed | set_id=%s error=%s", set_id, exc)
                    continue

                ai_results[set_id] = insight
                if insight.fallback_used:
                    stats["ai_top_pick_rescue_failures"] += 1
                    prepared_row = prepared_by_set.get(set_id)
                    if prepared_row is not None:
                        prepared_row["ai_rescue_failed"] = True
                        prepared_row["ai_rescue_reason"] = "provider_fallback"
                    continue
                self._set_cached_ai_insight(candidate, insight)
                stats["ai_top_pick_rescue_successes"] += 1
                prepared_row = prepared_by_set.get(set_id)
                if prepared_row is not None:
                    prepared_row["ai_rescue_failed"] = False
                    prepared_row["ai_rescue_reason"] = "success"

        if stats["ai_top_pick_rescue_attempts"] > 0:
            LOGGER.info(
                "Top pick AI rescue summary | top_k=%s forced=%s candidates=%s attempts=%s successes=%s failures=%s timeouts=%s cache_hits=%s",
                top_k,
                len(force_set_ids or []),
                len(rescue_candidates),
                stats["ai_top_pick_rescue_attempts"],
                stats["ai_top_pick_rescue_successes"],
                stats["ai_top_pick_rescue_failures"],
                stats["ai_top_pick_rescue_timeouts"],
                stats["ai_top_pick_rescue_cache_hits"],
            )
        return stats

    def _prepare_quantitative_context(self, source_candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        set_ids = [str(row.get("set_id") or "").strip() for row in source_candidates if str(row.get("set_id") or "").strip()]
        history_by_set = self._load_history_by_set(set_ids)
        theme_baseline_cache: Dict[str, Dict[str, float]] = {}
        prepared: list[Dict[str, Any]] = []

        for candidate in source_candidates:
            set_id = str(candidate.get("set_id") or "").strip()
            theme = str(candidate.get("theme") or "Unknown")
            history = history_by_set.get(set_id, [])
            history_30 = self._recent_rows_within_days(history, days=30)

            if theme in theme_baseline_cache:
                theme_baseline = theme_baseline_cache[theme]
            else:
                try:
                    theme_baseline = self.repository.get_theme_radar_baseline(
                        theme,
                        days=self.history_window_days,
                        limit=120,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Theme baseline fetch failed for theme=%s: %s", theme, exc)
                    theme_baseline = {}
                theme_baseline_cache[theme] = theme_baseline

            try:
                forecast = self.forecaster.forecast(
                    candidate=candidate,
                    history_rows=history,
                    theme_baseline=theme_baseline,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Forecast failed for set_id=%s: %s", set_id, exc)
                forecast = ForecastInsight(
                    forecast_score=22,
                    probability_upside_12m=0.22,
                    expected_roi_12m_pct=0.0,
                    interval_low_pct=-18.0,
                    interval_high_pct=24.0,
                    target_roi_pct=float(self.target_roi_pct),
                    estimated_months_to_target=None,
                    confidence_score=28,
                    data_points=len(history),
                    rationale="Fallback forecast per errore calcolo.",
                )

            prefilter_score = self._calculate_prefilter_score(
                candidate=candidate,
                forecast=forecast,
                recent_rows=history_30,
            )
            prepared.append(
                {
                    "candidate": candidate,
                    "set_id": set_id,
                    "theme": theme,
                    "forecast": forecast,
                    "history_30": history_30,
                    "prefilter_score": prefilter_score,
                    "prefilter_rank": 0,
                    "ai_shortlisted": False,
                }
            )

        prepared.sort(
            key=lambda row: (
                int(row.get("prefilter_score") or 0),
                int(row["forecast"].forecast_score),
                int(self._source_priority(row["candidate"].get("source"))),
            ),
            reverse=True,
        )
        for idx, row in enumerate(prepared, start=1):
            row["prefilter_rank"] = idx
        return prepared

    def _select_ai_shortlist(self, prepared: list[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        if not prepared:
            return [], []

        if self.ai_single_call_scoring_enabled:
            shortlist_cap = self._effective_ai_single_call_shortlist_cap(len(prepared))
            shortlist = prepared[:shortlist_cap]
            skipped = prepared[shortlist_cap:]
            for row in shortlist:
                row["ai_shortlisted"] = True
            if skipped:
                LOGGER.info(
                    "AI single-call shortlist cap applied | candidates=%s selected=%s skipped=%s cap_effective=%s cap_configured=%s",
                    len(prepared),
                    len(shortlist),
                    len(skipped),
                    shortlist_cap,
                    self.ai_single_call_max_candidates,
                )
            return shortlist, skipped

        shortlist_count = self._effective_ai_shortlist_limit(len(prepared))
        shortlist = prepared[:shortlist_count]
        skipped = prepared[shortlist_count:]
        for row in shortlist:
            row["ai_shortlisted"] = True
        return shortlist, skipped

    def _effective_ai_single_call_shortlist_cap(self, candidate_count: int) -> int:
        if candidate_count <= 0:
            return 0

        base_cap = min(candidate_count, max(3, int(self.ai_single_call_max_candidates)))
        if not self.ai_dynamic_shortlist_enabled:
            return base_cap

        engine = str(self.ai_runtime.get("engine") or "")
        if engine != "openrouter":
            return base_cap

        inventory_available = int(
            self.ai_runtime.get("inventory_available")
            or len(self._openrouter_available_candidates)
            or 0
        )
        if inventory_available <= 0:
            return base_cap

        adaptive_floor = max(3, int(self.ai_dynamic_shortlist_floor))
        if inventory_available >= 2:
            adaptive_floor = max(adaptive_floor, int(self.ai_dynamic_shortlist_multi_model_floor))

        effective = min(candidate_count, max(base_cap, adaptive_floor))
        if effective > base_cap:
            LOGGER.info(
                "AI single-call shortlist floor raised | candidates=%s base=%s effective=%s inventory_available=%s engine=%s",
                candidate_count,
                base_cap,
                effective,
                inventory_available,
                engine,
            )
        return effective

    def _effective_ai_shortlist_limit(self, candidate_count: int) -> int:
        if candidate_count <= 0:
            return 0

        base_limit = min(candidate_count, max(1, int(self.ai_rank_max_candidates)))
        if not self.ai_dynamic_shortlist_enabled:
            return base_limit

        engine = str(self.ai_runtime.get("engine") or "")
        if engine != "openrouter":
            return base_limit

        inventory_available = int(
            self.ai_runtime.get("inventory_available")
            or len(self._openrouter_available_candidates)
            or 0
        )
        adaptive_floor = int(self.ai_dynamic_shortlist_floor)
        if inventory_available >= 2:
            adaptive_floor = max(adaptive_floor, int(self.ai_dynamic_shortlist_multi_model_floor))
        if inventory_available <= 0:
            return min(base_limit, max(2, adaptive_floor))

        dynamic_limit = int(self.ai_dynamic_shortlist_bonus) + (inventory_available * int(self.ai_dynamic_shortlist_per_model))
        dynamic_limit = max(adaptive_floor, dynamic_limit)
        effective = min(base_limit, max(2, dynamic_limit))
        if effective < base_limit:
            LOGGER.info(
                "AI shortlist dynamically reduced | candidates=%s base=%s effective=%s inventory_available=%s engine=%s",
                candidate_count,
                base_limit,
                effective,
                inventory_available,
                engine,
            )
        return effective

    def _calculate_prefilter_score(
        self,
        *,
        candidate: Dict[str, Any],
        forecast: ForecastInsight,
        recent_rows: list[Dict[str, Any]],
    ) -> int:
        source_component = float(min(100, max(1, self._source_priority(candidate.get("source")))))
        confidence_component = float(max(1, min(100, int(forecast.confidence_score))))
        forecast_component = float(max(1, min(100, int(forecast.forecast_score))))
        liquidity_component = float(min(100, len(recent_rows) * 5))
        price = float(candidate.get("current_price") or 0.0)
        price_band_bonus = 10.0 if 20.0 <= price <= 250.0 else 2.0

        score = (
            0.58 * forecast_component
            + 0.22 * confidence_component
            + 0.12 * source_component
            + 0.08 * min(100.0, liquidity_component + price_band_bonus)
        )
        return max(1, min(100, int(round(score))))

    async def _score_ai_shortlist(
        self,
        shortlist: list[Dict[str, Any]],
    ) -> tuple[Dict[str, AIInsight], Dict[str, int]]:
        stats = {
            "ai_scored_count": 0,
            "ai_cache_hits": 0,
            "ai_cache_misses": 0,
            "ai_persisted_cache_hits": 0,
            "ai_errors": 0,
            "ai_budget_exhausted": 0,
            "ai_timeout_count": 0,
            "ai_batch_scored_count": 0,
            "ai_single_call_rescue_attempted": 0,
            "ai_single_call_rescue_scored": 0,
        }
        if not shortlist:
            return {}, stats

        stats["ai_persisted_cache_hits"] = self._prime_ai_cache_from_repository(
            [entry.get("candidate") or {} for entry in shortlist],
        )

        effective_concurrency = max(1, int(self.ai_scoring_concurrency))
        if str(self.ai_runtime.get("engine") or "") == "openrouter":
            inventory_available = int(
                self.ai_runtime.get("inventory_available")
                or len(self._openrouter_available_candidates)
                or 0
            )
            if inventory_available <= 1:
                effective_concurrency = min(effective_concurrency, 2)
        semaphore = asyncio.Semaphore(effective_concurrency)
        results: Dict[str, AIInsight] = {}
        started = time.monotonic()
        deadline = started + self.ai_scoring_hard_budget_sec
        entry_by_set: Dict[str, Dict[str, Any]] = {str(row["set_id"]): row for row in shortlist}
        pending_entries: list[Dict[str, Any]] = []
        single_call_mode = bool(self.ai_single_call_scoring_enabled)

        # First resolve cache hits in a deterministic pass.
        for entry in shortlist:
            set_id = str(entry["set_id"])
            candidate = entry["candidate"]
            cached = self._get_cached_ai_insight(candidate)
            if cached is None:
                pending_entries.append(entry)
                continue
            results[set_id] = cached
            stats["ai_cache_hits"] += 1

        # Batch scoring pass: in single-call mode this is the only external scoring call.
        if pending_entries and (self.ai_batch_scoring_enabled or single_call_mode):
            if single_call_mode:
                batch_entries = list(pending_entries)
            else:
                min_batch = int(self.ai_batch_min_candidates)
                if len(pending_entries) < min_batch:
                    batch_entries = []
                else:
                    batch_cap = max(min_batch, int(self.ai_batch_max_candidates))
                    batch_entries = pending_entries[: min(len(pending_entries), batch_cap)]

            batch_results: Dict[str, AIInsight] = {}
            batch_error: Optional[str] = None
            if batch_entries:
                batch_results, batch_error = await self._score_ai_shortlist_batch(
                    batch_entries,
                    deadline=deadline,
                    allow_repair_calls=(not single_call_mode) or bool(self.ai_single_call_allow_repair_calls),
                    allow_failover_call=not single_call_mode,
                )

            if batch_error:
                non_error_reasons = {"no_external_ai_available", "insufficient_budget_for_batch"}
                if str(batch_error) in non_error_reasons:
                    LOGGER.info(
                        "AI batch scoring skipped | candidates=%s reason=%s",
                        len(batch_entries),
                        str(batch_error)[:220],
                    )
                else:
                    stats["ai_errors"] += 1
                    LOGGER.warning(
                        "AI batch scoring failed | candidates=%s error=%s",
                        len(batch_entries),
                        str(batch_error)[:220],
                    )
            for entry in batch_entries:
                set_id = str(entry["set_id"])
                ai = batch_results.get(set_id)
                if ai is None:
                    continue
                results[set_id] = ai
                stats["ai_cache_misses"] += 1
                if not ai.fallback_used:
                    stats["ai_scored_count"] += 1
                    stats["ai_batch_scored_count"] += 1
                    self._set_cached_ai_insight(entry["candidate"], ai)
            pending_entries = [entry for entry in pending_entries if str(entry["set_id"]) not in results]

        if single_call_mode:
            if pending_entries and self.ai_single_call_missing_rescue_enabled:
                rescue_entries = pending_entries[
                    : max(1, int(self.ai_single_call_missing_rescue_max_candidates))
                ]
                budget_left_for_rescue = max(0.0, deadline - time.monotonic())
                rescue_timeout_sec = min(
                    float(self.ai_single_call_missing_rescue_timeout_sec),
                    budget_left_for_rescue,
                )
                if rescue_timeout_sec >= 6.0 and rescue_entries:
                    stats["ai_single_call_rescue_attempted"] = 1
                    rescue_deadline = time.monotonic() + rescue_timeout_sec
                    rescue_results, rescue_error = await self._score_ai_shortlist_batch(
                        rescue_entries,
                        deadline=rescue_deadline,
                        allow_repair_calls=True,
                        allow_failover_call=False,
                    )
                    if rescue_error:
                        non_error_reasons = {"no_external_ai_available", "insufficient_budget_for_batch"}
                        if str(rescue_error) in non_error_reasons:
                            LOGGER.info(
                                "AI single-call missing rescue skipped | candidates=%s reason=%s",
                                len(rescue_entries),
                                str(rescue_error)[:220],
                            )
                        else:
                            stats["ai_errors"] += 1
                            LOGGER.warning(
                                "AI single-call missing rescue failed | candidates=%s error=%s",
                                len(rescue_entries),
                                str(rescue_error)[:220],
                            )
                    for entry in rescue_entries:
                        set_id = str(entry["set_id"])
                        ai = rescue_results.get(set_id)
                        if ai is None:
                            continue
                        results[set_id] = ai
                        stats["ai_cache_misses"] += 1
                        if not ai.fallback_used:
                            stats["ai_scored_count"] += 1
                            stats["ai_batch_scored_count"] += 1
                            stats["ai_single_call_rescue_scored"] += 1
                            self._set_cached_ai_insight(entry["candidate"], ai)
                    pending_entries = [entry for entry in pending_entries if str(entry["set_id"]) not in results]
                elif rescue_entries:
                    LOGGER.info(
                        "AI single-call missing rescue skipped | reason=insufficient_budget timeout_sec=%.2f pending=%s",
                        rescue_timeout_sec,
                        len(pending_entries),
                    )

            for entry in pending_entries:
                set_id = str(entry["set_id"])
                fallback = self._heuristic_ai_fallback(entry["candidate"])
                fallback = AIInsight(
                    score=fallback.score,
                    summary=fallback.summary,
                    predicted_eol_date=fallback.predicted_eol_date,
                    fallback_used=True,
                    confidence="LOW_CONFIDENCE",
                    risk_note=(
                        "AI single-call batch non ha restituito output valido per questo set: "
                        "applicato fallback euristico."
                    ),
                )
                results[set_id] = fallback
                stats["ai_cache_misses"] += 1
                stats["ai_budget_exhausted"] += 1

            elapsed = time.monotonic() - started
            LOGGER.info(
                "AI shortlist scoring summary | candidates=%s scored=%s batch_scored=%s cache_hits=%s persisted_cache_hits=%s cache_misses=%s errors=%s timeouts=%s budget_exhausted=%s concurrency=%s elapsed=%.2fs budget=%.2fs single_call=%s single_call_rescue_attempted=%s single_call_rescue_scored=%s",
                len(shortlist),
                stats["ai_scored_count"],
                stats["ai_batch_scored_count"],
                stats["ai_cache_hits"],
                stats["ai_persisted_cache_hits"],
                stats["ai_cache_misses"],
                stats["ai_errors"],
                stats["ai_timeout_count"],
                stats["ai_budget_exhausted"],
                effective_concurrency,
                elapsed,
                self.ai_scoring_hard_budget_sec,
                single_call_mode,
                stats["ai_single_call_rescue_attempted"],
                stats["ai_single_call_rescue_scored"],
            )
            return results, stats

        budget_left = max(1.0, deadline - time.monotonic())
        item_timeout_sec, retry_timeout_sec, timeout_retries = self._compute_fast_fail_timeouts(
            pending_count=len(pending_entries),
            budget_left_sec=budget_left,
        )

        if pending_entries and self.ai_fast_fail_enabled:
            LOGGER.info(
                "AI fast-fail active | pending=%s budget_left=%.1fs item_timeout=%.1fs retry_timeout=%.1fs retries=%s",
                len(pending_entries),
                budget_left,
                item_timeout_sec,
                retry_timeout_sec,
                timeout_retries,
            )

        async def worker(entry: Dict[str, Any]) -> tuple[str, AIInsight, bool, Optional[Exception]]:
            candidate = entry["candidate"]
            set_id = entry["set_id"]

            async with semaphore:
                retries = max(0, int(timeout_retries))
                first_timeout = float(item_timeout_sec)
                retry_timeout = min(first_timeout, float(retry_timeout_sec))
                attempt_timeouts = [first_timeout] + [retry_timeout] * retries
                total_attempts = len(attempt_timeouts)

                for attempt_idx, attempt_timeout in enumerate(attempt_timeouts, start=1):
                    try:
                        ai = await asyncio.wait_for(
                            self._get_ai_insight(candidate),
                            timeout=attempt_timeout,
                        )
                        return set_id, ai, False, None
                    except asyncio.TimeoutError:
                        if attempt_idx < total_attempts:
                            LOGGER.warning(
                                "AI scoring timeout | set_id=%s source=%s attempt=%s/%s timeout_sec=%.1f engine=%s model=%s mode=%s",
                                set_id,
                                candidate.get("source"),
                                attempt_idx,
                                total_attempts,
                                attempt_timeout,
                                self.ai_runtime.get("engine"),
                                self.ai_runtime.get("model"),
                                self.ai_runtime.get("mode"),
                            )
                            if self._openrouter_model_id is not None:
                                await self._advance_openrouter_model_locked(
                                    reason=f"scoring_timeout:{set_id}:attempt_{attempt_idx}",
                                )
                            continue
                        recovered = await self._recover_openrouter_after_timeout(set_id=str(set_id))
                        if recovered:
                            LOGGER.warning(
                                "AI scoring timeout recovery active | set_id=%s source=%s model=%s",
                                set_id,
                                candidate.get("source"),
                                self.ai_runtime.get("model"),
                            )
                            try:
                                ai = await asyncio.wait_for(
                                    self._get_ai_insight(candidate),
                                    timeout=retry_timeout,
                                )
                                return set_id, ai, False, None
                            except Exception as recovery_exc:  # noqa: BLE001
                                timeout_err = asyncio.TimeoutError(
                                    f"timeout after recovery (attempt {attempt_idx}/{total_attempts}): {recovery_exc}"
                                )
                                return set_id, self._heuristic_ai_fallback(candidate), False, timeout_err
                        timeout_err = asyncio.TimeoutError(
                            f"timeout after {attempt_timeout:.1f}s (attempt {attempt_idx}/{total_attempts})"
                        )
                        return set_id, self._heuristic_ai_fallback(candidate), False, timeout_err
                    except Exception as exc:  # noqa: BLE001
                        return set_id, self._heuristic_ai_fallback(candidate), False, exc

                timeout_err = asyncio.TimeoutError(
                    f"timeout after retries ({total_attempts} attempts)"
                )
                return set_id, self._heuristic_ai_fallback(candidate), False, timeout_err

        tasks_by_set = {
            str(entry["set_id"]): asyncio.create_task(worker(entry))
            for entry in pending_entries
        }

        while tasks_by_set:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for pending_set_id, pending_task in list(tasks_by_set.items()):
                    pending_task.cancel()
                    entry = entry_by_set.get(pending_set_id)
                    candidate = entry["candidate"] if entry else {"set_id": pending_set_id}
                    if pending_set_id not in results:
                        fallback = self._heuristic_ai_fallback(candidate)
                        fallback = AIInsight(
                            score=fallback.score,
                            summary=fallback.summary,
                            predicted_eol_date=fallback.predicted_eol_date,
                            fallback_used=True,
                            confidence="LOW_CONFIDENCE",
                            risk_note=(
                                "AI budget esaurito nel ciclo corrente: score calcolato con fallback euristico."
                            ),
                        )
                        results[pending_set_id] = fallback
                        stats["ai_cache_misses"] += 1
                        stats["ai_budget_exhausted"] += 1
                tasks_by_set.clear()
                break

            done, _pending = await asyncio.wait(
                list(tasks_by_set.values()),
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue

            for task in done:
                set_id = next((key for key, value in tasks_by_set.items() if value is task), None)
                if set_id is None:
                    continue
                entry = entry_by_set.get(set_id)
                candidate = entry["candidate"] if entry else {"set_id": set_id}
                tasks_by_set.pop(set_id, None)
                err: Optional[Exception] = None
                from_cache = False

                try:
                    result_set_id, ai, from_cache, err = task.result()
                    set_id = str(result_set_id)
                except Exception as exc:  # noqa: BLE001
                    ai = self._heuristic_ai_fallback(candidate)
                    err = exc
                    from_cache = False

                if from_cache:
                    stats["ai_cache_hits"] += 1
                else:
                    stats["ai_cache_misses"] += 1
                    if not ai.fallback_used:
                        stats["ai_scored_count"] += 1
                        self._set_cached_ai_insight(candidate, ai)

                if isinstance(err, asyncio.TimeoutError):
                    stats["ai_timeout_count"] += 1
                if err is not None:
                    stats["ai_errors"] += 1
                    err_type, err_message = self._format_exception_for_log(err)
                    LOGGER.warning(
                        "AI scoring worker failed | set_id=%s source=%s engine=%s model=%s mode=%s error_type=%s error=%s",
                        set_id,
                        candidate.get("source"),
                        self.ai_runtime.get("engine"),
                        self.ai_runtime.get("model"),
                        self.ai_runtime.get("mode"),
                        err_type,
                        err_message,
                    )
                results[set_id] = ai

        # Safety net: ensure every shortlisted candidate has a score.
        for set_id, entry in entry_by_set.items():
            if set_id in results:
                continue
            fallback = self._heuristic_ai_fallback(entry["candidate"])
            fallback = AIInsight(
                score=fallback.score,
                summary=fallback.summary,
                predicted_eol_date=fallback.predicted_eol_date,
                fallback_used=True,
                confidence="LOW_CONFIDENCE",
                risk_note="Score fallback: candidato non processato entro il budget AI del ciclo.",
            )
            results[set_id] = fallback
            stats["ai_cache_misses"] += 1
            stats["ai_budget_exhausted"] += 1

        elapsed = time.monotonic() - started
        LOGGER.info(
            "AI shortlist scoring summary | candidates=%s scored=%s batch_scored=%s cache_hits=%s persisted_cache_hits=%s cache_misses=%s errors=%s timeouts=%s budget_exhausted=%s concurrency=%s elapsed=%.2fs budget=%.2fs single_call=%s single_call_rescue_attempted=%s single_call_rescue_scored=%s",
            len(shortlist),
            stats["ai_scored_count"],
            stats["ai_batch_scored_count"],
            stats["ai_cache_hits"],
            stats["ai_persisted_cache_hits"],
            stats["ai_cache_misses"],
            stats["ai_errors"],
            stats["ai_timeout_count"],
            stats["ai_budget_exhausted"],
            effective_concurrency,
            elapsed,
            self.ai_scoring_hard_budget_sec,
            single_call_mode,
            stats["ai_single_call_rescue_attempted"],
            stats["ai_single_call_rescue_scored"],
        )
        return results, stats

    async def _score_ai_shortlist_batch(
        self,
        entries: list[Dict[str, Any]],
        *,
        deadline: float,
        allow_repair_calls: bool = True,
        allow_failover_call: bool = True,
    ) -> tuple[Dict[str, AIInsight], Optional[str]]:
        if not entries:
            return {}, None

        remaining = deadline - time.monotonic()
        if remaining <= 6.0:
            return {}, "insufficient_budget_for_batch"

        timeout_sec = min(
            float(self.ai_batch_timeout_sec),
            float(self.ai_generation_timeout_sec),
            max(6.0, remaining - 1.0),
        )
        candidates = [entry["candidate"] for entry in entries]
        prompt = self._build_batch_ai_prompt(candidates)

        if self._model is not None:
            current_model = str(self.gemini_model or "")
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(self._gemini_generate, prompt),
                    timeout=timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001
                if current_model:
                    self._record_model_failure("gemini", current_model, str(exc), phase="batch_scoring")
                return {}, str(exc)

            try:
                payload = self._extract_json(text)
                insights = self._batch_payload_to_ai_insights(payload, candidates)
            except Exception:
                insights = self._batch_insights_from_unstructured_text(text, candidates)
            insights = self._normalize_batch_ai_insights(insights, candidates)

            if insights:
                if current_model:
                    self._record_model_success("gemini", current_model, phase="batch_scoring")
                LOGGER.info(
                    "AI batch scoring success | provider=gemini model=%s candidates=%s scored=%s",
                    current_model or "unknown",
                    len(candidates),
                    len(insights),
                )
                return insights, None
            return {}, "batch_payload_no_valid_rows"

        if self._openrouter_model_id is not None:
            async def _insights_from_openrouter_text(raw_text: str) -> Dict[str, AIInsight]:
                try:
                    payload = self._extract_json(raw_text)
                    return self._batch_payload_to_ai_insights(payload, candidates)
                except Exception:
                    parsed = self._batch_insights_from_unstructured_text(raw_text, candidates)
                    if parsed:
                        return parsed
                    if not allow_repair_calls:
                        return {}
                    repaired = await self._repair_openrouter_non_json_batch_output(
                        raw_text=raw_text,
                        candidates=candidates,
                        timeout_sec=timeout_sec,
                    )
                    return repaired or {}

            current_model = str(self._openrouter_model_id or "")
            try:
                text = await asyncio.to_thread(
                    self._openrouter_generate,
                    prompt,
                    request_timeout=timeout_sec,
                )
                insights = await _insights_from_openrouter_text(text)
                insights = self._normalize_batch_ai_insights(insights, candidates)
                if insights:
                    self._record_model_success("openrouter", current_model, phase="batch_scoring")
                    LOGGER.info(
                        "AI batch scoring success | provider=openrouter model=%s candidates=%s scored=%s",
                        current_model,
                        len(candidates),
                        len(insights),
                    )
                    return insights, None
                return {}, "batch_payload_no_valid_rows"
            except Exception as exc:  # noqa: BLE001
                self._record_model_failure("openrouter", current_model, str(exc), phase="batch_scoring")
                if not allow_failover_call:
                    return {}, str(exc)
                rotated = await self._advance_openrouter_model_locked(reason=f"batch_scoring:{exc}")
                if rotated:
                    rotated_model = str(self._openrouter_model_id or "")
                    try:
                        text = await asyncio.to_thread(
                            self._openrouter_generate,
                            prompt,
                            request_timeout=timeout_sec,
                        )
                        insights = await _insights_from_openrouter_text(text)
                        insights = self._normalize_batch_ai_insights(insights, candidates)
                        if insights:
                            self._record_model_success("openrouter", rotated_model, phase="batch_scoring_after_failover")
                            LOGGER.info(
                                "AI batch scoring success after failover | provider=openrouter model=%s candidates=%s scored=%s",
                                rotated_model,
                                len(candidates),
                                len(insights),
                            )
                            return insights, None
                    except Exception as exc_after_switch:  # noqa: BLE001
                        self._record_model_failure(
                            "openrouter",
                            rotated_model,
                            str(exc_after_switch),
                            phase="batch_scoring_after_failover",
                        )
                        return {}, str(exc_after_switch)
                return {}, str(exc)

        return {}, "no_external_ai_available"

    def _normalize_batch_ai_insights(
        self,
        insights: Dict[str, AIInsight],
        candidates: list[Dict[str, Any]],
    ) -> Dict[str, AIInsight]:
        if not insights:
            return {}
        candidate_by_set_id = {
            str(row.get("set_id") or "").strip(): row
            for row in candidates
            if str(row.get("set_id") or "").strip()
        }
        normalized: Dict[str, AIInsight] = {}
        for set_id, insight in insights.items():
            candidate = candidate_by_set_id.get(str(set_id), {"set_id": set_id})
            normalized[str(set_id)] = self._normalize_ai_insight(insight, candidate)
        return normalized

    def _compute_fast_fail_timeouts(
        self,
        *,
        pending_count: int,
        budget_left_sec: float,
    ) -> tuple[float, float, int]:
        first_timeout = float(self.ai_scoring_item_timeout_sec)
        retry_timeout = min(first_timeout, float(self.ai_scoring_retry_timeout_sec))
        retries = max(0, int(self.ai_scoring_timeout_retries))

        if not self.ai_fast_fail_enabled or pending_count <= 0:
            return first_timeout, retry_timeout, retries

        per_candidate_budget = max(1.0, float(budget_left_sec)) / max(1, int(pending_count))
        adaptive_first = min(first_timeout, max(4.0, per_candidate_budget * 1.35))
        adaptive_retry = min(retry_timeout, max(2.0, adaptive_first * 0.60))
        reduced = adaptive_first + 1e-9 < first_timeout
        if reduced:
            if adaptive_first <= 6.0:
                retries = 0
            elif adaptive_first <= 9.0:
                retries = min(retries, 1)

        return round(adaptive_first, 2), round(adaptive_retry, 2), retries

    @classmethod
    def _batch_payload_to_ai_insights(
        cls,
        payload: Any,
        candidates: list[Dict[str, Any]],
    ) -> Dict[str, AIInsight]:
        candidate_by_set_id = {
            str(row.get("set_id") or "").strip(): row
            for row in candidates
            if str(row.get("set_id") or "").strip()
        }
        if not candidate_by_set_id:
            return {}

        rows: list[Dict[str, Any]] = []
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            for key in ("results", "items", "picks", "scores"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows = [row for row in value if isinstance(row, dict)]
                    break
            if not rows:
                # Accept map-style payload: {"75367": {"score": 80, ...}, ...}
                for set_id, row in payload.items():
                    if set_id not in candidate_by_set_id or not isinstance(row, dict):
                        continue
                    row_copy = dict(row)
                    row_copy.setdefault("set_id", set_id)
                    rows.append(row_copy)

        insights: Dict[str, AIInsight] = {}
        for row in rows:
            set_id = str(row.get("set_id") or "").strip()
            candidate = candidate_by_set_id.get(set_id)
            if candidate is None:
                continue
            try:
                insight = cls._payload_to_ai_insight(row, candidate)
            except Exception:
                continue
            insights[set_id] = insight
        return insights

    @classmethod
    def _candidate_unstructured_segment(
        cls,
        raw_text: str,
        candidate: Dict[str, Any],
    ) -> Optional[str]:
        text = str(raw_text or "")
        if not text.strip():
            return None

        set_id = str(candidate.get("set_id") or "").strip()
        if set_id:
            for match in re.finditer(re.escape(set_id), text, flags=re.IGNORECASE):
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 260)
                segment = text[start:end]
                if cls._extract_unstructured_score(segment) is not None:
                    return segment

        set_name = str(candidate.get("set_name") or "").strip()
        if set_name:
            tokens = [token for token in re.findall(r"[A-Za-z0-9]+", set_name) if len(token) >= 4][:3]
            if tokens:
                token_pattern = r"\W+".join(re.escape(token) for token in tokens[:2])
                pattern = re.compile(rf"(?is)({token_pattern}.{{0,240}})")
                match = pattern.search(text)
                if match:
                    segment = match.group(1)
                    if cls._extract_unstructured_score(segment) is not None:
                        return segment

        return None

    @classmethod
    def _batch_insights_from_unstructured_text(
        cls,
        raw_text: str,
        candidates: list[Dict[str, Any]],
    ) -> Dict[str, AIInsight]:
        insights: Dict[str, AIInsight] = {}
        text = str(raw_text or "")
        if not text.strip():
            return insights

        candidate_by_set_id = {
            str(row.get("set_id") or "").strip(): row
            for row in candidates
            if str(row.get("set_id") or "").strip()
        }

        # Pass 0: key-value/pipe rows, e.g. "set_id=76281|score=78|summary=...".
        for line in text.splitlines():
            line_text = str(line or "").strip()
            if not line_text:
                continue

            set_id = ""
            score_value: Optional[int] = None
            summary = ""
            eol = None

            kv_match = re.search(
                r"(?is)set[_\s-]?id\s*[:=]\s*(\d{4,6}).{0,180}?score\s*[:=]\s*(100|[1-9]?\d)\b",
                line_text,
            )
            if kv_match:
                set_id = str(kv_match.group(1))
                score_value = int(kv_match.group(2))
                summary_match = re.search(r"(?is)summary\s*[:=]\s*([^|;]+)", line_text)
                if summary_match:
                    summary = str(summary_match.group(1) or "").strip()
                eol = cls._extract_first_date(line_text)
            else:
                pipe_match = re.search(
                    r"(?is)^\s*(?:[-*]\s*)?(?:\d+[\)\.\-:]?\s*)?(\d{4,6})\s*[|;,]\s*(100|[1-9]?\d)\b(.*)$",
                    line_text,
                )
                if pipe_match:
                    set_id = str(pipe_match.group(1))
                    score_value = int(pipe_match.group(2))
                    tail = str(pipe_match.group(3) or "").strip(" |;")
                    summary = tail
                    eol = cls._extract_first_date(tail)

            if not set_id or score_value is None:
                continue
            if set_id in insights:
                continue
            candidate = candidate_by_set_id.get(set_id)
            if candidate is None:
                continue

            score = max(1, min(100, int(score_value)))
            if not summary:
                summary = f"Output AI non JSON: score {score}/100 estratto da formato key-value."
            summary = summary[:1200]
            insights[set_id] = AIInsight(
                score=score,
                summary=summary,
                predicted_eol_date=eol or candidate.get("eol_date_prediction"),
                fallback_used=False,
                confidence="LOW_CONFIDENCE",
                risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
            )

        if len(insights) == len(candidate_by_set_id):
            return insights

        # Pass 1: per-candidate extraction anchored on set_id/name snippets.
        for candidate in candidates:
            set_id = str(candidate.get("set_id") or "").strip()
            if not set_id or set_id in insights:
                continue
            segment = cls._candidate_unstructured_segment(text, candidate)
            if not segment:
                continue
            parsed = cls._ai_insight_from_unstructured_text(segment, candidate)
            if parsed is None:
                continue
            insights[set_id] = AIInsight(
                score=int(parsed.score),
                summary=str(parsed.summary or "")[:1200] or "Output AI non JSON con parsing robusto.",
                predicted_eol_date=parsed.predicted_eol_date or candidate.get("eol_date_prediction"),
                fallback_used=False,
                confidence="LOW_CONFIDENCE",
                risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
            )

        if len(insights) == len(candidate_by_set_id):
            return insights

        # Pass 2: strict set_id -> score mapping in free-text.
        for candidate in candidates:
            set_id = str(candidate.get("set_id") or "").strip()
            if not set_id or set_id in insights:
                continue
            pattern = re.compile(
                rf"(?is){re.escape(set_id)}.{{0,80}}?(?:score|punteggio|rating|valutazione)?\s*[:=]?\s*([1-9]\d?|100)\b"
            )
            match = pattern.search(text)
            if not match:
                continue
            score = int(match.group(1))
            score = max(1, min(100, score))
            summary = (
                f"Output AI non JSON: score {score}/100 estratto da testo libero "
                f"per set {set_id} con parsing robusto."
            )
            insights[set_id] = AIInsight(
                score=score,
                summary=summary,
                predicted_eol_date=candidate.get("eol_date_prediction"),
                fallback_used=False,
                confidence="LOW_CONFIDENCE",
                risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
            )

        # Pass 3: deterministic order fallback only if cardinality is compatible.
        missing = [
            candidate
            for candidate in candidates
            if str(candidate.get("set_id") or "").strip()
            and str(candidate.get("set_id") or "").strip() not in insights
        ]
        if not missing:
            return insights

        lines_with_scores: list[str] = []
        for line in text.splitlines():
            line_text = str(line or "").strip()
            if not line_text:
                continue
            if cls._extract_unstructured_score(line_text) is None:
                continue
            lines_with_scores.append(line_text)

        if lines_with_scores:
            for idx, candidate in enumerate(missing[: len(lines_with_scores)]):
                segment = lines_with_scores[idx]
                parsed = cls._ai_insight_from_unstructured_text(segment, candidate)
                if parsed is None:
                    continue
                set_id = str(candidate.get("set_id") or "").strip()
                if not set_id:
                    continue
                insights[set_id] = AIInsight(
                    score=int(parsed.score),
                    summary=str(parsed.summary or "")[:1200] or "Output AI non JSON con parsing robusto.",
                    predicted_eol_date=parsed.predicted_eol_date or candidate.get("eol_date_prediction"),
                    fallback_used=False,
                    confidence="LOW_CONFIDENCE",
                    risk_note="Output AI non JSON: score estratto da testo con parsing robusto.",
                )

        return insights

    @staticmethod
    def _build_batch_ai_prompt(candidates: list[Dict[str, Any]]) -> str:
        lines = [
            "Analizza i seguenti set LEGO per investimento a 12 mesi.",
            "Rispondi SOLO con JSON valido.",
            'Formato obbligatorio: {"results":[{"set_id":"...", "score":1-100, "summary":"max 2 frasi", "predicted_eol_date":"YYYY-MM-DD o null"}]}',
            (
                "Se il modello non riesce a produrre JSON valido, usa formato fallback "
                "una riga per set: set_id=<ID>|score=<1-100>|summary=<max 12 parole>|predicted_eol_date=<YYYY-MM-DD|null>"
            ),
            f"Devi restituire esattamente {len(candidates)} risultati (uno per ogni set_id). Non omettere nessun set.",
            "Non aggiungere testo fuori dal JSON.",
            "",
            "SET LIST:",
        ]
        for idx, row in enumerate(candidates, start=1):
            lines.append(
                f"- idx={idx} | set_id={row.get('set_id')} | nome={row.get('set_name')} | tema={row.get('theme')} | "
                f"prezzo={row.get('current_price')} | eol_hint={row.get('eol_date_prediction')}"
            )
        lines.append("")
        lines.append(
            "Criteri: domanda collezionisti, brand power, rivalutazione attesa, liquidita', "
            "pattern di successo (licenza esclusiva, completismo serie, display value adulto, scarsita' EOL)."
        )
        return "\n".join(lines)

    @staticmethod
    def _format_exception_for_log(exc: Exception) -> tuple[str, str]:
        err_type = type(exc).__name__
        message = str(exc or "").strip()
        if not message:
            if isinstance(exc, asyncio.TimeoutError):
                message = "timeout (empty exception message)"
            else:
                message = repr(exc).strip() or "<no_error_message>"
        return err_type, message

    @staticmethod
    def _normalize_theme_key(raw_theme: Any) -> str:
        text = str(raw_theme or "").strip().lower()
        if not text:
            return ""
        text = text.replace("&", " and ").replace("/", " ")
        text = NON_ALNUM_RE.sub(" ", text)
        text = SPACE_RE.sub(" ", text).strip()
        return text

    @classmethod
    def _historical_theme_keys_for_candidate(cls, raw_theme: Any) -> list[str]:
        primary = cls._normalize_theme_key(raw_theme)
        if not primary:
            return []

        keys: list[str] = [primary]
        aliases = HISTORICAL_THEME_ALIASES.get(primary, ())
        for alias in aliases:
            normalized = cls._normalize_theme_key(alias)
            if normalized and normalized not in keys:
                keys.append(normalized)

        # Reverse lookup: if primary appears as alias, include that anchor key as fallback.
        for anchor_key, alias_list in HISTORICAL_THEME_ALIASES.items():
            normalized_anchor = cls._normalize_theme_key(anchor_key)
            normalized_aliases = {cls._normalize_theme_key(item) for item in alias_list}
            if primary in normalized_aliases and normalized_anchor and normalized_anchor not in keys:
                keys.append(normalized_anchor)

        return keys

    @staticmethod
    def _parse_pattern_tags(raw_value: Any) -> list[str]:
        raw_text = str(raw_value or "").strip()
        if not raw_text:
            return []
        try:
            loaded = json.loads(raw_text)
            if isinstance(loaded, list):
                normalized = [
                    str(item).strip().lower()
                    for item in loaded
                    if str(item or "").strip()
                ]
                if normalized:
                    return normalized
        except Exception:  # noqa: BLE001
            pass
        tokens = [part.strip().lower() for part in re.split(r"[|,;]", raw_text)]
        return [token for token in tokens if token]

    def _evaluate_historical_reference_quality(self, cases: list[Dict[str, Any]]) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "tier": "none",
            "degraded": False,
            "cases": len(cases),
            "theme_count": 0,
            "top_theme_share": 0.0,
            "general_tag_share": 0.0,
            "global_win_rate_pct": 0.0,
            "global_avg_roi_12m_pct": 0.0,
            "median_end_year": None,
            "latest_end_year": None,
            "median_age_years": None,
            "issues": [],
            "guards": {
                "max_median_age_years": int(self.historical_quality_max_median_age_years),
                "min_theme_count": int(self.historical_quality_min_theme_count),
                "max_top_theme_share": float(self.historical_quality_max_top_theme_share),
                "max_general_tag_share": float(self.historical_quality_max_general_tag_share),
            },
        }
        if not cases:
            profile["tier"] = "empty"
            profile["degraded"] = True
            profile["issues"] = ["seed_vuoto"]
            return profile

        theme_counter: Counter[str] = Counter()
        tag_counter: Counter[str] = Counter()
        years: list[int] = []
        for row in cases:
            theme_key = str(row.get("theme_norm") or "").strip()
            if theme_key:
                theme_counter[theme_key] += 1

            tags = row.get("pattern_tags_list")
            if not isinstance(tags, list):
                tags = self._parse_pattern_tags(row.get("pattern_tags"))
            if tags:
                tag_counter.update(str(tag).strip().lower() for tag in tags if str(tag).strip())

            end_text = str(row.get("end_date") or "").strip()
            if len(end_text) >= 4 and end_text[:4].isdigit():
                years.append(int(end_text[:4]))

        rois = [float(row.get("roi_12m_pct")) for row in cases if row.get("roi_12m_pct") is not None]
        wins = [int(row.get("win_12m")) for row in cases if row.get("win_12m") in (0, 1)]
        if rois:
            profile["global_avg_roi_12m_pct"] = round(float(statistics.fmean(rois)), 4)
        if wins:
            profile["global_win_rate_pct"] = round(float(statistics.fmean(wins) * 100.0), 4)

        total_cases = max(1, len(cases))
        theme_count = len(theme_counter)
        top_theme_count = theme_counter.most_common(1)[0][1] if theme_counter else 0
        top_theme_share = float(top_theme_count) / float(total_cases)
        general_tag_count = int(tag_counter.get("general_collectible", 0))
        general_tag_share = float(general_tag_count) / float(total_cases)

        profile["theme_count"] = int(theme_count)
        profile["top_theme_share"] = round(top_theme_share, 4)
        profile["general_tag_share"] = round(general_tag_share, 4)
        profile["top_theme"] = theme_counter.most_common(1)[0][0] if theme_counter else None

        if years:
            median_end_year = int(round(statistics.median(years)))
            latest_end_year = int(max(years))
            profile["median_end_year"] = median_end_year
            profile["latest_end_year"] = latest_end_year
            profile["median_age_years"] = int(max(0, date.today().year - median_end_year))
        else:
            profile["issues"].append("assenza_end_date")

        issues: list[str] = list(profile.get("issues") or [])
        if profile.get("median_age_years") is not None and int(profile["median_age_years"]) > int(
            self.historical_quality_max_median_age_years
        ):
            issues.append(
                f"seed_datato_mediana_{profile['median_age_years']}y>{int(self.historical_quality_max_median_age_years)}y"
            )
        if theme_count < int(self.historical_quality_min_theme_count):
            issues.append(
                f"copertura_temi_bassa_{theme_count}<{int(self.historical_quality_min_theme_count)}"
            )
        if top_theme_share > float(self.historical_quality_max_top_theme_share):
            issues.append(
                f"concentrazione_tema_alta_{top_theme_share:.2f}>{float(self.historical_quality_max_top_theme_share):.2f}"
            )
        if general_tag_share > float(self.historical_quality_max_general_tag_share):
            issues.append(
                f"pattern_generico_alto_{general_tag_share:.2f}>{float(self.historical_quality_max_general_tag_share):.2f}"
            )

        profile["issues"] = issues
        if not issues:
            profile["tier"] = "healthy"
            profile["degraded"] = False
        elif any(issue.startswith("seed_datato_") for issue in issues):
            profile["tier"] = "degraded"
            profile["degraded"] = True
        elif len(issues) >= 2:
            profile["tier"] = "warning"
            profile["degraded"] = True
        else:
            profile["tier"] = "warning"
            profile["degraded"] = False

        return profile

    @staticmethod
    def _normalize_market_country_code(value: Any) -> str:
        text = str(value or "").strip().upper()
        if len(text) == 2 and text.isalpha():
            return text
        return ""

    @classmethod
    def _normalize_market_region(cls, value: Any, *, market_country: str = "") -> str:
        raw = str(value or "").strip().upper()
        normalized = raw
        if raw in {"EUROPE", "EUROPA", "EU", "EEA", "EMEA"}:
            normalized = "EU"
        elif raw in {"WORLD", "GLOBAL", "INTL", "INTERNATIONAL"}:
            normalized = "GLOBAL"
        if not normalized and market_country in EUROPE_MARKET_COUNTRY_CODES:
            normalized = "EU"
        return normalized

    @classmethod
    def _infer_market_scope_from_source_dataset(cls, source_dataset: Any) -> tuple[str, str]:
        token = str(source_dataset or "").strip().lower()
        if not token:
            return "", ""

        if token.startswith("ebay_sold_"):
            match = re.match(r"^ebay_sold_([a-z]{2})", token)
            if match:
                country = match.group(1).upper()
                region = "EU" if country in EUROPE_MARKET_COUNTRY_CODES else ""
                return country, region

        # Legacy academic seeds do not carry explicit geo columns:
        # prefer EU-region inference instead of discarding all rows under IT/EU gating.
        if token.startswith("mendeley_"):
            return "", "EU"

        return "", ""

    def _historical_market_allowed(self, market_country: str, market_region: str) -> bool:
        country = self._normalize_market_country_code(market_country)
        region = self._normalize_market_region(market_region, market_country=country)
        if country and self.historical_allowed_countries and country in set(self.historical_allowed_countries):
            return True
        if region and self.historical_allowed_regions and region in set(self.historical_allowed_regions):
            return True
        if country or region:
            if self.historical_allowed_countries or self.historical_allowed_regions:
                return False
            return True
        return bool(self.historical_include_unknown_market)

    def _historical_case_weight(
        self,
        row: Dict[str, Any],
        *,
        now_utc: Optional[datetime] = None,
    ) -> float:
        now_dt = now_utc or datetime.now(timezone.utc)
        weight = 1.0

        base_weight_raw = row.get("case_weight")
        try:
            base_weight = float(base_weight_raw) if base_weight_raw not in (None, "") else 1.0
        except (TypeError, ValueError):
            base_weight = 1.0
        weight *= max(0.1, min(3.0, base_weight))

        explicit_recency_raw = row.get("recency_weight")
        try:
            explicit_recency = float(explicit_recency_raw) if explicit_recency_raw not in (None, "") else 1.0
        except (TypeError, ValueError):
            explicit_recency = 1.0
        weight *= max(0.1, min(2.0, explicit_recency))

        end_date_text = str(row.get("end_date") or "").strip()
        age_multiplier = float(self.historical_recency_min_weight)
        if end_date_text:
            try:
                end_dt = datetime.fromisoformat(end_date_text.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now_dt - end_dt).total_seconds() / 86400.0)
                half_life_days = max(30.0, float(self.historical_recency_halflife_days))
                decay = 2.0 ** (-age_days / half_life_days)
                age_multiplier = max(float(self.historical_recency_min_weight), min(1.0, decay))
            except ValueError:
                age_multiplier = float(self.historical_recency_min_weight)
        weight *= age_multiplier

        observation_months_raw = row.get("observation_months")
        try:
            observation_months = int(observation_months_raw) if observation_months_raw not in (None, "") else 0
        except (TypeError, ValueError):
            observation_months = 0
        if 0 < observation_months < 12:
            horizon_factor = max(0.55, min(1.0, float(observation_months) / 12.0))
            weight *= horizon_factor

        return max(0.05, min(3.0, float(weight)))

    @staticmethod
    def _weighted_mean(values: list[float], weights: list[float]) -> float:
        if not values:
            return 0.0
        if len(values) != len(weights):
            return float(statistics.fmean(values))
        positive_weights = [max(0.0, float(weight)) for weight in weights]
        total_weight = float(sum(positive_weights))
        if total_weight <= 0:
            return float(statistics.fmean(values))
        return float(sum(value * weight for value, weight in zip(values, positive_weights)) / total_weight)

    @classmethod
    def _weighted_stddev(cls, values: list[float], weights: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = cls._weighted_mean(values, weights)
        positive_weights = [max(0.0, float(weight)) for weight in weights]
        total_weight = float(sum(positive_weights))
        if total_weight <= 0:
            return float(statistics.pstdev(values))
        variance = sum(weight * ((value - mean) ** 2) for value, weight in zip(values, positive_weights)) / total_weight
        return float(math.sqrt(max(0.0, variance)))

    def _load_historical_reference_cases(self) -> list[Dict[str, Any]]:
        if not self.historical_reference_enabled:
            self._historical_market_filter_stats = {"enabled": False}
            return []

        paths = [Path(item) for item in self.historical_reference_paths if str(item).strip()]
        if not paths:
            self._historical_market_filter_stats = {"enabled": True, "paths": [], "rows_loaded": 0}
            return []

        cases: list[Dict[str, Any]] = []
        stats = {
            "enabled": True,
            "paths": [str(path) for path in paths],
            "rows_total": 0,
            "rows_loaded": 0,
            "rows_skipped_missing_id": 0,
            "rows_skipped_missing_roi": 0,
            "rows_skipped_market_scope": 0,
            "rows_inferred_market_scope": 0,
            "missing_paths": [],
        }

        for path in paths:
            if not path.exists():
                stats["missing_paths"].append(str(path))
                continue

            try:
                with path.open("r", encoding="utf-8", newline="") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        stats["rows_total"] += 1
                        set_id = str(row.get("set_id") or "").strip()
                        if not set_id:
                            stats["rows_skipped_missing_id"] += 1
                            continue
                        theme = str(row.get("theme") or "").strip() or "Unknown"
                        roi_12m_raw = row.get("roi_12m_pct")
                        try:
                            roi_12m = float(roi_12m_raw) if roi_12m_raw not in (None, "") else None
                        except (TypeError, ValueError):
                            roi_12m = None
                        if roi_12m is None:
                            stats["rows_skipped_missing_roi"] += 1
                            continue

                        market_country = self._normalize_market_country_code(
                            row.get("market_country") or row.get("country_code")
                        )
                        market_region = self._normalize_market_region(
                            row.get("market_region") or row.get("market_scope"),
                            market_country=market_country,
                        )
                        source_dataset = str(row.get("source_dataset") or "").strip()
                        if not market_country and not market_region:
                            inferred_country, inferred_region = self._infer_market_scope_from_source_dataset(source_dataset)
                            if inferred_country or inferred_region:
                                market_country = market_country or inferred_country
                                market_region = market_region or inferred_region
                                stats["rows_inferred_market_scope"] += 1
                        if not self._historical_market_allowed(market_country, market_region):
                            stats["rows_skipped_market_scope"] += 1
                            continue

                        msrp_raw = row.get("msrp_usd")
                        try:
                            msrp = float(msrp_raw) if msrp_raw not in (None, "") else None
                        except (TypeError, ValueError):
                            msrp = None

                        win_12m_raw = row.get("win_12m")
                        try:
                            win_12m = int(win_12m_raw) if win_12m_raw not in (None, "") else None
                        except (TypeError, ValueError):
                            win_12m = None
                        if win_12m is None:
                            win_12m = int(roi_12m >= float(self.target_roi_pct))

                        observation_months_raw = row.get("observation_months")
                        try:
                            observation_months = int(observation_months_raw) if observation_months_raw not in (None, "") else 0
                        except (TypeError, ValueError):
                            observation_months = 0

                        case: Dict[str, Any] = {
                            "set_id": set_id,
                            "theme": theme,
                            "theme_norm": self._normalize_theme_key(theme),
                            "set_name": str(row.get("set_name") or "").strip(),
                            "msrp_usd": msrp,
                            "roi_12m_pct": roi_12m,
                            "win_12m": int(win_12m),
                            "source_dataset": source_dataset,
                            "pattern_tags": str(row.get("pattern_tags") or "").strip(),
                            "pattern_tags_list": self._parse_pattern_tags(row.get("pattern_tags")),
                            "end_date": str(row.get("end_date") or "").strip(),
                            "observation_months": observation_months,
                            "market_country": market_country,
                            "market_region": market_region,
                        }

                        recency_weight_raw = row.get("recency_weight")
                        try:
                            recency_weight = (
                                float(recency_weight_raw)
                                if recency_weight_raw not in (None, "")
                                else None
                            )
                        except (TypeError, ValueError):
                            recency_weight = None
                        if recency_weight is not None:
                            case["recency_weight"] = recency_weight

                        case_weight_raw = row.get("case_weight")
                        try:
                            case_weight = (
                                float(case_weight_raw)
                                if case_weight_raw not in (None, "")
                                else None
                            )
                        except (TypeError, ValueError):
                            case_weight = None
                        if case_weight is not None:
                            case["case_weight"] = case_weight

                        case["resolved_weight"] = self._historical_case_weight(case)
                        cases.append(case)
                        stats["rows_loaded"] += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Historical reference cases load failed | path=%s error=%s", path, exc)

        if stats.get("missing_paths"):
            LOGGER.warning("Historical reference paths not found: %s", stats.get("missing_paths"))
        if not cases:
            LOGGER.warning(
                "Historical reference loaded 0 rows after filters | countries=%s regions=%s include_unknown=%s",
                ",".join(self.historical_allowed_countries) or "*",
                ",".join(self.historical_allowed_regions) or "*",
                self.historical_include_unknown_market,
            )

        self._historical_market_filter_stats = stats
        return cases

    @staticmethod
    def _clamp_score(value: float, *, minimum: float = 1.0, maximum: float = 100.0) -> float:
        return max(float(minimum), min(float(maximum), float(value)))

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        q = max(0.0, min(1.0, float(quantile)))
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return ordered[lower]
        weight = pos - lower
        return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)

    def _cohort_historical_strength(self, cases: list[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        scored_rows = [
            row
            for row in cases
            if row.get("roi_12m_pct") is not None
        ]
        if len(scored_rows) < 3:
            return None

        rois = [float(row["roi_12m_pct"]) for row in scored_rows]
        weights = [float(row.get("resolved_weight") or self._historical_case_weight(row)) for row in scored_rows]
        weighted_wins = []
        for row, roi in zip(scored_rows, rois):
            if row.get("win_12m") in (0, 1):
                weighted_wins.append(int(row.get("win_12m")))
            else:
                weighted_wins.append(1 if roi >= float(self.target_roi_pct) else 0)

        sample_size = len(rois)
        effective_sample_size = float(sum(max(0.0, weight) for weight in weights))
        avg_roi = self._weighted_mean(rois, weights)
        win_rate = self._weighted_mean([float(value) for value in weighted_wins], weights)
        roi_stdev = self._weighted_stddev(rois, weights)
        roi_component = self._clamp_score(50.0 + (avg_roi * 0.85))
        win_component = self._clamp_score(win_rate * 100.0)
        sample_bonus = min(10.0, math.log2(effective_sample_size + 1.0) * 2.0)
        prior_score = self._clamp_score((0.55 * win_component) + (0.45 * roi_component) + sample_bonus - 5.0)
        support_confidence = self._clamp_score(
            35.0 + (math.log2(effective_sample_size + 1.0) * 12.0) - min(25.0, roi_stdev * 0.25)
        )
        return {
            "sample_size": float(effective_sample_size),
            "raw_sample_size": float(sample_size),
            "win_rate_pct": float(win_rate * 100.0),
            "support_confidence": float(support_confidence),
            "prior_score": float(prior_score),
        }

    def _compute_adaptive_historical_thresholds(self) -> Dict[str, Any]:
        if not self.adaptive_historical_thresholds_enabled:
            return {"active": False, "reason": "disabled"}
        if not self._historical_reference_cases:
            return {"active": False, "reason": "no_reference_cases"}

        valid_cases = [row for row in self._historical_reference_cases if row.get("roi_12m_pct") is not None]
        min_cases = int(self.adaptive_historical_threshold_min_cases)
        if len(valid_cases) < min_cases:
            return {
                "active": False,
                "reason": "insufficient_cases",
                "cases_considered": len(valid_cases),
                "required_cases": min_cases,
            }

        by_theme: Dict[str, list[Dict[str, Any]]] = {}
        for row in valid_cases:
            theme_key = str(row.get("theme_norm") or "").strip()
            if not theme_key:
                continue
            by_theme.setdefault(theme_key, []).append(row)

        min_group_samples = max(8, int(self.historical_reference_min_samples // 2))
        theme_metrics: list[Dict[str, float]] = []
        for theme_key, theme_cases in by_theme.items():
            if len(theme_cases) < min_group_samples:
                continue
            cohort = self._cohort_historical_strength(theme_cases)
            if cohort is None:
                continue
            theme_metrics.append(cohort)

        min_themes = int(self.adaptive_historical_threshold_min_themes)
        if len(theme_metrics) < min_themes:
            return {
                "active": False,
                "reason": "insufficient_themes",
                "themes_considered": len(theme_metrics),
                "required_themes": min_themes,
                "cases_considered": len(valid_cases),
            }

        q = float(self.adaptive_historical_threshold_quantile)
        sample_values = [float(row["sample_size"]) for row in theme_metrics]
        win_values = [float(row["win_rate_pct"]) for row in theme_metrics]
        support_values = [float(row["support_confidence"]) for row in theme_metrics]
        prior_values = [float(row["prior_score"]) for row in theme_metrics]

        min_samples = int(
            round(
                min(
                    float(self.historical_high_conf_min_samples),
                    max(12.0, self._percentile(sample_values, q)),
                )
            )
        )
        min_win_rate_pct = float(
            min(
                float(self.historical_high_conf_min_win_rate_pct),
                max(45.0, self._percentile(win_values, q)),
            )
        )
        min_support_confidence = int(
            round(
                min(
                    float(self.historical_high_conf_min_support_confidence),
                    max(35.0, self._percentile(support_values, q)),
                )
            )
        )
        min_prior_score = int(
            round(
                min(
                    float(self.historical_high_conf_min_prior_score),
                    max(50.0, self._percentile(prior_values, q)),
                )
            )
        )

        return {
            "active": True,
            "quantile": q,
            "cases_considered": len(valid_cases),
            "themes_considered": len(theme_metrics),
            "min_samples": int(max(5, min_samples)),
            "min_win_rate_pct": round(float(min_win_rate_pct), 2),
            "min_support_confidence": int(max(1, min_support_confidence)),
            "min_prior_score": int(max(1, min_prior_score)),
            "static_min_samples": int(self.historical_high_conf_min_samples),
            "static_min_win_rate_pct": float(self.historical_high_conf_min_win_rate_pct),
            "static_min_support_confidence": int(self.historical_high_conf_min_support_confidence),
            "static_min_prior_score": int(self.historical_high_conf_min_prior_score),
        }

    def _effective_historical_high_confidence_thresholds(self) -> tuple[int, float, int, int, bool]:
        min_samples = int(self.historical_high_conf_min_samples)
        min_win_rate_pct = float(self.historical_high_conf_min_win_rate_pct)
        min_support_confidence = int(self.historical_high_conf_min_support_confidence)
        min_prior_score = int(self.historical_high_conf_min_prior_score)
        adaptive_active = False

        adaptive = self._adaptive_historical_thresholds if self.adaptive_historical_thresholds_enabled else {}
        if adaptive and adaptive.get("active"):
            min_samples = int(adaptive.get("min_samples") or min_samples)
            min_win_rate_pct = float(adaptive.get("min_win_rate_pct") or min_win_rate_pct)
            min_support_confidence = int(adaptive.get("min_support_confidence") or min_support_confidence)
            min_prior_score = int(adaptive.get("min_prior_score") or min_prior_score)
            adaptive_active = True

        quality_profile = self._historical_quality_profile or {}
        quality_tier = str(quality_profile.get("tier") or "").strip().lower()
        quality_degraded = bool(
            self.historical_quality_guard_enabled
            and self.historical_quality_soft_gate_enabled
            and quality_profile.get("degraded")
        )
        if quality_degraded:
            global_win_rate_pct = float(quality_profile.get("global_win_rate_pct") or 0.0)
            min_samples = min(
                min_samples,
                max(
                    int(self.historical_degraded_min_samples),
                    int(round(float(min_samples) * 0.5)),
                ),
            )

            if quality_tier == "empty" or not self.historical_degraded_gate_relax_enabled:
                # Empty seed has almost no usable signal quality: keep conservative softening.
                min_win_rate_pct = min(min_win_rate_pct, max(16.0, global_win_rate_pct + 8.0))
                min_support_confidence = min(min_support_confidence, max(45, int(self.historical_degraded_min_support_confidence)))
                min_prior_score = min(min_prior_score, max(50, int(self.historical_degraded_min_prior_score)))
            else:
                min_win_rate_pct = min(
                    min_win_rate_pct,
                    max(
                        float(self.historical_degraded_min_win_rate_pct),
                        global_win_rate_pct + float(self.historical_degraded_win_rate_margin_pct),
                    ),
                )
                min_support_confidence = min(min_support_confidence, int(self.historical_degraded_min_support_confidence))
                min_prior_score = min(min_prior_score, int(self.historical_degraded_min_prior_score))

        return (
            max(5, min_samples),
            max(1.0, min_win_rate_pct),
            max(1, min_support_confidence),
            max(1, min_prior_score),
            adaptive_active,
        )

    def _historical_prior_for_candidate(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._historical_reference_cases:
            return None

        theme = self._normalize_theme_key(candidate.get("theme"))
        if not theme:
            return None

        min_required_for_prior = max(8, int(self.historical_reference_min_samples // 2))
        direct_cases = [row for row in self._historical_reference_cases if row.get("theme_norm") == theme]
        match_mode = "direct"
        matched_theme_keys = [theme]
        theme_cases = direct_cases
        if len(theme_cases) < min_required_for_prior:
            candidate_keys = self._historical_theme_keys_for_candidate(theme)
            matched_theme_keys = candidate_keys or [theme]
            alias_cases = [
                row
                for row in self._historical_reference_cases
                if str(row.get("theme_norm") or "") in set(matched_theme_keys)
            ]
            if len(alias_cases) >= min_required_for_prior and len(matched_theme_keys) > 1:
                theme_cases = alias_cases
                match_mode = "alias"
            elif len(theme_cases) < min_required_for_prior:
                return None

        price_raw = candidate.get("current_price")
        try:
            current_price = float(price_raw) if price_raw is not None else None
        except (TypeError, ValueError):
            current_price = None

        selected = theme_cases
        if current_price is not None and current_price > 0:
            tolerance = float(self.historical_price_band_tolerance)
            low = current_price * max(0.05, 1.0 - tolerance)
            high = current_price * (1.0 + tolerance)
            price_band_cases = [
                row
                for row in theme_cases
                if row.get("msrp_usd") is not None and low <= float(row["msrp_usd"]) <= high
            ]
            if len(price_band_cases) >= int(self.historical_reference_min_samples):
                selected = price_band_cases

        scored_rows = [
            row
            for row in selected
            if row.get("roi_12m_pct") is not None
        ]
        rois = [float(row["roi_12m_pct"]) for row in scored_rows]
        if len(rois) < min_required_for_prior:
            return None

        weights = [float(row.get("resolved_weight") or self._historical_case_weight(row)) for row in scored_rows]
        weighted_wins = []
        for row, roi in zip(scored_rows, rois):
            if row.get("win_12m") in (0, 1):
                weighted_wins.append(int(row.get("win_12m")))
            else:
                weighted_wins.append(1 if roi >= float(self.target_roi_pct) else 0)

        sample_size = len(rois)
        effective_sample_size = float(sum(max(0.0, weight) for weight in weights))
        avg_roi = self._weighted_mean(rois, weights)
        median_roi = float(statistics.median(rois))
        win_rate = self._weighted_mean([float(value) for value in weighted_wins], weights)
        roi_stdev = self._weighted_stddev(rois, weights)

        roi_component = self._clamp_score(50.0 + (avg_roi * 0.85))
        win_component = self._clamp_score(win_rate * 100.0)
        sample_bonus = min(10.0, math.log2(effective_sample_size + 1.0) * 2.0)
        prior_score = self._clamp_score((0.55 * win_component) + (0.45 * roi_component) + sample_bonus - 5.0)
        support_confidence = self._clamp_score(
            35.0 + (math.log2(effective_sample_size + 1.0) * 12.0) - min(25.0, roi_stdev * 0.25)
        )
        source_counter = Counter(
            str(row.get("source_dataset") or "historical_reference_cases")
            for row in scored_rows
        )
        dominant_source = source_counter.most_common(1)[0][0] if source_counter else "historical_reference_cases"

        return {
            "theme": theme,
            "match_mode": match_mode,
            "matched_theme_keys": matched_theme_keys,
            "theme_case_count": len(theme_cases),
            "sample_size": sample_size,
            "effective_sample_size": round(effective_sample_size, 2),
            "avg_roi_12m_pct": round(avg_roi, 4),
            "median_roi_12m_pct": round(median_roi, 4),
            "win_rate_12m": round(win_rate, 4),
            "roi_stddev_12m_pct": round(roi_stdev, 4),
            "prior_score": int(round(prior_score)),
            "support_confidence": int(round(support_confidence)),
            "source": dominant_source,
        }

    def _load_history_by_set(self, set_ids: list[str]) -> Dict[str, list[Dict[str, Any]]]:
        grouped: Dict[str, list[Dict[str, Any]]] = {set_id: [] for set_id in set_ids}
        if not set_ids:
            return grouped

        try:
            rows = self.repository.get_market_history_for_sets(
                set_ids,
                days=self.history_window_days,
            )
            for row in rows:
                row_set_id = str(row.get("set_id") or "").strip()
                if not row_set_id:
                    continue
                grouped.setdefault(row_set_id, []).append(row)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Batch history fetch failed: %s", exc)

        missing = [set_id for set_id in set_ids if not grouped.get(set_id)]
        for set_id in missing:
            try:
                grouped[set_id] = self.repository.get_recent_market_prices(
                    set_id,
                    days=self.history_window_days,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("History fetch failed for %s: %s", set_id, exc)
                grouped[set_id] = []
        return grouped

    @staticmethod
    def _recent_rows_within_days(rows: list[Dict[str, Any]], *, days: int) -> list[Dict[str, Any]]:
        if not rows:
            return []
        now_utc = datetime.now(timezone.utc)
        limit_sec = float(days) * 24.0 * 60.0 * 60.0
        kept: list[Dict[str, Any]] = []
        for row in rows:
            raw = row.get("recorded_at")
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if (now_utc - parsed).total_seconds() <= limit_sec:
                kept.append(row)
        return kept

    def _ai_cache_key(self, candidate: Dict[str, Any]) -> Optional[str]:
        set_id = str(candidate.get("set_id") or "").strip()
        if not set_id:
            return None
        theme = str(candidate.get("theme") or "").strip().lower()
        source = str(candidate.get("source") or "").strip().lower()
        eol = str(candidate.get("eol_date_prediction") or "").strip()
        price_raw = candidate.get("current_price")
        try:
            price = f"{float(price_raw):.2f}"
        except (TypeError, ValueError):
            price = "na"
        return f"{set_id}|{theme}|{source}|{eol}|{price}"

    @staticmethod
    def _clone_ai_insight(insight: AIInsight) -> AIInsight:
        return AIInsight(
            score=int(insight.score),
            summary=str(insight.summary),
            predicted_eol_date=insight.predicted_eol_date,
            fallback_used=bool(insight.fallback_used),
            confidence=str(insight.confidence),
            risk_note=insight.risk_note,
            model_raw_score=(
                int(insight.model_raw_score)
                if insight.model_raw_score is not None
                else None
            ),
        )

    @staticmethod
    def _is_non_json_ai_note(note: Optional[str]) -> bool:
        lowered = str(note or "").strip().lower()
        if not lowered:
            return False
        return ("non json" in lowered) or ("parsing robusto" in lowered)

    @staticmethod
    def _merge_risk_note(existing: Optional[str], addition: str) -> str:
        base = str(existing or "").strip()
        if not base:
            return addition
        if addition.lower() in base.lower():
            return base
        return f"{base.rstrip('.')} | {addition}"

    def _normalize_ai_insight(self, insight: AIInsight, candidate: Dict[str, Any]) -> AIInsight:
        score = max(1, min(100, int(insight.score)))
        model_raw_score = (
            int(insight.model_raw_score)
            if insight.model_raw_score is not None
            else score
        )
        if bool(insight.fallback_used) or not self.ai_score_guardrail_enabled:
            return AIInsight(
                score=score,
                summary=str(insight.summary),
                predicted_eol_date=insight.predicted_eol_date or candidate.get("eol_date_prediction"),
                fallback_used=bool(insight.fallback_used),
                confidence=str(insight.confidence),
                risk_note=insight.risk_note,
                model_raw_score=(
                    model_raw_score
                    if insight.model_raw_score is not None
                    else None
                ),
            )

        adjusted = float(score)
        guardrails: list[str] = []

        if adjusted > float(self.ai_score_soft_cap):
            adjusted = float(self.ai_score_soft_cap) + (
                (adjusted - float(self.ai_score_soft_cap)) * float(self.ai_score_soft_cap_factor)
            )
            guardrails.append(f"soft-cap {self.ai_score_soft_cap}")

        low_confidence = str(insight.confidence or "").strip().upper() != "HIGH_CONFIDENCE"
        if low_confidence and adjusted > float(self.ai_low_confidence_score_cap):
            adjusted = float(self.ai_low_confidence_score_cap)
            guardrails.append(f"low-conf cap {self.ai_low_confidence_score_cap}")

        non_json_output = self._is_non_json_ai_note(insight.risk_note)
        if non_json_output and adjusted > float(self.ai_non_json_score_cap):
            adjusted = float(self.ai_non_json_score_cap)
            guardrails.append(f"non-json cap {self.ai_non_json_score_cap}")

        final_score = max(1, min(100, int(round(adjusted))))
        risk_note = insight.risk_note
        if final_score != score and guardrails:
            risk_note = self._merge_risk_note(
                risk_note,
                "AI guardrail applicato "
                f"({score}->{final_score}; {', '.join(guardrails)}).",
            )

        return AIInsight(
            score=final_score,
            summary=str(insight.summary),
            predicted_eol_date=insight.predicted_eol_date or candidate.get("eol_date_prediction"),
            fallback_used=bool(insight.fallback_used),
            confidence=str(insight.confidence),
            risk_note=risk_note,
            model_raw_score=(model_raw_score if final_score != model_raw_score else None),
        )

    def _prime_ai_cache_from_repository(self, candidates: list[Dict[str, Any]]) -> int:
        if self.ai_cache_ttl_sec <= 0:
            return 0
        if self.ai_persisted_cache_ttl_sec <= 0:
            return 0
        if not candidates:
            return 0

        now_ts = time.time()
        candidate_by_set: Dict[str, Dict[str, Any]] = {}
        fetch_set_ids: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            set_id = str(candidate.get("set_id") or "").strip()
            if not set_id or set_id in seen:
                continue
            seen.add(set_id)
            candidate_by_set[set_id] = candidate
            if self._get_cached_ai_insight(candidate) is None:
                fetch_set_ids.append(set_id)

        if not fetch_set_ids:
            return 0

        try:
            cached_rows = self.repository.get_recent_ai_insights(
                fetch_set_ids,
                max_age_hours=max(1.0, self.ai_persisted_cache_ttl_sec / 3600.0),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Persisted AI cache preload failed: %s", exc)
            return 0

        primed = 0
        for set_id, row in cached_rows.items():
            candidate = candidate_by_set.get(str(set_id))
            if candidate is None:
                continue
            insight = self._insight_from_persisted_row(row, candidate)
            if insight is None:
                continue
            insight = self._normalize_ai_insight(insight, candidate)
            key = self._ai_cache_key(candidate)
            if not key:
                continue
            self._ai_insight_cache[key] = (
                now_ts + min(self.ai_cache_ttl_sec, self.ai_persisted_cache_ttl_sec),
                self._clone_ai_insight(insight),
            )
            primed += 1
        return primed

    @staticmethod
    def _insight_from_persisted_row(row: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[AIInsight]:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if bool(metadata.get("ai_fallback_used")):
            return None

        raw_score = metadata.get("ai_raw_score")
        if raw_score is None:
            raw_score = row.get("ai_investment_score")
        try:
            score = int(float(raw_score))
        except (TypeError, ValueError):
            return None
        score = max(1, min(100, score))

        summary = str(row.get("ai_analysis_summary") or "").strip()
        if not summary:
            summary = "Score AI riusato da storico recente."
        predicted = (
            row.get("eol_date_prediction")
            or metadata.get("predicted_eol_date")
            or candidate.get("eol_date_prediction")
        )
        confidence = str(metadata.get("ai_confidence") or "HIGH_CONFIDENCE")
        model_raw_score = metadata.get("ai_model_raw_score")
        try:
            parsed_model_raw = int(float(model_raw_score)) if model_raw_score is not None else None
        except (TypeError, ValueError):
            parsed_model_raw = None
        return AIInsight(
            score=score,
            summary=summary,
            predicted_eol_date=predicted,
            fallback_used=False,
            confidence=confidence,
            risk_note=None,
            model_raw_score=(parsed_model_raw if parsed_model_raw != score else None),
        )

    def _get_cached_ai_insight(self, candidate: Dict[str, Any]) -> Optional[AIInsight]:
        if self.ai_cache_ttl_sec <= 0:
            return None
        key = self._ai_cache_key(candidate)
        if not key:
            return None
        row = self._ai_insight_cache.get(key)
        if row is None:
            return None
        expires_at, cached = row
        now_ts = time.time()
        if expires_at <= now_ts:
            self._ai_insight_cache.pop(key, None)
            return None
        return self._normalize_ai_insight(self._clone_ai_insight(cached), candidate)

    def _set_cached_ai_insight(self, candidate: Dict[str, Any], insight: AIInsight) -> None:
        if self.ai_cache_ttl_sec <= 0 or insight.fallback_used:
            return
        key = self._ai_cache_key(candidate)
        if not key:
            return
        now_ts = time.time()
        if len(self._ai_insight_cache) >= self.ai_cache_max_items:
            expired = [
                cache_key
                for cache_key, (expires_at, _cached) in self._ai_insight_cache.items()
                if expires_at <= now_ts
            ]
            for cache_key in expired:
                self._ai_insight_cache.pop(cache_key, None)
            while len(self._ai_insight_cache) >= self.ai_cache_max_items and self._ai_insight_cache:
                oldest_key = next(iter(self._ai_insight_cache))
                self._ai_insight_cache.pop(oldest_key, None)

        self._ai_insight_cache[key] = (
            now_ts + self.ai_cache_ttl_sec,
            self._clone_ai_insight(insight),
        )

    @staticmethod
    def _is_ai_score_collapse(ranked: list[Dict[str, Any]]) -> bool:
        if len(ranked) < 6:
            return False

        scores = [int(row.get("ai_raw_score") or row.get("ai_investment_score") or 0) for row in ranked]
        spread = max(scores) - min(scores)
        if spread > 8:
            return False

        mean = sum(scores) / len(scores)
        if mean < 42 or mean > 58:
            return False

        dominant = Counter(scores).most_common(1)[0][1]
        return dominant / len(scores) >= 0.6

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
            amazon_reader_url = (
                "https://r.jina.ai/http://https://www.amazon.it/"
                "s?i=toys&k=lego+set&rh=p_89%3ALEGO"
            )
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
            signals["amazon_proxy_lego_keyword_count"] = len(re.findall(r"\blego\b", amazon_md, re.IGNORECASE))
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
            price = cls._extract_price_from_text(inner)
            theme = cls._guess_theme_from_name(name)
            rows.append(
                {
                    "set_id": set_id,
                    "set_name": name,
                    "theme": theme,
                    "source": "lego_http_fallback",
                    "current_price": price,
                    "eol_date_prediction": cls._estimate_eol_prediction(
                        source="lego_http_fallback",
                        set_id=set_id,
                        theme=theme,
                        price=price,
                        context_text=inner,
                    ),
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
            theme = cls._guess_theme_from_name(name)
            rows.append(
                {
                    "set_id": set_id,
                    "set_name": name,
                    "theme": theme,
                    "source": "lego_proxy_reader",
                    "current_price": price,
                    "eol_date_prediction": cls._estimate_eol_prediction(
                        source="lego_proxy_reader",
                        set_id=set_id,
                        theme=theme,
                        price=price,
                        context_text=snippet,
                    ),
                    "listing_url": url,
                    "metadata": {"proxy_reader": True},
                }
            )
            if len(rows) >= limit:
                break
        return rows

    @classmethod
    def _parse_amazon_proxy_markdown(cls, markdown_text: str, *, limit: int) -> list[Dict[str, Any]]:
        rows_by_set_id: Dict[str, Dict[str, Any]] = {}
        dp_link_re = re.compile(
            r"\((https?://www\.amazon\.[^\s)]+/(?:dp|gp/product)/[^\s)]+)\)",
            re.IGNORECASE,
        )
        matches = list(dp_link_re.finditer(markdown_text))

        for idx, match in enumerate(matches):
            url = match.group(1)
            canonical_url = cls._canonical_amazon_product_url(url)
            lowered_url = url.lower()
            if "amazon." not in lowered_url:
                continue
            if "/dp/" not in lowered_url and "/gp/product/" not in lowered_url:
                continue

            context_start = max(0, match.start() - 1800)
            context_end = matches[idx + 1].start() if idx + 1 < len(matches) else min(
                len(markdown_text), match.end() + 500
            )
            snippet = markdown_text[context_start:context_end]
            price_window_start = max(0, match.start() - 80)
            price_window_end = min(len(markdown_text), match.end() + 700)
            price_snippet = markdown_text[price_window_start:price_window_end]
            name = cls._extract_amazon_proxy_name(snippet, url)
            is_lego = any(
                "lego" in (text or "").lower()
                for text in (name, snippet, url)
            )
            if not is_lego:
                continue

            set_id = cls._extract_amazon_proxy_set_id(name, snippet)
            if not set_id:
                continue

            current_price = cls._extract_price_from_text(price_snippet)
            row = {
                "set_id": set_id,
                "set_name": name,
                "theme": cls._guess_theme_from_name(name),
                "source": "amazon_proxy_reader",
                "current_price": current_price,
                "eol_date_prediction": None,
                "listing_url": canonical_url,
                "metadata": {"proxy_reader": True},
            }
            existing = rows_by_set_id.get(set_id)
            if existing is None:
                rows_by_set_id[set_id] = row
            else:
                existing_price = existing.get("current_price")
                if existing_price is None and current_price is not None:
                    rows_by_set_id[set_id] = row
                elif (
                    current_price is not None
                    and isinstance(existing_price, (int, float))
                    and current_price < float(existing_price)
                ):
                    rows_by_set_id[set_id] = row

            if len(rows_by_set_id) >= limit:
                break
        return list(rows_by_set_id.values())

    @staticmethod
    def _extract_amazon_proxy_set_id(name: str, snippet: str) -> Optional[str]:
        title_match = re.search(r"\b(\d{4,5})\b", name or "")
        if title_match:
            return title_match.group(1)

        for pattern in (
            r"-\s*(\d{4,5})\s*-{2,}",
            r"\b(?:set|model|kit)\s*(\d{4,5})\b",
        ):
            match = re.search(pattern, snippet or "", re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _canonical_amazon_product_url(url: str) -> str:
        parsed = urlparse(url)
        asin_match = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", parsed.path, re.IGNORECASE)
        if asin_match:
            return f"{parsed.scheme}://{parsed.netloc}/dp/{asin_match.group(1)}"
        return url.split("?", 1)[0]

    @classmethod
    def _extract_amazon_proxy_name(cls, snippet: str, url: str) -> str:
        candidate_re = re.compile(r"\[([^\]\n]{8,360})\]")
        candidates: list[tuple[int, str]] = []

        for raw in candidate_re.findall(snippet):
            candidate = cls._cleanup_html_text(raw)
            if not candidate:
                continue
            if not re.search(r"[A-Za-z]", candidate):
                continue
            if cls._is_noise_amazon_candidate(candidate):
                continue

            lowered = candidate.lower()
            score = 0
            if "lego" in lowered:
                score += 4
            if re.search(r"\b\d{4,6}\b", candidate):
                score += 3
            if len(candidate) >= 24:
                score += 1
            candidates.append((score, candidate))

        if candidates:
            candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
            return candidates[0][1]

        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        slug = ""
        for idx, part in enumerate(parts):
            lowered = part.lower()
            if lowered in {"dp", "gp"} and idx > 0:
                slug = parts[idx - 1]
                if slug.lower() in {"en", "it", "-", "product"} and idx > 1:
                    slug = parts[idx - 2]
                break
        fallback = cls._cleanup_html_text(slug.replace("-", " "))
        return fallback or "LEGO product"

    @staticmethod
    def _is_noise_amazon_candidate(text: str) -> bool:
        lowered = (text or "").strip().lower()
        if len(lowered) < 8:
            return True
        if lowered.startswith(("![image ", "image ", "price, product page", "see options", "learn about these results")):
            return True
        if lowered.startswith(("cookies and advertising choices", "add to basket", "best seller")):
            return True
        if any(token in lowered for token in ("rrp:", "new offers", "used & new offers", "delivery ", "bought in past month")):
            return True
        if "out of 5 stars" in lowered:
            return True
        if re.fullmatch(r"[\d\s.,%+()kKmM]+", lowered):
            return True
        return False

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
        keyword_map = [
            ("Star Wars", ("star wars", "guerre stellari")),
            (
                "Technic",
                (
                    "technic",
                    "ingranaggi",
                    "escavatore",
                    "gru",
                    "liebherr",
                    "rover lunare",
                    "lrv",
                    "apollo",
                    "nasa",
                    "xlt",
                ),
            ),
            ("City", ("city", "citta", "polizia", "vigili del fuoco", "ambulanza")),
            ("Icons", ("icons", "creator expert", "modular", "medieval", "castello")),
            ("Botanicals", ("botanical", "botanicals", "narcisi", "fiori", "bouquet", "rose", "orchidea")),
            ("Harry Potter", ("harry potter", "hogwarts")),
            (
                "Ideas",
                (
                    "ideas",
                    "hocus pocus",
                    "sanderson",
                    "typewriter",
                    "tree house",
                    "home alone",
                ),
            ),
            (
                "Marvel",
                (
                    "marvel",
                    "avengers",
                    "spider-man",
                    "spiderman",
                    "x-men",
                    "x jet",
                    "x-jet",
                    "goblin",
                    "iron man",
                    "captain america",
                    "thor",
                    "hulk",
                    "venom",
                ),
            ),
            ("Ninjago", ("ninjago",)),
            ("Friends", ("friends",)),
            ("Architecture", ("architecture",)),
            (
                "Seasonal",
                (
                    "parco giochi degli animali",
                    "animal playground",
                    "pasqua",
                    "easter",
                    "spring",
                    "natale",
                    "halloween",
                ),
            ),
            (
                "Animal Crossing",
                (
                    "animal crossing",
                    "dodo airlines",
                    "isabelle",
                    "tom nook",
                    "k.k.",
                    "blatero",
                    "fuffi",
                ),
            ),
        ]
        for theme, keywords in keyword_map:
            if any(keyword in lowered for keyword in keywords):
                return theme
        return "Unknown"

    async def _get_openrouter_insight_opportunistic(
        self,
        *,
        candidate: Dict[str, Any],
        prompt: str,
    ) -> Optional[AIInsight]:
        if self._openrouter_model_id is None:
            return None

        set_id = candidate.get("set_id")
        attempts = 1
        if self.openrouter_opportunistic_enabled:
            attempts = max(1, int(self.openrouter_opportunistic_attempts))
        timeout_sec = min(self.ai_generation_timeout_sec, self.openrouter_opportunistic_timeout_sec)
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            current_model = str(self._openrouter_model_id or "")
            if not current_model:
                break

            try:
                text = await asyncio.to_thread(
                    self._openrouter_generate,
                    prompt,
                    request_timeout=timeout_sec,
                )
                try:
                    payload = self._extract_json(text)
                except Exception:
                    repaired_insight = await self._repair_openrouter_non_json_output(
                        raw_text=text,
                        candidate=candidate,
                        timeout_sec=timeout_sec,
                    )
                    if repaired_insight is not None:
                        self._openrouter_malformed_errors = 0
                        if attempt > 1:
                            LOGGER.info(
                                "OpenRouter opportunistic recovery via JSON repair | set_id=%s attempt=%s model=%s",
                                set_id,
                                attempt,
                                current_model,
                            )
                        return repaired_insight
                    parsed_text_insight = self._ai_insight_from_unstructured_text(text, candidate)
                    if parsed_text_insight is not None:
                        self._openrouter_malformed_errors = 0
                        self._record_model_success("openrouter", current_model, phase="scoring_text_parse")
                        LOGGER.warning(
                            "OpenRouter non-JSON output parsed | model=%s set_id=%s",
                            current_model,
                            set_id,
                        )
                        if attempt > 1:
                            LOGGER.info(
                                "OpenRouter opportunistic recovery | set_id=%s attempt=%s model=%s",
                                set_id,
                                attempt,
                                current_model,
                            )
                        return parsed_text_insight
                    raise
                self._openrouter_malformed_errors = 0
                insight = self._payload_to_ai_insight(payload, candidate)
                self._record_model_success("openrouter", current_model, phase="scoring")
                if attempt > 1:
                    LOGGER.info(
                        "OpenRouter opportunistic recovery | set_id=%s attempt=%s model=%s",
                        set_id,
                        attempt,
                        current_model,
                    )
                return insight
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                malformed = self._is_openrouter_malformed_response_error(exc)
                rate_limited = self._is_openrouter_rate_limited_error(exc)
                self._record_model_failure("openrouter", current_model, str(exc), phase="scoring")
                if malformed and self._register_openrouter_malformed_failure(set_id=set_id, reason=str(exc)):
                    return None

                should_rotate = self._should_rotate_openrouter_model(exc) or self._is_model_temporarily_banned(
                    "openrouter",
                    current_model,
                )
                can_retry = attempt < attempts
                if should_rotate and can_retry:
                    rotated = await self._advance_openrouter_model_locked(
                        reason=f"opportunistic_attempt_{attempt}:{exc}",
                    )
                    if rotated:
                        continue
                    if rate_limited:
                        backoff = min(1.4, 0.4 * float(attempt))
                        LOGGER.warning(
                            "OpenRouter rate-limited, retrying same model | set_id=%s model=%s attempt=%s/%s sleep=%.1fs",
                            set_id,
                            current_model,
                            attempt,
                            attempts,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue

                if should_rotate:
                    if not malformed and not rate_limited:
                        LOGGER.warning(
                            "OpenRouter non-rate error without alternative model | set_id=%s model=%s attempts=%s. Using fallback only for this candidate.",
                            set_id,
                            current_model,
                            attempts,
                        )
                    elif rate_limited:
                        LOGGER.warning(
                            "OpenRouter rate-limited without alternative model | set_id=%s model=%s attempts=%s. Using heuristic fallback for this candidate only.",
                            set_id,
                            current_model,
                            attempts,
                        )
                    else:
                        LOGGER.warning(
                            "OpenRouter scoring malformed for %s without failover; using fallback insight.",
                            set_id,
                        )
                else:
                    LOGGER.warning("OpenRouter scoring failed for %s: %s", set_id, exc)
                break

        if last_exc is not None and attempts > 1:
            LOGGER.warning(
                "OpenRouter opportunistic exhausted | set_id=%s attempts=%s timeout_sec=%.1f last_error=%s",
                set_id,
                attempts,
                timeout_sec,
                str(last_exc)[:220],
            )
        return None

    async def _get_ai_insight(self, candidate: Dict[str, Any]) -> AIInsight:
        prompt = self._build_gemini_prompt(candidate)
        if self._model is not None:
            current_gemini_model = str(self.gemini_model or "")
            try:
                text = await asyncio.to_thread(self._gemini_generate, prompt)
                payload = self._extract_json(text)
                insight = self._payload_to_ai_insight(payload, candidate)
                if current_gemini_model:
                    self._record_model_success("gemini", current_gemini_model, phase="scoring")
                return self._normalize_ai_insight(insight, candidate)
            except Exception as exc:  # noqa: BLE001
                if current_gemini_model:
                    self._record_model_failure("gemini", current_gemini_model, str(exc), phase="scoring")
                rotated = False
                should_rotate = self._should_rotate_gemini_model(exc) or (
                    bool(current_gemini_model) and self._is_model_temporarily_banned("gemini", current_gemini_model)
                )
                if should_rotate:
                    rotated = await self._advance_gemini_model_locked(reason=str(exc))
                if rotated:
                    next_model = str(self.gemini_model or "")
                    try:
                        text = await asyncio.to_thread(self._gemini_generate, prompt)
                        payload = self._extract_json(text)
                        insight = self._payload_to_ai_insight(payload, candidate)
                        if next_model:
                            self._record_model_success("gemini", next_model, phase="scoring_after_failover")
                        return self._normalize_ai_insight(insight, candidate)
                    except Exception as exc_after_switch:  # noqa: BLE001
                        if next_model:
                            self._record_model_failure(
                                "gemini",
                                next_model,
                                str(exc_after_switch),
                                phase="scoring_after_failover",
                            )
                        self._disable_gemini("fallback_after_gemini_error", str(exc_after_switch))
                        LOGGER.warning(
                            "Gemini scoring failed after failover for %s: %s",
                            candidate.get("set_id"),
                            exc_after_switch,
                        )
                elif should_rotate:
                    self._disable_gemini("fallback_after_gemini_error", str(exc))
                else:
                    LOGGER.warning("Gemini scoring failed for %s: %s", candidate.get("set_id"), exc)

        if self._openrouter_model_id is None and self.openrouter_api_key and not self._openrouter_recovery_attempted:
            self._openrouter_recovery_attempted = True
            LOGGER.info(
                "OpenRouter recovery init attempt | mode=%s reason=%s",
                self.ai_runtime.get("mode"),
                self.ai_runtime.get("reason"),
            )
            self._initialize_openrouter_runtime()

        if self._openrouter_model_id is not None:
            insight = await self._get_openrouter_insight_opportunistic(
                candidate=candidate,
                prompt=prompt,
            )
            if insight is not None:
                return self._normalize_ai_insight(insight, candidate)

        return self._heuristic_ai_fallback(candidate)

    @staticmethod
    def _validate_ai_payload(payload: Dict[str, Any], candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("invalid ai payload: not a JSON object")

        if "score" not in payload:
            raise ValueError("invalid ai payload: missing score")
        raw_score = payload.get("score")
        try:
            score = int(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid ai payload: score is not integer") from exc
        if score < 1 or score > 100:
            raise ValueError("invalid ai payload: score out of range")

        summary = payload.get("summary")
        if summary is None or not str(summary).strip():
            raise ValueError("invalid ai payload: missing summary")

        predicted = payload.get("predicted_eol_date")
        if predicted not in (None, "", "null"):
            text = str(predicted).strip()
            try:
                date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError("invalid ai payload: predicted_eol_date not ISO date") from exc

        return payload

    @classmethod
    def _payload_to_ai_insight(cls, payload: Dict[str, Any], candidate: Dict[str, Any]) -> AIInsight:
        valid_payload = cls._validate_ai_payload(payload, candidate)
        score = int(valid_payload.get("score", 50))
        score = max(1, min(100, score))
        summary = str(valid_payload.get("summary") or "No summary")[:1200]
        predicted_eol_date = valid_payload.get("predicted_eol_date") or candidate.get("eol_date_prediction")
        return AIInsight(
            score=score,
            summary=summary,
            predicted_eol_date=predicted_eol_date,
            fallback_used=False,
            confidence="HIGH_CONFIDENCE",
            risk_note=None,
        )

    def _gemini_generate(self, prompt: str) -> str:
        if self._model is None:
            return "{}"

        generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 220,
            "response_mime_type": "application/json",
        }
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": self.ai_generation_timeout_sec},
            )
        except TypeError:
            response = self._model.generate_content(
                prompt,
                generation_config=generation_config,
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
            "Criteri: domanda collezionisti, brand power, probabilita' rivalutazione, velocita' di rotazione, "
            "pattern di successo (licenza esclusiva, completismo serie, display value adulto, scarsita' EOL)."
        )

    def _heuristic_ai_fallback(self, candidate: Dict[str, Any]) -> AIInsight:
        set_id = str(candidate.get("set_id") or "")
        name = str(candidate.get("set_name") or "")
        source = str(candidate.get("source") or "")
        price = float(candidate.get("current_price") or 0.0)

        if self.ai_runtime.get("engine") not in {"gemini", "openrouter"}:
            self.ai_runtime.setdefault("engine", "local_ai")
            self.ai_runtime.setdefault("provider", "local")
            self.ai_runtime.setdefault("model", "local-quant-ai-v1")
            self.ai_runtime.setdefault("mode", "local_ai_fallback")

        base = 55
        if source in {"lego_retiring", "lego_proxy_reader", "lego_http_fallback"}:
            base += 18
        if any(key in name.lower() for key in ("star wars", "icons", "technic", "modular")):
            base += 12
        if any(key in name.lower() for key in ("marvel", "x-men", "harry potter", "ninjago", "speed champions")):
            base += 8
        if 30 <= price <= 180:
            base += 6
        elif price > 500:
            base -= 4

        # Local AI surrogate: deterministic per set_id to avoid flat scores in fallback mode.
        id_hash_nudge = ((sum(ord(ch) for ch in set_id) % 9) - 4) if set_id else 0

        score = max(1, min(100, base + id_hash_nudge))
        eol = candidate.get("eol_date_prediction") or (date.today() + timedelta(days=80)).isoformat()
        return AIInsight(
            score=score,
            summary=(
                "Local AI fallback (cloud-safe): scoring quantitativo su fonte, tema, fascia prezzo "
                "e variabilita storica; confermare con storico prezzi prima dell'acquisto definitivo."
            ),
            predicted_eol_date=eol,
            fallback_used=True,
            confidence="LOW_CONFIDENCE",
            risk_note="Ranking calcolato con Local AI fallback (provider esterno temporaneamente non disponibile).",
        )

    @staticmethod
    def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
        lowered = str(text or "").lower()
        return any(keyword in lowered for keyword in keywords if keyword)

    @classmethod
    def _infer_success_pattern_features(cls, candidate: Dict[str, Any]) -> Dict[str, Any]:
        name = str(candidate.get("set_name") or "")
        theme = str(candidate.get("theme") or "Unknown")
        source = str(candidate.get("source") or "unknown")
        listing_url = str(candidate.get("listing_url") or "")
        metadata = candidate.get("metadata")
        raw_blob = ""
        if isinstance(metadata, dict):
            raw_blob = str(metadata.get("raw_blob") or metadata.get("description") or "")

        text_blob = " ".join((name, theme, listing_url, raw_blob))
        text_lower = text_blob.lower()

        franchise = "Unknown"
        if cls._contains_any_keyword(text_lower, CULT_MOVIE_KEYWORDS):
            franchise = "Cult Movie"
        elif cls._contains_any_keyword(text_lower, BLOCKBUSTER_FRANCHISE_KEYWORDS):
            franchise = "Franchise"

        minifigure_is_exclusive = cls._contains_any_keyword(text_lower, EXCLUSIVE_MINIFIGURE_KEYWORDS)
        set_type = "Collection/Series" if cls._contains_any_keyword(text_lower, SERIES_COLLECTION_KEYWORDS) else "Standalone"
        recommended_age = "18+" if cls._contains_any_keyword(text_lower, ("18+", "for adults", "adults")) else "n/d"
        category = "Art/Ideas" if theme in ART_DISPLAY_THEMES or cls._contains_any_keyword(text_lower, ADULT_DISPLAY_KEYWORDS) else theme

        price_value: Optional[float]
        try:
            price_value = float(candidate.get("current_price"))
        except (TypeError, ValueError):
            price_value = None

        eol_days: Optional[int] = None
        raw_eol = str(candidate.get("eol_date_prediction") or "").strip()
        parsed_eol = cls._extract_first_date(raw_eol) if raw_eol else None
        if parsed_eol:
            try:
                eol_days = (date.fromisoformat(parsed_eol) - date.today()).days
            except ValueError:
                eol_days = None

        return {
            "franchise": franchise,
            "minifigure_is_exclusive": minifigure_is_exclusive,
            "set_type": set_type,
            "recommended_age": recommended_age,
            "category": category,
            "theme": theme,
            "source": source,
            "price": price_value,
            "eol_days": eol_days,
            "is_display_theme": theme in ART_DISPLAY_THEMES,
            "is_vehicle_icon": cls._contains_any_keyword(text_lower, VEHICLE_NOSTALGIA_KEYWORDS),
            "is_modular_family": cls._contains_any_keyword(text_lower, MODULAR_COMPLETIST_KEYWORDS),
            "is_scarcity_flagged": cls._contains_any_keyword(text_lower, SCARCITY_KEYWORDS),
        }

    @classmethod
    def _evaluate_success_patterns(cls, candidate: Dict[str, Any]) -> PatternEvaluation:
        features = cls._infer_success_pattern_features(candidate)
        signals: list[Dict[str, Any]] = []

        def add_signal(code: str, label: str, score: int, confidence: float, rationale: str) -> None:
            signals.append(
                {
                    "code": code,
                    "label": label,
                    "score": int(max(1, min(100, score))),
                    "confidence": round(max(0.05, min(1.0, confidence)), 2),
                    "rationale": rationale,
                }
            )

        franchise = str(features.get("franchise") or "Unknown")
        set_type = str(features.get("set_type") or "Standalone")
        recommended_age = str(features.get("recommended_age") or "n/d")
        category = str(features.get("category") or "")
        theme = str(features.get("theme") or "Unknown")
        source = str(features.get("source") or "unknown")
        price = features.get("price")
        eol_days = features.get("eol_days")

        if bool(features.get("minifigure_is_exclusive")) and franchise == "Cult Movie":
            add_signal(
                "exclusive_cult_license",
                "Licenza esclusiva cult",
                95,
                0.88,
                "Minifigure/personaggio esclusivo legato a franchise cinematografico cult.",
            )

        if set_type == "Collection/Series":
            add_signal(
                "series_completism",
                "Completismo di serie",
                85,
                0.84,
                "Set parte di linea collezionabile: pressione FOMO sui collezionisti.",
            )

        if recommended_age == "18+" and category == "Art/Ideas":
            add_signal(
                "adult_display_value",
                "Display value adulto",
                90,
                0.86,
                "Target adulto/decorativo: domanda meno ciclica e piu' da esposizione.",
            )

        if franchise == "Cult Movie" and bool(features.get("is_vehicle_icon")):
            add_signal(
                "nostalgia_vehicle",
                "Icona veicolo nostalgia",
                88,
                0.78,
                "Veicolo iconico da franchise cult: forte componente emozionale.",
            )

        if (
            recommended_age == "18+"
            and (
                cls._contains_any_keyword(str(candidate.get("set_name") or ""), FLAGSHIP_COLLECTOR_KEYWORDS)
                or (isinstance(price, float) and price >= 180.0)
            )
            and theme in {"Star Wars", "Technic", "Icons", "Architecture"}
        ):
            add_signal(
                "flagship_collector",
                "Flagship da collezione",
                86,
                0.8,
                "Set premium/flagship con bassa sostituibilita' percepita.",
            )

        if source in LEGO_PRIMARY_SOURCES and isinstance(eol_days, int) and 0 <= eol_days <= 210:
            add_signal(
                "retiring_window",
                "Catalizzatore EOL",
                83,
                0.74,
                "Finestra di ritiro vicina: riduzione offerta primaria imminente.",
            )

        if franchise == "Franchise":
            add_signal(
                "franchise_strength",
                "Licenza franchise forte",
                78,
                0.63,
                "IP globale consolidata con domanda secondaria tipicamente resiliente.",
            )

        if cls._contains_any_keyword(str(candidate.get("set_name") or ""), SPACE_STEM_KEYWORDS):
            add_signal(
                "stem_mission_icon",
                "Icona STEM/missione",
                79,
                0.61,
                "Tema space/STEM con appeal trasversale collezionistico e didattico.",
            )

        if bool(features.get("is_modular_family")) and theme in {"Icons", "City", "Star Wars"}:
            add_signal(
                "modular_continuity",
                "Continuita' collezione",
                84,
                0.76,
                "Set inserito in famiglia seriale/modulare ad alta fedelta'.",
            )

        if franchise in {"Cult Movie", "Franchise"} and isinstance(price, float) and 20.0 <= price <= 90.0:
            add_signal(
                "accessible_license_entry",
                "Entry point licenza forte",
                80,
                0.68,
                "Prezzo d'ingresso accessibile su IP forte: ampiezza domanda secondaria.",
            )

        if bool(features.get("is_scarcity_flagged")):
            add_signal(
                "scarcity_signal",
                "Segnale di scarsita'",
                82,
                0.64,
                "Indicatori testuali di tiratura limitata/uscita imminente.",
            )

        signals.sort(
            key=lambda row: (float(row.get("score", 0)) * float(row.get("confidence", 0.0)), int(row.get("score", 0))),
            reverse=True,
        )

        if signals:
            top = signals[:4]
            den = sum(float(row.get("confidence", 0.0)) for row in top) or 1.0
            weighted = sum(float(row.get("score", 0.0)) * float(row.get("confidence", 0.0)) for row in top) / den
            stack_bonus = min(8.0, max(0.0, (len(top) - 1) * 1.5))
            score = int(round(max(1.0, min(100.0, weighted + stack_bonus))))
            confidence_score = int(
                round(
                    max(1.0, min(100.0, (sum(float(row.get("confidence", 0.0)) for row in top) / len(top)) * 100.0))
                )
            )
            if len(top) == 1 and str(top[0].get("code") or "") == "retiring_window":
                score = min(score, 72)
                confidence_score = min(confidence_score, 58)
            labels = [str(row.get("label") or "") for row in top if row.get("label")]
            summary = " + ".join(labels[:2]) if labels else "Pattern multipli rilevati."
            return PatternEvaluation(
                score=score,
                confidence_score=confidence_score,
                summary=summary,
                signals=signals[:5],
                features=features,
            )

        baseline = 50.0
        if source in LEGO_PRIMARY_SOURCES:
            baseline += 4.0
        if franchise in {"Cult Movie", "Franchise"}:
            baseline += 3.0
        if isinstance(price, float) and 20.0 <= price <= 180.0:
            baseline += 2.0
        if category == "Art/Ideas":
            baseline += 2.0
        score = int(round(max(1.0, min(100.0, baseline))))
        return PatternEvaluation(
            score=score,
            confidence_score=35,
            summary="Nessun pattern dominante: prevale analisi quantitativa.",
            signals=[],
            features=features,
        )

    def _calculate_composite_score(
        self,
        *,
        ai_score: int,
        demand_score: int,
        forecast_score: int,
        pattern_score: int,
        ai_fallback_used: bool,
        historical_score: Optional[int] = None,
    ) -> int:
        if ai_fallback_used:
            ai_weight = 0.14
            demand_weight = 0.22
            quant_weight = 0.49
            pattern_weight = 0.15
        else:
            ai_weight = 0.34
            demand_weight = 0.20
            quant_weight = 0.30
            pattern_weight = 0.16

        base_composite = (
            ai_weight * float(ai_score)
            + demand_weight * float(demand_score)
            + quant_weight * float(forecast_score)
            + pattern_weight * float(pattern_score)
        )
        if historical_score is None:
            return max(1, min(100, int(round(base_composite))))

        historical_weight = max(0.0, min(0.35, float(self.historical_prior_weight)))
        composite = ((1.0 - historical_weight) * base_composite) + (historical_weight * float(historical_score))
        return max(1, min(100, int(round(composite))))

    @staticmethod
    def _effective_pattern_score(
        *,
        pattern_eval: PatternEvaluation,
        ai_fallback_used: bool,
    ) -> int:
        score = int(pattern_eval.score)
        if not ai_fallback_used:
            return max(1, min(100, score))

        primary_code = ""
        if pattern_eval.signals:
            primary_code = str(pattern_eval.signals[0].get("code") or "")

        if primary_code == "retiring_window":
            factor = 0.72
        else:
            factor = 0.84
        return max(1, min(100, int(round(score * factor))))

    def _effective_confidence_score(
        self,
        *,
        forecast: ForecastInsight,
        historical_prior: Optional[Dict[str, Any]],
        ai: AIInsight,
        pattern_eval: PatternEvaluation,
    ) -> int:
        base = max(1, min(100, int(forecast.confidence_score)))
        if ai.fallback_used:
            return max(1, min(100, base - 6))

        support_conf = 0
        prior_score = 0
        effective_sample = 0.0
        if historical_prior:
            try:
                support_conf = max(0, min(100, int(historical_prior.get("support_confidence") or 0)))
            except (TypeError, ValueError):
                support_conf = 0
            try:
                prior_score = max(0, min(100, int(historical_prior.get("prior_score") or 0)))
            except (TypeError, ValueError):
                prior_score = 0
            try:
                effective_sample = max(0.0, float(historical_prior.get("effective_sample_size") or 0.0))
            except (TypeError, ValueError):
                effective_sample = 0.0

        bonus = 0.0
        bonus += max(0.0, float(support_conf - 60)) * 0.25
        bonus += max(0.0, float(prior_score - 35)) * 0.20
        bonus += max(0.0, float(effective_sample - 10.0)) * 0.50

        if pattern_eval.signals and int(pattern_eval.confidence_score) >= 70:
            bonus += min(2.0, 0.6 * float(len(pattern_eval.signals)))

        if self._is_non_json_ai_note(ai.risk_note):
            bonus -= 2.0

        if self.historical_quality_guard_enabled and bool(self._historical_quality_profile.get("degraded")):
            bonus *= 0.85

        bonus = max(-8.0, min(12.0, bonus))
        return max(1, min(100, int(round(base + bonus))))

    def _row_forecast_data_points(self, row: Dict[str, Any]) -> int:
        top_level = row.get("forecast_data_points")
        if top_level is not None:
            try:
                return max(0, int(top_level))
            except (TypeError, ValueError):
                pass
        meta = row.get("metadata")
        if isinstance(meta, dict):
            raw = meta.get("forecast_data_points")
            try:
                return max(0, int(raw or 0))
            except (TypeError, ValueError):
                return 0
        return 0

    def _effective_high_confidence_thresholds(
        self,
        row: Dict[str, Any],
    ) -> tuple[float, int, bool, int]:
        probability_threshold = float(self.min_upside_probability)
        confidence_threshold = int(self.min_confidence_score)
        data_points = self._row_forecast_data_points(row)
        bootstrap_active = False

        if (
            self.bootstrap_thresholds_enabled
            and data_points > 0
            and data_points < int(self.bootstrap_min_history_points)
        ):
            probability_threshold = min(probability_threshold, float(self.bootstrap_min_upside_probability))
            confidence_threshold = min(confidence_threshold, int(self.bootstrap_min_confidence_score))
            bootstrap_active = True

        return probability_threshold, confidence_threshold, bootstrap_active, data_points

    @staticmethod
    def _row_metric_value(row: Dict[str, Any], key: str) -> Any:
        if row.get(key) is not None:
            return row.get(key)
        meta = row.get("metadata")
        if isinstance(meta, dict):
            return meta.get(key)
        return None

    def _row_pattern_signals(self, row: Dict[str, Any]) -> list[Dict[str, Any]]:
        raw = row.get("pattern_signals")
        if raw is None:
            raw = self._row_metric_value(row, "success_patterns")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _row_pattern_strength_score(self, row: Dict[str, Any]) -> int:
        direct_raw = row.get("pattern_score")
        if direct_raw is None:
            direct_raw = self._row_metric_value(row, "success_pattern_score")
        try:
            direct_score = max(0, int(float(direct_raw or 0)))
        except (TypeError, ValueError):
            direct_score = 0

        signals = self._row_pattern_signals(row)
        if not signals:
            return min(100, direct_score)

        ranked = sorted(
            signals,
            key=lambda item: (float(item.get("score") or 0.0) * float(item.get("confidence") or 0.0)),
            reverse=True,
        )[:3]
        total_conf = sum(max(0.05, float(item.get("confidence") or 0.0)) for item in ranked)
        if total_conf <= 0:
            return min(100, direct_score)
        weighted = sum(
            max(1.0, float(item.get("score") or 0.0)) * max(0.05, float(item.get("confidence") or 0.0))
            for item in ranked
        ) / total_conf
        signal_score = int(round(max(1.0, min(100.0, weighted))))
        if direct_score > 0:
            return max(min(100, direct_score), signal_score)
        return signal_score

    def _historical_contextual_thresholds_for_row(
        self,
        *,
        row: Dict[str, Any],
        min_samples: int,
        min_win_rate_pct: float,
        min_support_confidence: int,
        min_prior_score: int,
    ) -> tuple[int, float, int, int, bool, Optional[str]]:
        if not self.historical_contextual_gate_enabled:
            return min_samples, min_win_rate_pct, min_support_confidence, min_prior_score, False, None

        signals = self._row_pattern_signals(row)
        pattern_strength = self._row_pattern_strength_score(row)
        if not signals or pattern_strength < int(self.historical_context_strong_pattern_min_score):
            return min_samples, min_win_rate_pct, min_support_confidence, min_prior_score, False, None

        strong_codes = {
            "exclusive_cult_license",
            "series_completism",
            "adult_display_value",
            "flagship_collector",
            "modular_continuity",
            "nostalgia_vehicle",
            "stem_mission_icon",
            "accessible_license_entry",
        }
        strong_signal_count = sum(
            1
            for signal in signals
            if str(signal.get("code") or "") in strong_codes
            and float(signal.get("confidence") or 0.0) >= 0.6
        )
        if strong_signal_count <= 0:
            return min_samples, min_win_rate_pct, min_support_confidence, min_prior_score, False, None

        sample_raw = self._row_metric_value(row, "historical_sample_size")
        support_raw = self._row_metric_value(row, "historical_support_confidence")
        prior_raw = self._row_metric_value(row, "historical_prior_score")
        avg_roi_raw = self._row_metric_value(row, "historical_avg_roi_12m_pct")
        try:
            sample_size = max(0, int(float(sample_raw or 0)))
        except (TypeError, ValueError):
            sample_size = 0
        try:
            support_conf = max(0, int(float(support_raw or 0)))
        except (TypeError, ValueError):
            support_conf = 0
        try:
            prior_score = max(0, int(float(prior_raw or 0)))
        except (TypeError, ValueError):
            prior_score = 0
        try:
            avg_roi_pct = float(avg_roi_raw) if avg_roi_raw is not None else None
        except (TypeError, ValueError):
            avg_roi_pct = None

        if sample_size < int(min_samples) or support_conf < int(min_support_confidence):
            return min_samples, min_win_rate_pct, min_support_confidence, min_prior_score, False, None
        if prior_score <= 0:
            return min_samples, min_win_rate_pct, min_support_confidence, min_prior_score, False, None
        if avg_roi_pct is not None and avg_roi_pct < max(6.0, float(self.target_roi_pct) * 0.30):
            return min_samples, min_win_rate_pct, min_support_confidence, min_prior_score, False, None

        sample_factor = min(1.0, max(0.0, (sample_size - float(min_samples)) / max(8.0, float(min_samples) * 0.8)))
        support_factor = min(
            1.0,
            max(0.0, (support_conf - float(min_support_confidence)) / max(8.0, float(100 - min_support_confidence))),
        )
        pattern_factor = min(
            1.0,
            max(
                0.0,
                (pattern_strength - float(self.historical_context_strong_pattern_min_score)) / 25.0,
            ),
        )
        if avg_roi_pct is None:
            roi_factor = 0.45
        else:
            roi_factor = min(1.0, max(0.0, avg_roi_pct / max(1.0, float(self.target_roi_pct))))

        relax_strength = max(0.15, min(1.0, (sample_factor + support_factor + pattern_factor + roi_factor) / 4.0))
        win_relax = int(round(float(self.historical_context_max_win_rate_relax_pct) * relax_strength))
        support_relax = int(round(float(self.historical_context_max_support_relax) * relax_strength))
        prior_relax = int(round(float(self.historical_context_max_prior_relax) * relax_strength))

        if win_relax <= 0 and support_relax <= 0 and prior_relax <= 0:
            return min_samples, min_win_rate_pct, min_support_confidence, min_prior_score, False, None

        adjusted_win_rate = max(35.0, float(min_win_rate_pct) - float(win_relax))
        adjusted_support = max(35, int(min_support_confidence) - int(support_relax))
        adjusted_prior = max(40, int(min_prior_score) - int(prior_relax))

        reason = (
            "Gate storico contestuale attivo "
            f"(pattern forte + coorte robusta: win-rate>={adjusted_win_rate:.0f}%, "
            f"supporto>={adjusted_support}, prior>={adjusted_prior})."
        )
        return min_samples, adjusted_win_rate, adjusted_support, adjusted_prior, True, reason

    def _historical_high_confidence_status(self, row: Dict[str, Any]) -> tuple[bool, list[str]]:
        if not self.historical_high_conf_required:
            return True, []

        reasons: list[str] = []
        (
            min_samples,
            min_win_rate_pct,
            min_support_confidence,
            min_prior_score,
            adaptive_active,
        ) = self._effective_historical_high_confidence_thresholds()
        (
            min_samples,
            min_win_rate_pct,
            min_support_confidence,
            min_prior_score,
            contextual_active,
            contextual_reason,
        ) = self._historical_contextual_thresholds_for_row(
            row=row,
            min_samples=min_samples,
            min_win_rate_pct=min_win_rate_pct,
            min_support_confidence=min_support_confidence,
            min_prior_score=min_prior_score,
        )
        sample_size_raw = self._row_metric_value(row, "historical_sample_size")
        try:
            sample_size = max(0, int(float(sample_size_raw or 0)))
        except (TypeError, ValueError):
            sample_size = 0

        quality_soft_gate_active = bool(
            self.historical_quality_guard_enabled
            and self.historical_quality_soft_gate_enabled
            and self._historical_quality_profile.get("degraded")
        )
        if sample_size < int(min_samples):
            if quality_soft_gate_active and sample_size == 0:
                return True, []
            reasons.append(
                f"Evidenza storica insufficiente ({sample_size} < {min_samples} campioni)"
            )
            if quality_soft_gate_active:
                reasons.insert(0, "Gate storico quality-aware attivo")
            return False, reasons

        win_rate_pct_raw = self._row_metric_value(row, "historical_win_rate_12m_pct")
        try:
            win_rate_pct = float(win_rate_pct_raw or 0.0)
        except (TypeError, ValueError):
            win_rate_pct = 0.0
        if win_rate_pct < float(min_win_rate_pct):
            reasons.append(
                f"Win-rate storico 12m sotto soglia ({win_rate_pct:.1f}% < {min_win_rate_pct:.0f}%)"
            )

        support_conf_raw = self._row_metric_value(row, "historical_support_confidence")
        try:
            support_conf = max(0, int(float(support_conf_raw or 0)))
        except (TypeError, ValueError):
            support_conf = 0
        if support_conf < int(min_support_confidence):
            reasons.append(
                f"Confidenza supporto storico sotto soglia ({support_conf} < {min_support_confidence})"
            )

        prior_score_raw = self._row_metric_value(row, "historical_prior_score")
        try:
            prior_score = max(0, int(float(prior_score_raw or 0)))
        except (TypeError, ValueError):
            prior_score = 0
        if prior_score < int(min_prior_score):
            reasons.append(
                f"Prior storico sotto soglia ({prior_score} < {min_prior_score})"
            )

        if reasons and contextual_active and contextual_reason:
            reasons.insert(0, contextual_reason)
        if reasons and adaptive_active:
            reasons.insert(0, "Gate storico adattivo attivo")
        return not reasons, reasons

    def _is_high_confidence_pick(self, row: Dict[str, Any]) -> bool:
        return self._high_confidence_signal_strength(row).startswith("HIGH_CONFIDENCE")

    def _high_confidence_signal_strength(self, row: Dict[str, Any]) -> str:
        if row.get("ai_fallback_used"):
            return "LOW_CONFIDENCE"
        probability_pct = float(row.get("forecast_probability_upside_12m") or 0.0)
        confidence_score = int(row.get("confidence_score") or 0)
        composite_score = int(row.get("composite_score") or row.get("ai_investment_score") or 0)
        probability_threshold, confidence_threshold, bootstrap_active, _points = self._effective_high_confidence_thresholds(row)
        historical_ok, _historical_reasons = self._historical_high_confidence_status(row)
        passed = (
            composite_score >= self.min_composite_score
            and probability_pct >= (probability_threshold * 100.0)
            and confidence_score >= confidence_threshold
            and historical_ok
        )
        if not passed:
            return "LOW_CONFIDENCE"
        if bootstrap_active:
            return "HIGH_CONFIDENCE_BOOTSTRAP"
        return "HIGH_CONFIDENCE_STRICT"

    def _build_low_confidence_note(self, row: Dict[str, Any]) -> str:
        existing = str(row.get("risk_note") or "").strip()
        if existing:
            return existing

        if row.get("ai_fallback_used"):
            return "Score AI in fallback nel ciclo corrente: verifica manuale consigliata."

        reasons: list[str] = []
        probability_pct = float(row.get("forecast_probability_upside_12m") or 0.0)
        probability_threshold, confidence_threshold, bootstrap_active, data_points = self._effective_high_confidence_thresholds(row)
        min_probability_pct = probability_threshold * 100.0
        if probability_pct < min_probability_pct:
            reasons.append(
                f"Probabilita upside 12m sotto soglia ({probability_pct:.1f}% < {min_probability_pct:.0f}%)"
            )

        confidence_score = int(row.get("confidence_score") or 0)
        if confidence_score < confidence_threshold:
            reasons.append(
                f"Confidenza dati sotto soglia ({confidence_score} < {confidence_threshold})"
            )

        if bootstrap_active:
            reasons.append(
                f"Bootstrap soglie attivo (data points {data_points} < {self.bootstrap_min_history_points})"
            )

        historical_ok, historical_reasons = self._historical_high_confidence_status(row)
        if not historical_ok:
            reasons.extend(historical_reasons)

        if reasons:
            return "; ".join(reasons) + "."
        return "Segnale sotto criteri HIGH_CONFIDENCE: verifica manuale consigliata."

    def _estimate_market_demand(
        self,
        candidate: Dict[str, Any],
        ai_score: int,
        *,
        forecast: Optional[ForecastInsight] = None,
        recent_prices: Optional[list[Dict[str, Any]]] = None,
    ) -> int:
        set_id = str(candidate.get("set_id") or "")
        if not set_id:
            return max(1, min(100, ai_score))

        if recent_prices is not None:
            recent = recent_prices
        else:
            try:
                recent = self.repository.get_recent_market_prices(set_id, days=30)
            except Exception:  # noqa: BLE001
                recent = []

        unique_days = len(
            {
                str(row.get("recorded_at") or "")[:10]
                for row in recent
                if str(row.get("recorded_at") or "").strip()
            }
        )
        liquidity_factor = min(32.0, (len(recent) * 1.8) + (unique_days * 1.1))
        source = str(candidate.get("source") or "")
        source_bonus = 12.0 if source in {"lego_retiring", "lego_proxy_reader", "lego_http_fallback"} else 6.0
        quant_bonus = 0.0
        confidence_bonus = 0.0
        if forecast is not None:
            quant_bonus = min(
                24.0,
                (forecast.forecast_score * 0.10)
                + (forecast.probability_upside_12m * 14.0),
            )
            confidence_bonus = min(
                12.0,
                max(0.0, (float(forecast.confidence_score) - 30.0) * 0.16),
            )

        price_penalty = 0.0
        try:
            price_value = float(candidate.get("current_price"))
        except (TypeError, ValueError):
            price_value = None
        if price_value is not None:
            if price_value < 12.0:
                price_penalty = 4.0
            elif price_value > 450.0:
                price_penalty = 6.0
            elif price_value > 300.0:
                price_penalty = 3.0

        raw_score = (
            (float(ai_score) * 0.48)
            + liquidity_factor
            + source_bonus
            + quant_bonus
            + confidence_bonus
            - price_penalty
        )
        if raw_score > 92.0:
            raw_score = 92.0 + ((raw_score - 92.0) * 0.35)

        final_score = int(round(raw_score))
        return max(1, min(100, final_score))

    @staticmethod
    def _extract_first_date(text: str) -> Optional[str]:
        if not text:
            return None

        iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if iso_match:
            value = iso_match.group(1)
            try:
                date.fromisoformat(value)
                return value
            except ValueError:
                pass

        eu_match = re.search(r"\b([0-3]?\d)[/.\-]([0-1]?\d)[/.\-](20\d{2})\b", text)
        if eu_match:
            day = int(eu_match.group(1))
            month = int(eu_match.group(2))
            year = int(eu_match.group(3))
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None

        return None

    @classmethod
    def _estimate_eol_prediction(
        cls,
        *,
        source: str,
        set_id: Optional[str],
        theme: str,
        price: Optional[float],
        context_text: str = "",
    ) -> str:
        explicit = cls._extract_first_date(context_text)
        if explicit:
            return explicit

        base_days_by_source = {
            "lego_retiring": 72.0,
            "lego_proxy_reader": 92.0,
            "lego_http_fallback": 98.0,
        }
        base = base_days_by_source.get(source, 90.0)

        if price is not None:
            if price <= 20.0:
                base -= 16.0
            elif price <= 60.0:
                base -= 8.0
            elif price <= 200.0:
                base += 4.0
            elif price <= 400.0:
                base += 14.0
            else:
                base += 22.0

        theme_adjust = {
            "Star Wars": 10.0,
            "Technic": 12.0,
            "Icons": 8.0,
            "Botanicals": -4.0,
            "City": -6.0,
            "Friends": -8.0,
        }
        base += theme_adjust.get(theme, 0.0)

        sid = str(set_id or "").strip()
        if sid:
            hash_adjust = (sum(ord(ch) for ch in sid) % 21) - 10
            base += float(hash_adjust)

        days = int(round(max(30.0, min(240.0, base))))
        return (date.today() + timedelta(days=days)).isoformat()

    @staticmethod
    def _extract_json(raw_text: str) -> Any:
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
