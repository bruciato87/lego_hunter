from __future__ import annotations

import argparse
import asyncio
import html
import logging
import os
from datetime import date, datetime
from typing import Any, Awaitable, Callable, Iterable, Optional
from urllib.parse import quote_plus

from telegram import Bot, BotCommand, Update
from telegram.constants import ParseMode
from telegram.error import NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes

from fiscal import FiscalGuardian
from models import LegoHunterRepository, MarketTimeSeriesRecord
from oracle import DiscoveryOracle
from scrapers import PLAYWRIGHT_AVAILABLE, SecondaryMarketValidator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("lego_hunter.bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def _telegram_call_with_retry(
    *,
    operation_name: str,
    fn: Callable[[], Awaitable[Any]],
    max_attempts: int = 3,
    non_fatal: bool = False,
    base_delay_seconds: float = 1.5,
    retry_timeouts: bool = True,
    timeout_assume_delivered: bool = False,
) -> Optional[Any]:
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except RetryAfter as exc:
            retry_seconds = max(1.0, float(getattr(exc, "retry_after", 1.0)))
            LOGGER.warning(
                "Telegram flood-control on %s (attempt %s/%s). Retry in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                retry_seconds,
            )
            if attempt >= max_attempts:
                if non_fatal:
                    LOGGER.warning("Non-fatal Telegram operation failed after retries: %s", operation_name)
                    return None
                raise
            await asyncio.sleep(retry_seconds)
        except (TimedOut, NetworkError) as exc:
            if isinstance(exc, TimedOut) and not retry_timeouts:
                if timeout_assume_delivered:
                    LOGGER.warning(
                        "Telegram timeout on %s (attempt %s/%s): delivery uncertain, skip retry to avoid duplicates.",
                        operation_name,
                        attempt,
                        max_attempts,
                    )
                    return {"delivery": "uncertain_timeout_assumed"}
                if non_fatal:
                    LOGGER.warning("Non-fatal Telegram operation timed out: %s", operation_name)
                    return None
                raise
            retry_seconds = base_delay_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "Telegram transient error on %s (attempt %s/%s): %s. Retrying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                exc,
                retry_seconds,
            )
            if attempt >= max_attempts:
                if non_fatal:
                    LOGGER.warning("Non-fatal Telegram operation failed after retries: %s", operation_name)
                    return None
                raise
            await asyncio.sleep(retry_seconds)
        except TelegramError:
            if non_fatal:
                LOGGER.exception("Non-fatal Telegram operation failed: %s", operation_name)
                return None
            raise


async def validate_secondary_deals_with_scrapers(
    *,
    repository: LegoHunterRepository,
    opportunities: Iterable[dict[str, Any]],
    per_set_limit: int = 3,
) -> list[dict[str, Any]]:
    candidates = list(opportunities)
    if not candidates:
        return []

    validator = SecondaryMarketValidator()
    results = await validator.compare_secondary_prices(candidates, per_set_limit=per_set_limit)
    merged: list[dict[str, Any]] = []

    for opportunity in candidates:
        key = str(opportunity.get("set_id") or opportunity.get("set_name"))
        listings = results.get(key, [])
        primary_price = float(opportunity.get("current_price") or 0.0)

        best_secondary = None
        if listings:
            best_secondary = min(listings, key=lambda row: row.price)
            try:
                await asyncio.to_thread(
                    repository.insert_market_snapshot,
                    MarketTimeSeriesRecord(
                        set_id=best_secondary.set_id,
                        set_name=best_secondary.set_name,
                        platform=best_secondary.platform,
                        listing_type=best_secondary.condition,
                        price=best_secondary.price,
                        shipping_cost=0.0,
                        listing_url=best_secondary.listing_url,
                        raw_payload={"source_note": best_secondary.source_note},
                    ),
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


class LegoHunterTelegramBot:
    def __init__(
        self,
        repository: LegoHunterRepository,
        oracle: Optional[DiscoveryOracle],
        fiscal_guardian: FiscalGuardian,
        *,
        allowed_chat_id: Optional[str] = None,
        oracle_factory: Optional[Callable[[], DiscoveryOracle]] = None,
    ) -> None:
        self.repository = repository
        self._oracle = oracle
        self._oracle_factory = oracle_factory
        self.fiscal_guardian = fiscal_guardian
        self.allowed_chat_id = str(allowed_chat_id) if allowed_chat_id else None

    def _get_oracle(self) -> DiscoveryOracle:
        if self._oracle is None:
            if self._oracle_factory is None:
                raise RuntimeError("DiscoveryOracle non configurato")
            self._oracle = self._oracle_factory()
        return self._oracle

    @staticmethod
    def supported_commands() -> list[BotCommand]:
        return [
            BotCommand("start", "Attiva il bot e mostra guida rapida"),
            BotCommand("scova", "Scopre i migliori set da monitorare/acquistare"),
            BotCommand("radar", "Mostra le opportunita' piu' forti in radar"),
            BotCommand("cerca", "Cerca un set nel radar: /cerca 75367"),
            BotCommand("offerte", "Confronta prezzo primario vs secondario"),
            BotCommand("collezione", "Valore attuale portfolio e ROI latente"),
            BotCommand("vendi", "Segnali vendita con ROI netto > 30%"),
            BotCommand("help", "Guida completa ai comandi"),
        ]

    async def register_commands(self, app: Application) -> None:
        await app.bot.set_my_commands(self.supported_commands())

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text(
            "Lego_Hunter attivo. Usa /help per i comandi disponibili.",
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text(
            "🧱 LEGO HUNTER - Guida comandi\n\n"
            "Comandi principali:\n"
            "/scova - Discovery completa (LEGO + Amazon + ranking AI) e Top Picks del ciclo.\n"
            "/radar - Mostra le opportunita' gia' in Opportunity Radar, ordinate per score.\n"
            "/cerca <id|nome|tema> - Cerca set nel radar (es. /cerca 75367).\n"
            "/offerte - Verifica se sul secondario trovi prezzi migliori del primario.\n"
            "/collezione - Riepilogo portfolio: capitale investito, valore stimato, ROI latente.\n"
            "/vendi - Segnali vendita con ROI netto > 30% (Vinted/Subito/eBay.it, bloccati se DAC7 a rischio).\n\n"
            "Come leggere i segnali:\n"
            "Score = composito (AI + Quant + Demand).\n"
            "HIGH_CONFIDENCE = score sopra soglia + probabilita' upside 12m + confidenza dati alta + AI non in fallback.\n\n"
            "Alias compatibilita':\n"
            "/hunt -> /scova\n"
            "/portfolio -> /collezione\n"
            "/sell_signal -> /vendi\n\n"
            "Esempi rapidi:\n"
            "/cerca millennium falcon\n"
            "/cerca 10332\n"
            "/offerte\n"
            "/help",
        )

    async def cmd_scova(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        await update.message.reply_text("Scansione in corso: Lego Retiring Soon + Amazon + ranking AI...")
        oracle = self._get_oracle()
        try:
            report = await oracle.discover_with_diagnostics(
                persist=True,
                top_limit=20,
                fallback_limit=3,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Discovery failed")
            await update.message.reply_text(f"Errore discovery: {exc}")
            return

        selected = report.get("selected", [])
        diagnostics = report.get("diagnostics", {})
        if not selected:
            lines = [
                "🧱 <b>Discovery LEGO</b>",
                "Nessuna opportunita' disponibile in questo ciclo.",
            ]
            if diagnostics.get("anti_bot_alert"):
                lines.append(f"🚨 {html.escape(str(diagnostics.get('anti_bot_message') or 'Possibile anti-bot.'))}")
            sell_signals = await self._compute_sell_signals(respect_fiscal_gate=True)
            if sell_signals:
                lines.append("")
                lines.append("💸 <b>Occasioni vendita rilevate</b>")
                lines.extend(self._format_sell_signal_lines(sell_signals[:3]))
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        lines = ["🧱 <b>Discovery LEGO</b>"]
        lines.extend(self._format_discovery_report(report, top_limit=3))
        sell_signals = await self._compute_sell_signals(respect_fiscal_gate=True)
        if sell_signals:
            lines.append("")
            lines.append("💸 <b>Occasioni vendita rilevate</b>")
            lines.extend(self._format_sell_signal_lines(sell_signals[:3]))
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def cmd_radar(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        try:
            rows = await asyncio.to_thread(self.repository.get_top_opportunities, 8, 50)
        except Exception as exc:  # noqa: BLE001
            await update.message.reply_text(f"Errore lettura radar: {exc}")
            return

        if not rows:
            await update.message.reply_text("Radar vuoto. Lancia /scova per popolarlo.")
            return

        lines = ["Radar opportunita' LEGO:"]
        for idx, row in enumerate(rows, start=1):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            composite = int(row.get("composite_score") or row.get("ai_investment_score") or metadata.get("composite_score") or 0)
            ai_raw = int(row.get("ai_raw_score") or metadata.get("ai_raw_score") or row.get("ai_investment_score") or 0)
            quant = int(row.get("forecast_score") or metadata.get("forecast_score") or 0)
            lines.append(
                f"{idx}. {row.get('set_name')} ({row.get('set_id')}) - "
                f"Score {composite}/100 | AI {ai_raw}/100 | Quant {quant}/100 | "
                f"Demand {row.get('market_demand_score')}/100"
            )
        await update.message.reply_text("\n".join(lines))

    async def cmd_cerca(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        query = " ".join(context.args).strip()
        if not query:
            await update.message.reply_text("Uso: /cerca <set_id|nome|tema>")
            return

        try:
            rows = await asyncio.to_thread(self.repository.search_opportunities, query, 10)
        except Exception as exc:  # noqa: BLE001
            await update.message.reply_text(f"Errore ricerca: {exc}")
            return

        if not rows:
            await update.message.reply_text(f"Nessun risultato per '{query}'.")
            return

        lines = [f"Risultati per '{query}':"]
        for row in rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            composite = int(row.get("composite_score") or row.get("ai_investment_score") or metadata.get("composite_score") or 0)
            ai_raw = int(row.get("ai_raw_score") or metadata.get("ai_raw_score") or row.get("ai_investment_score") or 0)
            quant = int(row.get("forecast_score") or metadata.get("forecast_score") or 0)
            lines.append(
                f"- {row.get('set_name')} ({row.get('set_id')}) | Theme: {row.get('theme') or 'n/d'} | "
                f"Score {composite}/100 (AI {ai_raw}/100 | Quant {quant}/100)"
            )
        await update.message.reply_text("\n".join(lines))

    async def cmd_offerte(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        opportunities = await asyncio.to_thread(self.repository.get_top_opportunities, 8, 50)
        if not opportunities:
            await update.message.reply_text("Nessuna opportunita' in radar. Lancia /scova e riprova.")
            return

        if not PLAYWRIGHT_AVAILABLE:
            cached_deals = await self._get_cached_secondary_deals(opportunities[:8], max_age_hours=72.0)
            if not cached_deals:
                await update.message.reply_text(
                    "Nessuna offerta recente in cache cloud (ultime 72h). "
                    "Riprova dopo il prossimo ciclo schedulato."
                )
                return
            lines = ["Offerte LEGO (cache cloud, ultime 72h):"]
            lines.extend(self._format_secondary_deals(cached_deals[:5]))
            await update.message.reply_text("\n\n".join(lines), disable_web_page_preview=True)
            return

        await update.message.reply_text("Verifica offerte secondarie (Vinted/Subito) in corso...")

        try:
            deals = await validate_secondary_deals_with_scrapers(
                repository=self.repository,
                opportunities=opportunities[:6],
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Deal validation failed")
            cached_deals = await self._get_cached_secondary_deals(opportunities[:8], max_age_hours=72.0)
            if cached_deals:
                lines = ["Offerte LEGO (cache cloud, fallback):"]
                lines.extend(self._format_secondary_deals(cached_deals[:5]))
                await update.message.reply_text("\n\n".join(lines), disable_web_page_preview=True)
                return
            await update.message.reply_text(f"Errore verifica offerte: {exc}")
            return

        good_deals = [row for row in deals if (row.get("discount_vs_primary_pct") or 0) >= 10]
        if not good_deals:
            await update.message.reply_text("Nessuna offerta >=10% di sconto trovata ora.")
            return

        lines = ["Offerte LEGO interessanti:"]
        lines.extend(self._format_secondary_deals(good_deals[:5]))
        await update.message.reply_text("\n\n".join(lines), disable_web_page_preview=True)

    async def _get_cached_secondary_deals(
        self,
        opportunities: list[dict[str, Any]],
        *,
        max_age_hours: float,
    ) -> list[dict[str, Any]]:
        deals: list[dict[str, Any]] = []
        for opportunity in opportunities:
            set_id = str(opportunity.get("set_id") or "").strip()
            if not set_id:
                continue
            secondary = await asyncio.to_thread(
                self.repository.get_best_recent_secondary_price,
                set_id,
                max_age_hours,
            )
            if not secondary:
                continue

            primary_price = float(opportunity.get("current_price") or 0.0)
            secondary_price = float(secondary.get("price") or 0.0)
            discount_pct = 0.0
            if primary_price > 0 and secondary_price > 0:
                discount_pct = ((primary_price - secondary_price) / primary_price) * 100

            deals.append(
                {
                    **opportunity,
                    "secondary_best_price": secondary_price,
                    "secondary_platform": secondary.get("platform"),
                    "secondary_url": secondary.get("listing_url"),
                    "discount_vs_primary_pct": round(discount_pct, 2),
                }
            )

        deals.sort(
            key=lambda row: (
                row.get("discount_vs_primary_pct") or 0.0,
                row.get("ai_investment_score") or 0,
            ),
            reverse=True,
        )
        return deals

    @staticmethod
    def _format_secondary_deals(rows: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            message = (
                f"- {row.get('set_name')} ({row.get('set_id')})\n"
                f"Secondario: {LegoHunterTelegramBot._fmt_eur(row.get('secondary_best_price'))} su {row.get('secondary_platform')}\n"
                f"Sconto vs prezzo primario: {row.get('discount_vs_primary_pct')}%"
            )
            url = str(row.get("secondary_url") or "").strip()
            if url:
                message += f"\nLink: {url}"
            lines.append(message)
        return lines

    async def cmd_collezione(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        try:
            holdings = await asyncio.to_thread(self.repository.get_portfolio, "holding")
        except Exception as exc:  # noqa: BLE001
            await update.message.reply_text(f"Errore portfolio: {exc}")
            return

        if not holdings:
            await update.message.reply_text("Portfolio vuoto: nessun set in stato holding.")
            return

        invested = 0.0
        current_value = 0.0
        lines = ["Portfolio LEGO:"]

        for row in holdings:
            qty = int(row.get("quantity") or 1)
            buy_price = float(row.get("purchase_price") or 0.0)
            ship_in = float(row.get("shipping_in_cost") or 0.0)
            cost_total = (buy_price + ship_in) * qty
            invested += cost_total

            latest = await asyncio.to_thread(self.repository.get_best_secondary_price, row.get("set_id"))
            current_unit = float(latest.get("price") if latest else (row.get("estimated_market_price") or buy_price))
            value_total = current_unit * qty
            current_value += value_total

            roi = ((value_total - cost_total) / cost_total * 100) if cost_total > 0 else 0.0
            lines.append(
                f"- {row.get('set_name')} ({row.get('set_id')}) x{qty}: "
                f"valore {self._fmt_eur(value_total)} | ROI {roi:+.1f}%"
            )

        total_roi = ((current_value - invested) / invested * 100) if invested > 0 else 0.0
        lines.append("")
        lines.append(f"Capitale investito: {self._fmt_eur(invested)}")
        lines.append(f"Valore stimato: {self._fmt_eur(current_value)}")
        lines.append(f"ROI latente totale: {total_roi:+.2f}%")

        await update.message.reply_text("\n".join(lines))

    async def cmd_vendi(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        safety = await asyncio.to_thread(self.fiscal_guardian.check_safety_status)
        if not safety.get("allow_sell_signals", False):
            await update.message.reply_text(
                f"Segnali vendita sospesi per sicurezza fiscale: {safety.get('message')}"
            )
            return

        try:
            signals = await self._compute_sell_signals()
        except Exception as exc:  # noqa: BLE001
            await update.message.reply_text(f"Errore calcolo segnali vendita: {exc}")
            return

        if not signals:
            await update.message.reply_text("Nessun set supera ROI netto > 30% al momento.")
            return

        lines = ["Segnali uscita (ROI netto > 30%):"]
        lines.extend(self._format_sell_signal_lines(signals[:10]))

        await update.message.reply_text("\n".join(lines))

    async def _compute_sell_signals(self, *, respect_fiscal_gate: bool = False) -> list[dict[str, Any]]:
        if respect_fiscal_gate:
            safety = await asyncio.to_thread(self.fiscal_guardian.check_safety_status)
            if not safety.get("allow_sell_signals", False):
                return []

        holdings = await asyncio.to_thread(self.repository.get_portfolio, "holding")
        signals: list[dict[str, Any]] = []
        estimated_shipping_out = 7.0
        sale_platforms = ("vinted", "subito", "ebay")

        for row in holdings:
            set_id = str(row.get("set_id") or "").strip()
            if not set_id:
                continue

            qty = int(row.get("quantity") or 1)
            buy_price = float(row.get("purchase_price") or 0.0)
            ship_in = float(row.get("shipping_in_cost") or 0.0)
            total_cost = (buy_price + ship_in) * qty
            if total_cost <= 0:
                continue

            platform_quotes: list[dict[str, Any]] = []
            for platform in sale_platforms:
                latest = await asyncio.to_thread(self.repository.get_latest_price, set_id, platform)
                if not latest:
                    continue
                sale_unit = float(latest.get("price") or 0.0)
                if sale_unit <= 0:
                    continue
                platform_quotes.append(
                    {
                        "platform": platform,
                        "sale_unit": sale_unit,
                    }
                )

            if not platform_quotes:
                continue

            best_quote = max(platform_quotes, key=lambda item: item["sale_unit"])
            sale_unit = float(best_quote["sale_unit"])
            net_sale_total = (sale_unit - estimated_shipping_out) * qty
            roi_net = ((net_sale_total - total_cost) / total_cost) * 100
            if roi_net <= 30.0:
                continue

            signals.append(
                {
                    "set_name": row.get("set_name"),
                    "set_id": set_id,
                    "platform": best_quote.get("platform") or "n/d",
                    "sale_unit": sale_unit,
                    "roi_net": roi_net,
                }
            )

        signals.sort(key=lambda row: row["roi_net"], reverse=True)
        return signals

    def _format_sell_signal_lines(self, rows: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            platform = str(row.get("platform") or "n/d").lower()
            if platform == "ebay":
                platform = "ebay.it"
            lines.append(
                f"- {row.get('set_name')} ({row.get('set_id')}) | "
                f"{platform} {self._fmt_eur(row.get('sale_unit'))} | "
                f"ROI netto {float(row.get('roi_net') or 0.0):.1f}%"
            )
        return lines

    def _is_authorized(self, update: Update) -> bool:
        if self.allowed_chat_id is None:
            return True

        chat_id = str(update.effective_chat.id) if update.effective_chat else None
        if chat_id != self.allowed_chat_id:
            LOGGER.warning("Unauthorized chat attempted access: %s", chat_id)
            if update.message:
                asyncio.create_task(update.message.reply_text("Chat non autorizzata."))
            return False
        return True

    @staticmethod
    def _fmt_eur(value: Any) -> str:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            amount = 0.0
        return f"€{amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    @staticmethod
    def _format_eol_date(value: Any) -> str:
        if value is None:
            return "n/d"

        if isinstance(value, datetime):
            return value.date().strftime("%d/%m/%Y")
        if isinstance(value, date):
            return value.strftime("%d/%m/%Y")

        raw = str(value).strip()
        if not raw:
            return "n/d"

        raw_date = raw.split("T", 1)[0]
        try:
            parsed = date.fromisoformat(raw_date)
            return parsed.strftime("%d/%m/%Y")
        except ValueError:
            return raw

    @staticmethod
    def _extract_listing_url(row: dict[str, Any]) -> Optional[str]:
        direct_url = str(row.get("listing_url") or "").strip()
        if direct_url.startswith("http://") or direct_url.startswith("https://"):
            return direct_url

        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            meta_url = str(metadata.get("listing_url") or "").strip()
            if meta_url.startswith("http://") or meta_url.startswith("https://"):
                return meta_url

        return None

    @staticmethod
    def _format_pick_link(row: dict[str, Any]) -> str:
        set_id = str(row.get("set_id") or "").strip()
        listing_url = LegoHunterTelegramBot._extract_listing_url(row)

        if listing_url:
            safe_url = html.escape(listing_url, quote=True)
            lower_url = listing_url.lower()
            if "lego.com" in lower_url:
                return f'🔗 <a href="{safe_url}">Apri su LEGO</a>'
            return f'🔗 <a href="{safe_url}">Apri sorgente</a>'

        if set_id:
            lego_search_url = f"https://www.lego.com/it-it/search?q={quote_plus(set_id)}"
            safe_url = html.escape(lego_search_url, quote=True)
            return f'🔎 <a href="{safe_url}">Cerca su LEGO</a>'

        return "🔎 Link non disponibile"

    @staticmethod
    def _format_discovery_report(report: dict[str, Any], *, top_limit: int = 3) -> list[str]:
        selected = list(report.get("selected") or [])[:top_limit]
        diagnostics = report.get("diagnostics") or {}
        ai_runtime = diagnostics.get("ai_runtime") or {}
        bootstrap_enabled = bool(diagnostics.get("bootstrap_thresholds_enabled"))
        bootstrap_min_history_points = int(diagnostics.get("bootstrap_min_history_points") or 0)
        bootstrap_min_probability_pct = (
            float(diagnostics.get("bootstrap_min_probability_high_confidence") or 0.0) * 100.0
        )
        bootstrap_min_confidence = int(diagnostics.get("bootstrap_min_confidence_score_high_confidence") or 0)

        lines: list[str] = []
        if diagnostics.get("fallback_used"):
            above_threshold_count = int(diagnostics.get("above_threshold_count") or 0)
            high_conf_count = int(diagnostics.get("above_threshold_high_confidence_count") or 0)
            if above_threshold_count > 0 and high_conf_count == 0:
                lines.append("⚠️ Nessun set <b>HIGH_CONFIDENCE</b> nel ciclo: mostro i migliori <b>LOW_CONFIDENCE</b>.")
            else:
                lines.append("⚠️ Nessun set sopra soglia composita nel ciclo: mostro i migliori <b>LOW_CONFIDENCE</b>.")
        else:
            lines.append("✅ Opportunita' sopra soglia trovate.")

        if selected:
            lines.append("")
            lines.append("<b>Top Picks</b>")
            for idx, row in enumerate(selected, start=1):
                strength = str(row.get("signal_strength") or "HIGH_CONFIDENCE")
                badge = "🟢" if strength == "HIGH_CONFIDENCE" else "🟡"
                set_name = html.escape(str(row.get("set_name") or "n/d"))
                set_id = html.escape(str(row.get("set_id") or "n/d"))
                source = html.escape(str(row.get("source") or "unknown"))
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                composite = int(row.get("composite_score") or row.get("ai_investment_score") or metadata.get("composite_score") or 0)
                ai_score = int(row.get("ai_raw_score") or metadata.get("ai_raw_score") or row.get("ai_investment_score") or 0)
                quant_score = int(row.get("forecast_score") or metadata.get("forecast_score") or 0)
                pattern_score = int(row.get("pattern_score") or metadata.get("success_pattern_score") or 0)
                pattern_summary = str(row.get("pattern_summary") or metadata.get("success_pattern_summary") or "").strip()
                prob_12m = float(
                    row.get("forecast_probability_upside_12m")
                    or metadata.get("forecast_probability_upside_12m")
                    or 0.0
                )
                roi_12m = float(row.get("expected_roi_12m_pct") or metadata.get("expected_roi_12m_pct") or 0.0)
                months_to_target = row.get("estimated_months_to_target") or metadata.get("forecast_estimated_months_to_target")
                confidence_score = int(row.get("confidence_score") or metadata.get("forecast_confidence_score") or 0)
                demand_score = int(row.get("market_demand_score") or 0)
                price = LegoHunterTelegramBot._fmt_eur(row.get("current_price"))
                eol = html.escape(LegoHunterTelegramBot._format_eol_date(row.get("eol_date_prediction")))

                lines.append(f"{badge} <b>{idx}) {set_name}</b> ({set_id})")
                lines.append(
                    f"Score {composite}/100 | AI {ai_score}/100 | Quant {quant_score}/100 | Demand {demand_score}/100 | Pattern {pattern_score}/100"
                )
                target_line = f"Target {int(months_to_target)} mesi" if months_to_target is not None else "Target n/d"
                lines.append(
                    f"Prob Upside 12m {prob_12m:.1f}% | ROI atteso 12m {roi_12m:+.1f}% | {target_line} | EOL {eol} | Conf {confidence_score}/100"
                )
                historical_sample_size = int(
                    row.get("historical_sample_size")
                    or metadata.get("historical_sample_size")
                    or 0
                )
                historical_win_rate = row.get("historical_win_rate_12m_pct")
                if historical_win_rate is None:
                    historical_win_rate = metadata.get("historical_win_rate_12m_pct")
                historical_prior_score = row.get("historical_prior_score")
                if historical_prior_score is None:
                    historical_prior_score = metadata.get("historical_prior_score")
                historical_support_confidence = row.get("historical_support_confidence")
                if historical_support_confidence is None:
                    historical_support_confidence = metadata.get("historical_support_confidence")

                if historical_sample_size > 0:
                    try:
                        historical_win_rate_value = float(historical_win_rate or 0.0)
                    except (TypeError, ValueError):
                        historical_win_rate_value = 0.0
                    try:
                        historical_prior_value = int(historical_prior_score or 0)
                    except (TypeError, ValueError):
                        historical_prior_value = 0
                    try:
                        historical_support_value = int(historical_support_confidence or 0)
                    except (TypeError, ValueError):
                        historical_support_value = 0
                    lines.append(
                        "Storico: "
                        f"{historical_sample_size} campioni | "
                        f"Win-rate 12m {historical_win_rate_value:.1f}% | "
                        f"Prior {historical_prior_value}/100 | "
                        f"Supporto {historical_support_value}/100"
                    )
                if pattern_summary:
                    lines.append(f"Pattern: {html.escape(pattern_summary)}")
                lines.append(f"Fonte: {source} | Segnale: {strength}")
                lines.append(LegoHunterTelegramBot._format_pick_link(row))
                if strength == "HIGH_CONFIDENCE" and bootstrap_enabled:
                    data_points = int(row.get("forecast_data_points") or metadata.get("forecast_data_points") or 0)
                    if bootstrap_min_history_points > 0 and 0 < data_points < bootstrap_min_history_points:
                        lines.append(
                            "Nota: HIGH_CONFIDENCE in bootstrap "
                            f"(data points {data_points} inferiori a {bootstrap_min_history_points}; "
                            f"soglie bootstrap Prob minimo {bootstrap_min_probability_pct:.0f}% "
                            f"e Conf minima {bootstrap_min_confidence})."
                        )
                if strength == "LOW_CONFIDENCE":
                    risk_note = str(row.get("risk_note") or "Conferma manuale consigliata.")
                    lines.append(f"Nota: {html.escape(risk_note)}")
                lines.append("")
        else:
            lines.append("🟠 Nessun candidato utile in questo ciclo.")

        source_raw = diagnostics.get("source_raw_counts") or {}
        source_strategy = str(diagnostics.get("source_strategy") or "n/d")
        selected_source = str(diagnostics.get("selected_source") or "none")
        lines.append("<b>Diagnostica Discovery</b>")
        ai_engine = str(ai_runtime.get("engine") or "unknown")
        ai_model = str(ai_runtime.get("model") or "unknown")
        ai_mode = str(ai_runtime.get("mode") or "unknown")
        lines.append(f"🤖 IA: {html.escape(ai_engine)} | modello: <code>{html.escape(ai_model)}</code> | mode: {html.escape(ai_mode)}")
        lines.append(
            f"🧭 Pipeline fonti: <code>{html.escape(source_strategy)}</code> | attiva: <code>{html.escape(selected_source)}</code>"
        )
        lines.append(
            "Proxy LEGO/Amazon: "
            f"{int(source_raw.get('lego_proxy_reader', 0))}/"
            f"{int(source_raw.get('amazon_proxy_reader', 0))} | "
            "Playwright LEGO/Amazon: "
            f"{int(source_raw.get('lego_retiring', 0))}/"
            f"{int(source_raw.get('amazon_bestsellers', 0))} | "
            "HTTP fallback LEGO/Amazon: "
            f"{int(source_raw.get('lego_http_fallback', 0))}/"
            f"{int(source_raw.get('amazon_http_fallback', 0))} | "
            f"Dedup: {int(diagnostics.get('dedup_candidates', 0))}"
        )
        lines.append(
            f"Soglia composita: {int(diagnostics.get('threshold', 0))} | "
            f"Min Prob: {float(diagnostics.get('min_probability_high_confidence', 0.0)) * 100.0:.0f}% | "
            f"Min Conf: {int(diagnostics.get('min_confidence_score_high_confidence', 0))} | "
            f"Sopra soglia: {int(diagnostics.get('above_threshold_count', 0))} | "
            f"Max AI: {int(diagnostics.get('max_ai_score', 0))} | "
            f"Max Score: {int(diagnostics.get('max_composite_score', 0))} | "
            f"Max Prob12m: {float(diagnostics.get('max_probability_upside_12m', 0.0)):.1f}%"
        )
        if diagnostics.get("historical_high_conf_required"):
            eff_hist_samples = int(
                diagnostics.get("historical_high_conf_effective_min_samples")
                or diagnostics.get("historical_high_conf_min_samples")
                or 0
            )
            eff_hist_win_rate = float(
                diagnostics.get("historical_high_conf_effective_min_win_rate_pct")
                or diagnostics.get("historical_high_conf_min_win_rate_pct")
                or 0.0
            )
            eff_hist_support = int(
                diagnostics.get("historical_high_conf_effective_min_support_confidence")
                or diagnostics.get("historical_high_conf_min_support_confidence")
                or 0
            )
            eff_hist_prior = int(
                diagnostics.get("historical_high_conf_effective_min_prior_score")
                or diagnostics.get("historical_high_conf_min_prior_score")
                or 0
            )
            adaptive_hist_active = bool(diagnostics.get("adaptive_historical_thresholds_active"))
            adaptive_badge = "adattive" if adaptive_hist_active else "statiche"
            lines.append(
                f"📚 Gate storico ({adaptive_badge}): campioni>={eff_hist_samples} | "
                f"Win-rate>={eff_hist_win_rate:.0f}% | Supporto>={eff_hist_support} | Prior>={eff_hist_prior}"
            )
        historical_quality = diagnostics.get("historical_quality")
        if isinstance(historical_quality, dict) and historical_quality:
            quality_tier = str(historical_quality.get("tier") or "n/d").upper()
            degraded = bool(historical_quality.get("degraded"))
            median_age = historical_quality.get("median_age_years")
            theme_count = historical_quality.get("theme_count")
            median_age_label = str(median_age) if median_age is not None else "n/d"
            theme_count_label = str(theme_count) if theme_count is not None else "n/d"
            max_age = (
                historical_quality.get("guards", {}).get("max_median_age_years")
                if isinstance(historical_quality.get("guards"), dict)
                else None
            )
            max_age_label = str(max_age) if max_age is not None else "n/d"
            if degraded:
                lines.append(
                    "🧱 Seed storico: "
                    f"{quality_tier} (mediana eta' {median_age_label}y/{max_age_label}y, temi {theme_count_label}) "
                    "-> quality-aware gate attivo."
                )
            else:
                lines.append(
                    "🧱 Seed storico: "
                    f"{quality_tier} (mediana eta' {median_age_label}y, temi {theme_count_label})."
                )
        if bootstrap_enabled:
            bootstrap_rows_count = int(diagnostics.get("bootstrap_rows_count") or 0)
            bootstrap_status = "attivo" if bootstrap_rows_count > 0 else "abilitato (non attivo nel ciclo)"
            lines.append(
                f"🧪 Bootstrap soglie: {bootstrap_status} | "
                f"Min Prob {bootstrap_min_probability_pct:.0f}% | "
                f"Min Conf {bootstrap_min_confidence} | "
                f"Set bootstrap: {bootstrap_rows_count}"
            )

        threshold_profile = diagnostics.get("threshold_profile")
        if isinstance(threshold_profile, dict):
            profile_source = html.escape(str(threshold_profile.get("source") or "n/d"))
            lines.append(f"🎛️ Profilo soglie: <code>{profile_source}</code>")

        backtest_runtime = diagnostics.get("backtest_runtime")
        if isinstance(backtest_runtime, dict) and backtest_runtime:
            status = str(backtest_runtime.get("status") or "n/d")
            if status == "ok":
                lines.append(
                    "📈 Backtest: "
                    f"sample {int(backtest_runtime.get('sample_size', 0))} | "
                    f"P@K {float(backtest_runtime.get('precision_at_k', 0.0)):.2f} | "
                    f"Precision {float(backtest_runtime.get('precision', 0.0)):.2f} | "
                    f"Brier {float(backtest_runtime.get('brier_score', 0.0)):.2f}"
                )
            elif status == "insufficient_data":
                lines.append(
                    "📈 Backtest: dati insufficienti "
                    f"({int(backtest_runtime.get('sample_size', 0))}/"
                    f"{int(backtest_runtime.get('required', 0))})"
                )

        if diagnostics.get("fallback_source_used"):
            lines.append("🛟 Fallback sorgente HTTP attivo (scraper primari a zero).")

        failures = list(diagnostics.get("source_failures") or [])
        if failures:
            compact = "; ".join(str(item) for item in failures[:2])
            if len(failures) > 2:
                compact += f" (+{len(failures) - 2} altri)"
            lines.append(f"⚠️ Fonti con errori: {html.escape(compact)}")

        if diagnostics.get("anti_bot_alert"):
            anti_bot_message = str(
                diagnostics.get("anti_bot_message")
                or "Entrambe le fonti discovery sono a zero: possibile anti-bot."
            )
            lines.append(f"🚨 {html.escape(anti_bot_message)}")

        return lines


def build_application(
    manager: LegoHunterTelegramBot,
    token: str,
    *,
    register_commands_on_init: bool = True,
) -> Application:
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", manager.cmd_start))
    app.add_handler(CommandHandler("help", manager.cmd_help))

    # New LEGO-focused commands.
    app.add_handler(CommandHandler("scova", manager.cmd_scova))
    app.add_handler(CommandHandler("radar", manager.cmd_radar))
    app.add_handler(CommandHandler("cerca", manager.cmd_cerca))
    app.add_handler(CommandHandler("offerte", manager.cmd_offerte))
    app.add_handler(CommandHandler("collezione", manager.cmd_collezione))
    app.add_handler(CommandHandler("vendi", manager.cmd_vendi))

    # Backward-compatible aliases.
    app.add_handler(CommandHandler("hunt", manager.cmd_scova))
    app.add_handler(CommandHandler("portfolio", manager.cmd_collezione))
    app.add_handler(CommandHandler("sell_signal", manager.cmd_vendi))

    if register_commands_on_init:
        async def _post_init(application: Application) -> None:
            await manager.register_commands(application)

        app.post_init = _post_init
    return app


async def run_scheduled_cycle(
    *,
    token: str,
    chat_id: str,
    oracle: DiscoveryOracle,
    repository: LegoHunterRepository,
    fiscal_guardian: FiscalGuardian,
) -> None:
    bot = Bot(token=token)
    LOGGER.info("Scheduled cycle started | chat_id_set=%s", bool(chat_id))
    try:
        await _telegram_call_with_retry(
            operation_name="bot.set_my_commands",
            fn=lambda: bot.set_my_commands(LegoHunterTelegramBot.supported_commands()),
            max_attempts=3,
            non_fatal=True,
        )
        LOGGER.info("Scheduled cycle command sync attempted")

        report = await oracle.discover_with_diagnostics(
            persist=True,
            top_limit=12,
            fallback_limit=3,
        )
        diagnostics = report.get("diagnostics") or {}
        LOGGER.info(
            "Scheduled discovery diagnostics | strategy=%s selected_source=%s raw=%s dedup=%s ranked=%s above_threshold=%s score_fallback=%s source_fallback=%s anti_bot=%s ai=%s",
            diagnostics.get("source_strategy"),
            diagnostics.get("selected_source"),
            diagnostics.get("source_raw_counts"),
            diagnostics.get("dedup_candidates"),
            diagnostics.get("ranked_candidates"),
            diagnostics.get("above_threshold_count"),
            diagnostics.get("fallback_used"),
            diagnostics.get("fallback_source_used"),
            diagnostics.get("anti_bot_alert"),
            diagnostics.get("ai_runtime"),
        )
        secondary_candidates = list(report.get("selected") or [])[:6]
        if not secondary_candidates:
            secondary_candidates = repository.get_top_opportunities(6, 50)
        secondary_refreshed = 0
        if PLAYWRIGHT_AVAILABLE and secondary_candidates:
            try:
                secondary_rows = await validate_secondary_deals_with_scrapers(
                    repository=repository,
                    opportunities=secondary_candidates,
                )
                secondary_refreshed = sum(
                    1 for row in secondary_rows if row.get("secondary_best_price") is not None
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Secondary cache refresh failed: %s", exc)
        LOGGER.info(
            "Secondary cache refresh | playwright=%s candidates=%s refreshed=%s",
            PLAYWRIGHT_AVAILABLE,
            len(secondary_candidates),
            secondary_refreshed,
        )
        lines = [
            "<b>🧱 LEGO HUNTER</b> <i>Update automatico (ogni 6 ore)</i>",
            "",
        ]
        lines.extend(LegoHunterTelegramBot._format_discovery_report(report, top_limit=3))

        safety = fiscal_guardian.check_safety_status()
        status = str(safety.get("status") or "UNKNOWN")
        status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(status, "⚪")
        fiscal_message = html.escape(str(safety.get("message") or "n/d"))
        lines.append("")
        lines.append("<b>Fiscal Guard</b>")
        lines.append(f"{status_emoji} DAC7 {status} | {fiscal_message}")

        holdings = repository.get_portfolio("holding")
        lines.append(f"📦 Set in collezione: <b>{len(holdings)}</b>")

        payload = "\n".join(lines)
        LOGGER.info(
            "Sending scheduled Telegram report | lines=%s chars=%s holdings=%s",
            len(lines),
            len(payload),
            len(holdings),
        )
        send_result = await _telegram_call_with_retry(
            operation_name="bot.send_message",
            fn=lambda: bot.send_message(
                chat_id=chat_id,
                text=payload,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                connect_timeout=15,
                read_timeout=45,
                write_timeout=30,
                pool_timeout=15,
            ),
            max_attempts=4,
            non_fatal=False,
            retry_timeouts=False,
            timeout_assume_delivered=True,
        )
        if isinstance(send_result, dict) and send_result.get("delivery") == "uncertain_timeout_assumed":
            LOGGER.warning("Scheduled Telegram report delivery uncertain (timeout); duplicate-safe mode prevented retries")
        else:
            LOGGER.info("Scheduled Telegram report sent successfully")
    except RetryAfter as exc:
        LOGGER.error("Telegram flood-control while sending scheduled report: %s", exc)
        raise
    except TelegramError as exc:
        LOGGER.exception("Telegram error during scheduled report")
        raise
    except Exception:
        LOGGER.exception("Scheduled cycle failed")
        raise
    finally:
        try:
            await bot.shutdown()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Bot shutdown warning (non-fatal): %s", exc)
        LOGGER.info("Scheduled cycle finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lego Hunter Telegram Bot")
    parser.add_argument(
        "--mode",
        choices=["polling", "scheduled"],
        default=os.getenv("RUN_MODE", "polling"),
        help="polling = interactive telegram bot, scheduled = one-shot cycle for GitHub Actions",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    LOGGER.info("Application start | mode=%s", args.mode)

    telegram_token = os.getenv("TELEGRAM_TOKEN")
    if not telegram_token:
        raise RuntimeError("Missing TELEGRAM_TOKEN")

    repository = LegoHunterRepository.from_env()
    fiscal_guardian = FiscalGuardian(repository)

    if args.mode == "scheduled":
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id:
            raise RuntimeError("Missing TELEGRAM_CHAT_ID for scheduled mode")
        oracle = DiscoveryOracle(repository=repository)
        await run_scheduled_cycle(
            token=telegram_token,
            chat_id=chat_id,
            oracle=oracle,
            repository=repository,
            fiscal_guardian=fiscal_guardian,
        )
        return

    manager = LegoHunterTelegramBot(
        repository=repository,
        oracle=None,
        oracle_factory=lambda: DiscoveryOracle(repository=repository),
        fiscal_guardian=fiscal_guardian,
        allowed_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    )

    app = build_application(manager, telegram_token)
    await app.initialize()
    await manager.register_commands(app)
    await app.start()
    await app.updater.start_polling(drop_pending_updates=False)

    LOGGER.info("Lego Hunter bot running in polling mode")
    await asyncio.Event().wait()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        LOGGER.info("Bot stopped")


if __name__ == "__main__":
    main()
