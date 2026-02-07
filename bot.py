from __future__ import annotations

import argparse
import asyncio
import html
import logging
import os
from typing import Any, Optional
from urllib.parse import quote_plus

from telegram import Bot, BotCommand, Update
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes

from fiscal import FiscalGuardian
from models import LegoHunterRepository
from oracle import DiscoveryOracle

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("lego_hunter.bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class LegoHunterTelegramBot:
    def __init__(
        self,
        repository: LegoHunterRepository,
        oracle: DiscoveryOracle,
        fiscal_guardian: FiscalGuardian,
        *,
        allowed_chat_id: Optional[str] = None,
    ) -> None:
        self.repository = repository
        self.oracle = oracle
        self.fiscal_guardian = fiscal_guardian
        self.allowed_chat_id = str(allowed_chat_id) if allowed_chat_id else None

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
            "/vendi - Segnali vendita con ROI netto > 30% (bloccati automaticamente se DAC7 a rischio).\n\n"
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
        try:
            report = await self.oracle.discover_with_diagnostics(
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
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        lines = ["🧱 <b>Discovery LEGO</b>"]
        lines.extend(self._format_discovery_report(report, top_limit=3))
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

        await update.message.reply_text("Verifica offerte secondarie (Vinted/Subito) in corso...")

        try:
            opportunities = await asyncio.to_thread(self.repository.get_top_opportunities, 6, 50)
            if not opportunities:
                opportunities = await self.oracle.discover_opportunities(persist=True, top_limit=6)
            deals = await self.oracle.validate_secondary_deals(opportunities[:6])
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Deal validation failed")
            await update.message.reply_text(f"Errore verifica offerte: {exc}")
            return

        good_deals = [row for row in deals if (row.get("discount_vs_primary_pct") or 0) >= 10]
        if not good_deals:
            await update.message.reply_text("Nessuna offerta >=10% di sconto trovata ora.")
            return

        lines = ["Offerte LEGO interessanti:"]
        for row in good_deals[:5]:
            lines.append(
                f"- {row.get('set_name')} ({row.get('set_id')})\n"
                f"Secondario: {self._fmt_eur(row.get('secondary_best_price'))} su {row.get('secondary_platform')}\n"
                f"Sconto vs prezzo primario: {row.get('discount_vs_primary_pct')}%"
            )
        await update.message.reply_text("\n\n".join(lines), disable_web_page_preview=True)

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
            holdings = await asyncio.to_thread(self.repository.get_portfolio, "holding")
        except Exception as exc:  # noqa: BLE001
            await update.message.reply_text(f"Errore portfolio: {exc}")
            return

        signals: list[dict[str, Any]] = []
        estimated_shipping_out = 7.0

        for row in holdings:
            qty = int(row.get("quantity") or 1)
            buy_price = float(row.get("purchase_price") or 0.0)
            ship_in = float(row.get("shipping_in_cost") or 0.0)
            total_cost = (buy_price + ship_in) * qty
            if total_cost <= 0:
                continue

            latest = await asyncio.to_thread(self.repository.get_latest_price, row.get("set_id"), "vinted")
            if not latest:
                latest = await asyncio.to_thread(self.repository.get_best_secondary_price, row.get("set_id"))
            if not latest:
                continue

            sale_unit = float(latest.get("price") or 0.0)
            if sale_unit <= 0:
                continue

            net_sale_total = (sale_unit - estimated_shipping_out) * qty
            roi_net = ((net_sale_total - total_cost) / total_cost) * 100
            if roi_net > 30:
                signals.append(
                    {
                        "set_name": row.get("set_name"),
                        "set_id": row.get("set_id"),
                        "platform": latest.get("platform"),
                        "sale_unit": sale_unit,
                        "roi_net": roi_net,
                    }
                )

        if not signals:
            await update.message.reply_text("Nessun set supera ROI netto > 30% al momento.")
            return

        signals.sort(key=lambda row: row["roi_net"], reverse=True)
        lines = ["Segnali uscita (ROI netto > 30%):"]
        for row in signals[:10]:
            lines.append(
                f"- {row['set_name']} ({row['set_id']}) | {row['platform']} {self._fmt_eur(row['sale_unit'])} | "
                f"ROI netto {row['roi_net']:.1f}%"
            )

        await update.message.reply_text("\n".join(lines))

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

        lines: list[str] = []
        if diagnostics.get("fallback_used"):
            lines.append("⚠️ Nessun set sopra soglia: mostro i migliori <b>LOW_CONFIDENCE</b>.")
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
                eol = html.escape(str(row.get("eol_date_prediction") or "n/d"))

                lines.append(f"{badge} <b>{idx}) {set_name}</b> ({set_id})")
                lines.append(
                    f"Score {composite}/100 | AI {ai_score}/100 | Quant {quant_score}/100 | Demand {demand_score}/100"
                )
                target_line = f"Target {int(months_to_target)} mesi" if months_to_target is not None else "Target n/d"
                lines.append(
                    f"Prob Upside 12m {prob_12m:.1f}% | ROI atteso 12m {roi_12m:+.1f}% | {target_line} | Conf {confidence_score}/100"
                )
                lines.append(f"Fonte: {source} | Segnale: {strength}")
                lines.append(LegoHunterTelegramBot._format_pick_link(row))
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


def build_application(manager: LegoHunterTelegramBot, token: str) -> Application:
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
        await bot.set_my_commands(LegoHunterTelegramBot.supported_commands())
        LOGGER.info("Scheduled cycle commands synced with Telegram")

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
        lines = [
            "<b>🧱 LEGO HUNTER</b> <i>Update automatico (ogni ora)</i>",
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
        await bot.send_message(
            chat_id=chat_id,
            text=payload,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
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
    oracle = DiscoveryOracle(repository=repository)
    fiscal_guardian = FiscalGuardian(repository)

    if args.mode == "scheduled":
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id:
            raise RuntimeError("Missing TELEGRAM_CHAT_ID for scheduled mode")
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
        oracle=oracle,
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
