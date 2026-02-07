from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any, Optional

from telegram import Bot, BotCommand, Update
from telegram.constants import ParseMode
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

    async def register_commands(self, app: Application) -> None:
        commands = [
            BotCommand("scova", "Discovery completa: trova set LEGO promettenti"),
            BotCommand("radar", "Mostra top opportunita' scoperte"),
            BotCommand("cerca", "Cerca nel radar: /cerca 75367"),
            BotCommand("offerte", "Confronta prezzo ufficiale vs secondario"),
            BotCommand("collezione", "Valore attuale portfolio LEGO"),
            BotCommand("vendi", "Segnali uscita con ROI netto > 30%"),
            BotCommand("help", "Lista comandi disponibili"),
        ]
        await app.bot.set_my_commands(commands)

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
            "Comandi disponibili:\n"
            "/scova - Discovery automatica e top picks\n"
            "/radar - Opportunita' attuali dal Data Moat\n"
            "/cerca <testo> - Cerca set (ID, nome, tema)\n"
            "/offerte - Verifica sconti Vinted/Subito\n"
            "/collezione - Valore e ROI latente dei set posseduti\n"
            "/vendi - Candidati vendita con ROI netto > 30% (DAC7-safe)",
        )

    async def cmd_scova(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        await update.message.reply_text("Scansione in corso: Lego Retiring Soon + Amazon + ranking AI...")
        try:
            opportunities = await self.oracle.discover_opportunities(persist=True, top_limit=20)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Discovery failed")
            await update.message.reply_text(f"Errore discovery: {exc}")
            return

        if not opportunities:
            await update.message.reply_text("Nessuna opportunita' valida trovata in questo ciclo.")
            return

        top = opportunities[:3]
        lines = ["Top Picks LEGO di oggi:"]
        for idx, row in enumerate(top, start=1):
            lines.append(
                f"{idx}. {row.get('set_name')} ({row.get('set_id')})\n"
                f"AI {row.get('ai_investment_score')}/100 | Demand {row.get('market_demand_score')}/100\n"
                f"Prezzo: {self._fmt_eur(row.get('current_price'))} | EOL: {row.get('eol_date_prediction') or 'n/d'}"
            )

        await update.message.reply_text("\n\n".join(lines))

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
            lines.append(
                f"{idx}. {row.get('set_name')} ({row.get('set_id')}) - "
                f"AI {row.get('ai_investment_score')}/100 | Demand {row.get('market_demand_score')}/100"
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
            lines.append(
                f"- {row.get('set_name')} ({row.get('set_id')}) | Theme: {row.get('theme') or 'n/d'} | "
                f"AI {row.get('ai_investment_score')}/100"
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
    try:
        opportunities = await oracle.discover_opportunities(persist=True, top_limit=12)
        top = opportunities[:3]

        if top:
            lines = ["[LEGO HUNTER] Aggiornamento automatico (6h)"]
            for idx, row in enumerate(top, start=1):
                lines.append(
                    f"{idx}) {row.get('set_name')} ({row.get('set_id')}) - AI {row.get('ai_investment_score')}/100 - "
                    f"Prezzo {LegoHunterTelegramBot._fmt_eur(row.get('current_price'))}"
                )
        else:
            lines = ["[LEGO HUNTER] Nessuna nuova opportunita' sopra soglia in questo ciclo."]

        safety = fiscal_guardian.check_safety_status()
        lines.append(f"Stato DAC7: {safety.get('status')} | {safety.get('message')}")

        holdings = repository.get_portfolio("holding")
        lines.append(f"Set in collezione: {len(holdings)}")

        await bot.send_message(
            chat_id=chat_id,
            text="\n".join(lines),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    finally:
        await bot.close()


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
