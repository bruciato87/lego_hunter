# Lego Hunter

Bot Telegram autonomo per discovery e monitoraggio opportunita LEGO con data moat su Supabase.

## Funzioni principali
- Discovery ogni 6 ore (pipeline cloud-first)
- Sorgente primaria discovery: reader esterno cloud (`external_first`)
- Fallback automatici: Playwright, poi HTTP parser
- Ranking AI multi-provider: Gemini primario + OpenRouter fallback + heuristic finale
- Selezione automatica provider/modello: inventory completo modelli text-capable free-tier + quota-check + scelta del migliore disponibile
- Garanzia runtime: se Gemini e OpenRouter non hanno quota/API disponibili, passa a `heuristic-ai-v2` (fallback sempre operativo)
- Ranking predittivo ibrido: `Composite Score` = AI + domanda + forecast quantitativo su serie storiche
- Forecast quantitativo: probabilita upside 12 mesi, ROI atteso, intervallo di confidenza, tempo stimato al target ROI
- Definizione `HIGH_CONFIDENCE`: score sopra soglia + probabilita minima + confidenza dati minima + AI non in fallback
- Backtesting walk-forward su Data Moat: precision@k, coverage, calibrazione probabilistica (Brier score)
- Auto-tuning soglie opzionale: regola automaticamente `MIN_*` su storico reale (senza hardcode)
- Validazione secondario (Vinted/Subito)
- Guardia fiscale DAC7 (blocco segnali vendita vicino soglia)
- Comandi Telegram orientati agli oggetti LEGO: `/scova`, `/radar`, `/cerca`, `/offerte`, `/collezione`, `/vendi`

## Setup locale
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
python -m unittest discover -s tests -v
```

## Esecuzione
```bash
# modalità bot interattiva
python bot.py --mode polling

# ciclo schedulato one-shot (usato da GitHub Actions)
python bot.py --mode scheduled
```

## Variabili ambiente richieste
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TELEGRAM_WEBHOOK_SECRET` (obbligatoria per endpoint webhook su Vercel)
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY` (fallback AI provider)
- `GEMINI_MODEL` (opzionale: modello preferito; il bot fa comunque auto-detect/failover)
- `OPENROUTER_MODEL` (opzionale: modello OpenRouter preferito)
- `OPENROUTER_API_BASE` (opzionale, default `https://openrouter.ai/api/v1`)
- `GEMINI_FREE_TIER_ONLY` (opzionale, default `true`: usa solo candidati Gemini flash/lite, esclude pro/ultra)
- `OPENROUTER_FREE_TIER_ONLY` (opzionale, default `true`: usa solo modelli OpenRouter con suffisso `:free`)
- `DISCOVERY_SOURCE_MODE` (opzionale, default `external_first`; valori: `external_first`, `playwright_first`, `external_only`)
- `MIN_COMPOSITE_SCORE` (opzionale, default `60`)
- `MIN_UPSIDE_PROBABILITY` (opzionale, default `0.60`, scala 0-1)
- `MIN_CONFIDENCE_SCORE` (opzionale, default `68`)
- `HISTORY_WINDOW_DAYS` (opzionale, default `180`)
- `TARGET_ROI_PCT` (opzionale, default `30`)
- `AUTO_TUNE_THRESHOLDS` (opzionale, default `false`)
- `BACKTEST_LOOKBACK_DAYS` (opzionale, default `365`)
- `BACKTEST_HORIZON_DAYS` (opzionale, default `180`)
- `BACKTEST_MIN_SELECTED` (opzionale, default `15`)
- `HISTORICAL_REFERENCE_ENABLED` (opzionale, default `true`)
- `HISTORICAL_REFERENCE_CASES_PATH` (opzionale, default `data/historical_seed/historical_reference_cases.csv`)
- `HISTORICAL_REFERENCE_MIN_SAMPLES` (opzionale, default `24`)
- `HISTORICAL_PRIOR_WEIGHT` (opzionale, default `0.10`, range `0.0-0.35`)
- `HISTORICAL_PRICE_BAND_TOLERANCE` (opzionale, default `0.45`)
- `WEBHOOK_BASE_URL` (opzionale per script setup webhook, es. `https://lego-hunter.vercel.app`)

## Deploy Vercel (comandi Telegram live)
- Questo repo usa:
  - GitHub Actions per il ciclo schedulato ogni 6 ore (`--mode scheduled`)
  - Vercel webhook per i comandi Telegram in tempo reale
- Endpoint webhook: `/api/telegram_webhook` (alias `/telegram/webhook`)
- Health check: `/healthz`

Configurazione webhook Telegram:
```bash
python scripts/configure_telegram_webhook.py --base-url https://<tuo-progetto>.vercel.app
```

Alternativa cloud-only (senza locale):
- workflow GitHub `telegram-webhook` (`workflow_dispatch`)
- input `base_url` oppure variabile repo `WEBHOOK_BASE_URL`
- secrets richiesti: `TELEGRAM_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`

Verifica rapida:
```bash
curl -s https://<tuo-progetto>.vercel.app/healthz
```

## Database
Eseguire lo script SQL:
- `supabase_schema.sql`

## Seed storico (Data Moat bootstrap)
- Il ranking usa un prior storico da `data/historical_seed/historical_reference_cases.csv`.
- Il prior non sostituisce i dati live: viene usato come fattore additivo controllato (`HISTORICAL_PRIOR_WEIGHT`) per migliorare la stabilita' dei punteggi nei primi mesi.
- Script di rigenerazione seed: `scripts/build_historical_reference_cases.py`.
- Lo script legge il dataset raw ZIP in `data/historical_seed/raw/` (non versionato) e produce il CSV finale versionato.
