# Lego Hunter

Bot Telegram autonomo per discovery e monitoraggio opportunita LEGO con data moat su Supabase.

## Funzioni principali
- Discovery ogni ora (pipeline cloud-first)
- Sorgente primaria discovery: reader esterno cloud (`external_first`)
- Fallback automatici: Playwright, poi HTTP parser
- Ranking AI multi-provider: Gemini primario + OpenRouter fallback + heuristic finale
- Selezione automatica provider/modello: inventory completo modelli text-capable free-tier + quota-check + scelta del migliore disponibile
- Garanzia runtime: se Gemini e OpenRouter non hanno quota/API disponibili, passa a `heuristic-ai-v2` (fallback sempre operativo)
- Ranking predittivo ibrido: `Composite Score` = AI + domanda + forecast quantitativo su serie storiche
- Forecast quantitativo: probabilita upside 12 mesi, ROI atteso, intervallo di confidenza, tempo stimato al target ROI
- Definizione `HIGH_CONFIDENCE`: score sopra soglia + probabilita minima + confidenza dati minima + AI non in fallback
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
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY` (fallback AI provider)
- `GEMINI_MODEL` (opzionale: modello preferito; il bot fa comunque auto-detect/failover)
- `OPENROUTER_MODEL` (opzionale: modello OpenRouter preferito)
- `OPENROUTER_API_BASE` (opzionale, default `https://openrouter.ai/api/v1`)
- `DISCOVERY_SOURCE_MODE` (opzionale, default `external_first`; valori: `external_first`, `playwright_first`, `external_only`)
- `MIN_COMPOSITE_SCORE` (opzionale, default `60`)
- `MIN_UPSIDE_PROBABILITY` (opzionale, default `0.60`, scala 0-1)
- `MIN_CONFIDENCE_SCORE` (opzionale, default `68`)
- `HISTORY_WINDOW_DAYS` (opzionale, default `180`)
- `TARGET_ROI_PCT` (opzionale, default `30`)

## Database
Eseguire lo script SQL:
- `supabase_schema.sql`
