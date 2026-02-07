# Lego Hunter

Bot Telegram autonomo per discovery e monitoraggio opportunita LEGO con data moat su Supabase.

## Funzioni principali
- Discovery ogni ora (pipeline cloud-first)
- Sorgente primaria discovery: reader esterno cloud (`external_first`)
- Fallback automatici: Playwright, poi HTTP parser
- Ranking AI con Gemini (fallback euristico se API non disponibile)
- Selezione automatica Gemini: inventory completo dei modelli text-capable + quota-check + scelta del migliore disponibile
- Garanzia runtime: se nessun modello Gemini ha quota/API disponibili, passa a `heuristic-ai-v2` (fallback sempre operativo)
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
- `GEMINI_MODEL` (opzionale: modello preferito; il bot fa comunque auto-detect/failover)
- `GEMINI_MAX_PROBE_MODELS` (opzionale, default `6`)
- `DISCOVERY_SOURCE_MODE` (opzionale, default `external_first`; valori: `external_first`, `playwright_first`, `external_only`)

## Database
Eseguire lo script SQL:
- `supabase_schema.sql`
