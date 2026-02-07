from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DEFAULT_ZIP_PATH = Path("data/historical_seed/raw/lego_secondary_market_v9hhs66vm3.zip")
DEFAULT_OUTPUT_PATH = Path("data/historical_seed/historical_reference_cases.csv")

SMALL_SAMPLE_FILE = "DiB small sample Dec 2015 - Apr 2019 prices.xlsx"
WHOLE_SAMPLE_FILE = "DiB whole sample Jan 2018 - Apr 2019 prices.xlsx"
DEC2015_FILE = "DiB whole sample Dec 2015 prices.xlsx"


@dataclass
class CaseRow:
    set_id: str
    set_number: str
    set_name: str
    theme: str
    release_year: Optional[int]
    msrp_usd: Optional[float]
    start_date: str
    end_date: str
    observation_months: int
    start_price_usd: float
    price_12m_usd: Optional[float]
    price_24m_usd: Optional[float]
    roi_12m_pct: Optional[float]
    roi_24m_pct: Optional[float]
    annualized_roi_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    win_12m: Optional[int]
    win_24m: Optional[int]
    source_dataset: str
    pattern_tags: str


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _safe_int(value: object) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _to_date_label(value: object) -> Optional[str]:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return None
    return None


def _extract_monthly_series(row: pd.Series, monthly_columns: list[object]) -> list[tuple[str, float]]:
    series: list[tuple[str, float]] = []
    for col in monthly_columns:
        date_label = _to_date_label(col)
        if date_label is None:
            continue
        price = _safe_float(row.get(col))
        if price is None or price <= 0:
            continue
        series.append((date_label, round(price, 4)))
    return series


def _pct_return(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None or start <= 0:
        return None
    return (end - start) / start * 100.0


def _max_drawdown_pct(prices: list[float]) -> Optional[float]:
    if not prices:
        return None
    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        peak = max(peak, price)
        if peak <= 0:
            continue
        drawdown = (price - peak) / peak
        max_dd = min(max_dd, drawdown)
    return max_dd * 100.0


def _pattern_tags(theme: str, set_name: str) -> list[str]:
    tags: list[str] = []
    text = f"{theme} {set_name}".lower()

    if any(k in text for k in ("star wars", "marvel", "harry potter", "batman", "jurassic", "disney")):
        tags.append("franchise_power")
    if any(k in text for k in ("modular", "collection", "helmet", "diorama", "botanical", "series", "casco")):
        tags.append("series_completism")
    if any(k in text for k in ("18+", "ideas", "art", "architecture", "display", "icons")):
        tags.append("adult_display")
    if any(k in text for k in ("limited", "exclusive", "gwp", "gift with purchase", "rare")):
        tags.append("scarcity")
    if any(k in text for k in ("nasa", "apollo", "space", "lunar", "moon", "rover")):
        tags.append("stem_icon")
    if not tags:
        tags.append("general_collectible")
    return tags


def _build_case_from_row(
    row: pd.Series,
    *,
    monthly_columns: list[object],
    source_dataset: str,
    set_id_col: str,
    set_name_col: str,
    theme_col: str,
    release_year_col: str,
    set_number_col: Optional[str] = None,
    msrp_col: Optional[str] = None,
) -> Optional[CaseRow]:
    set_id = _normalize_text(row.get(set_id_col))
    if not set_id:
        return None

    set_name = _normalize_text(row.get(set_name_col))
    theme = _normalize_text(row.get(theme_col)) or "Unknown"
    release_year = _safe_int(row.get(release_year_col))
    msrp = _safe_float(row.get(msrp_col)) if msrp_col else None
    set_number = _normalize_text(row.get(set_number_col)) if set_number_col else ""

    monthly = _extract_monthly_series(row, monthly_columns)
    if len(monthly) < 6:
        return None

    start_date, start_price = monthly[0]
    end_date, _end_price = monthly[-1]
    prices_only = [price for _d, price in monthly]

    price_12m = monthly[12][1] if len(monthly) > 12 else None
    price_24m = monthly[24][1] if len(monthly) > 24 else None

    roi_12m = _pct_return(start_price, price_12m)
    roi_24m = _pct_return(start_price, price_24m)

    annualized = None
    if roi_24m is not None:
        annualized = ((1.0 + (roi_24m / 100.0)) ** 0.5 - 1.0) * 100.0

    tags = _pattern_tags(theme=theme, set_name=set_name)

    return CaseRow(
        set_id=set_id,
        set_number=set_number,
        set_name=set_name,
        theme=theme,
        release_year=release_year,
        msrp_usd=msrp,
        start_date=start_date,
        end_date=end_date,
        observation_months=len(monthly),
        start_price_usd=round(start_price, 4),
        price_12m_usd=round(price_12m, 4) if price_12m is not None else None,
        price_24m_usd=round(price_24m, 4) if price_24m is not None else None,
        roi_12m_pct=round(roi_12m, 4) if roi_12m is not None else None,
        roi_24m_pct=round(roi_24m, 4) if roi_24m is not None else None,
        annualized_roi_pct=round(annualized, 4) if annualized is not None else None,
        max_drawdown_pct=round(_max_drawdown_pct(prices_only) or 0.0, 4),
        win_12m=int(roi_12m is not None and roi_12m >= 20.0),
        win_24m=int(roi_24m is not None and roi_24m >= 35.0) if roi_24m is not None else None,
        source_dataset=source_dataset,
        pattern_tags=json.dumps(tags, ensure_ascii=True),
    )


def _monthly_columns(df: pd.DataFrame) -> list[object]:
    cols: list[object] = []
    for col in df.columns:
        if isinstance(col, datetime):
            cols.append(col)
    cols.sort()
    return cols


def _read_workbook_from_zip(archive: zipfile.ZipFile, name: str, sheet: str, header_row: int) -> pd.DataFrame:
    raw = archive.read(name)
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=header_row)


def build_cases(zip_path: Path) -> list[CaseRow]:
    with zipfile.ZipFile(zip_path) as archive:
        small_df = _read_workbook_from_zip(archive, SMALL_SAMPLE_FILE, sheet="Sheet1", header_row=1)
        whole_df = _read_workbook_from_zip(archive, WHOLE_SAMPLE_FILE, sheet="Sheet1", header_row=1)
        dec2015_df = _read_workbook_from_zip(archive, DEC2015_FILE, sheet="DATA", header_row=0)

    msrp_map = {
        _normalize_text(row["id"]): _safe_float(row.get("Primary market price at release"))
        for _idx, row in dec2015_df.iterrows()
        if _normalize_text(row.get("id"))
    }

    cases: list[CaseRow] = []

    small_monthly_cols = _monthly_columns(small_df)
    for _idx, row in small_df.iterrows():
        case = _build_case_from_row(
            row,
            monthly_columns=small_monthly_cols,
            source_dataset="mendeley_small_2015_2019",
            set_id_col="id",
            set_name_col="Name",
            theme_col="Theme",
            release_year_col="Year of release",
            set_number_col="#",
            msrp_col="Retail price at release ($)",
        )
        if case is None:
            continue
        cases.append(case)

    whole_monthly_cols = _monthly_columns(whole_df)
    for _idx, row in whole_df.iterrows():
        case = _build_case_from_row(
            row,
            monthly_columns=whole_monthly_cols,
            source_dataset="mendeley_whole_2018_2019",
            set_id_col="id",
            set_name_col="name",
            theme_col="theme",
            release_year_col="year of release",
            set_number_col=None,
            msrp_col=None,
        )
        if case is None:
            continue
        if case.msrp_usd is None:
            case.msrp_usd = msrp_map.get(case.set_id)
        cases.append(case)

    # Deduplicate by set_id, prefer longer observation horizon.
    best_by_set: dict[str, CaseRow] = {}
    for case in cases:
        existing = best_by_set.get(case.set_id)
        if existing is None:
            best_by_set[case.set_id] = case
            continue
        prefer_new = False
        if case.observation_months > existing.observation_months:
            prefer_new = True
        elif case.observation_months == existing.observation_months:
            # Prefer row with msrp and 24m ROI.
            existing_quality = int(existing.msrp_usd is not None) + int(existing.roi_24m_pct is not None)
            new_quality = int(case.msrp_usd is not None) + int(case.roi_24m_pct is not None)
            if new_quality > existing_quality:
                prefer_new = True
        if prefer_new:
            best_by_set[case.set_id] = case

    final_rows = sorted(
        best_by_set.values(),
        key=lambda item: (0, int(item.set_id)) if item.set_id.isdigit() else (1, item.set_id),
    )
    return final_rows


def write_cases_csv(rows: Iterable[CaseRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "set_id",
        "set_number",
        "set_name",
        "theme",
        "release_year",
        "msrp_usd",
        "start_date",
        "end_date",
        "observation_months",
        "start_price_usd",
        "price_12m_usd",
        "price_24m_usd",
        "roi_12m_pct",
        "roi_24m_pct",
        "annualized_roi_pct",
        "max_drawdown_pct",
        "win_12m",
        "win_24m",
        "source_dataset",
        "pattern_tags",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "set_id": row.set_id,
                    "set_number": row.set_number,
                    "set_name": row.set_name,
                    "theme": row.theme,
                    "release_year": row.release_year if row.release_year is not None else "",
                    "msrp_usd": f"{row.msrp_usd:.4f}" if row.msrp_usd is not None else "",
                    "start_date": row.start_date,
                    "end_date": row.end_date,
                    "observation_months": row.observation_months,
                    "start_price_usd": f"{row.start_price_usd:.4f}",
                    "price_12m_usd": f"{row.price_12m_usd:.4f}" if row.price_12m_usd is not None else "",
                    "price_24m_usd": f"{row.price_24m_usd:.4f}" if row.price_24m_usd is not None else "",
                    "roi_12m_pct": f"{row.roi_12m_pct:.4f}" if row.roi_12m_pct is not None else "",
                    "roi_24m_pct": f"{row.roi_24m_pct:.4f}" if row.roi_24m_pct is not None else "",
                    "annualized_roi_pct": f"{row.annualized_roi_pct:.4f}" if row.annualized_roi_pct is not None else "",
                    "max_drawdown_pct": f"{row.max_drawdown_pct:.4f}" if row.max_drawdown_pct is not None else "",
                    "win_12m": row.win_12m if row.win_12m is not None else "",
                    "win_24m": row.win_24m if row.win_24m is not None else "",
                    "source_dataset": row.source_dataset,
                    "pattern_tags": row.pattern_tags,
                }
            )


def summarize(rows: list[CaseRow]) -> str:
    roi12 = [row.roi_12m_pct for row in rows if row.roi_12m_pct is not None]
    roi24 = [row.roi_24m_pct for row in rows if row.roi_24m_pct is not None]
    win12 = [row.win_12m for row in rows if row.win_12m is not None]
    return (
        f"rows={len(rows)} | roi12_n={len(roi12)} avg={statistics.fmean(roi12):.2f}% "
        f"| roi24_n={len(roi24)} avg={statistics.fmean(roi24):.2f}% "
        f"| win12_rate={statistics.fmean(win12) * 100:.1f}%"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build historical reference cases CSV from Mendeley LEGO dataset")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"Input zip not found: {args.zip}")

    rows = build_cases(args.zip)
    write_cases_csv(rows, args.out)
    print(summarize(rows))
    print(f"written={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
