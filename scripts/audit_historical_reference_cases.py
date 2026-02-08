from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path("data/historical_seed/historical_reference_cases.csv")


def _parse_pattern_tags(raw_value: Any) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [str(item).strip().lower() for item in loaded if str(item).strip()]
    except Exception:  # noqa: BLE001
        pass
    return [part.strip().lower() for part in raw.replace("|", ",").replace(";", ",").split(",") if part.strip()]


def _build_quality_report(
    rows: list[dict[str, Any]],
    *,
    max_median_age_years: int,
    min_theme_count: int,
    max_top_theme_share: float,
    max_general_tag_share: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "tier": "EMPTY",
            "degraded": True,
            "issues": ["seed_vuoto"],
            "rows_total": 0,
        }

    theme_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    years: list[int] = []
    rois: list[float] = []
    wins: list[int] = []

    for row in rows:
        theme = str(row.get("theme") or "").strip().lower()
        if theme:
            theme_counter[theme] += 1

        tags = _parse_pattern_tags(row.get("pattern_tags"))
        if tags:
            tag_counter.update(tags)

        end_date = str(row.get("end_date") or "").strip()
        if len(end_date) >= 4 and end_date[:4].isdigit():
            years.append(int(end_date[:4]))

        roi_raw = row.get("roi_12m_pct")
        if roi_raw not in (None, ""):
            try:
                rois.append(float(roi_raw))
            except (TypeError, ValueError):
                pass

        win_raw = row.get("win_12m")
        if win_raw not in (None, ""):
            try:
                value = int(win_raw)
                if value in (0, 1):
                    wins.append(value)
            except (TypeError, ValueError):
                pass

    rows_total = len(rows)
    top_theme, top_count = ("", 0)
    if theme_counter:
        top_theme, top_count = theme_counter.most_common(1)[0]
    top_theme_share = float(top_count) / float(max(1, rows_total))
    general_tag_share = float(tag_counter.get("general_collectible", 0)) / float(max(1, rows_total))

    median_end_year = int(round(statistics.median(years))) if years else None
    latest_end_year = int(max(years)) if years else None
    median_age_years = (
        int(max(0, date.today().year - int(median_end_year)))
        if median_end_year is not None
        else None
    )

    issues: list[str] = []
    if median_age_years is None:
        issues.append("assenza_end_date")
    elif median_age_years > max_median_age_years:
        issues.append(f"seed_datato_mediana_{median_age_years}y>{max_median_age_years}y")
    if len(theme_counter) < min_theme_count:
        issues.append(f"copertura_temi_bassa_{len(theme_counter)}<{min_theme_count}")
    if top_theme_share > max_top_theme_share:
        issues.append(f"concentrazione_tema_alta_{top_theme_share:.2f}>{max_top_theme_share:.2f}")
    if general_tag_share > max_general_tag_share:
        issues.append(f"pattern_generico_alto_{general_tag_share:.2f}>{max_general_tag_share:.2f}")

    degraded = bool(any(item.startswith("seed_datato_") for item in issues) or len(issues) >= 2)
    tier = "HEALTHY" if not issues else ("DEGRADED" if degraded else "WARNING")

    report = {
        "tier": tier,
        "degraded": degraded,
        "issues": issues,
        "rows_total": rows_total,
        "rows_with_roi": len(rois),
        "global_avg_roi_12m_pct": round(float(statistics.fmean(rois)), 4) if rois else None,
        "global_win_rate_pct": round(float(statistics.fmean(wins) * 100.0), 4) if wins else None,
        "theme_count": len(theme_counter),
        "top_theme": top_theme or None,
        "top_theme_share": round(top_theme_share, 4),
        "general_tag_share": round(general_tag_share, 4),
        "median_end_year": median_end_year,
        "latest_end_year": latest_end_year,
        "median_age_years": median_age_years,
        "guards": {
            "max_median_age_years": max_median_age_years,
            "min_theme_count": min_theme_count,
            "max_top_theme_share": max_top_theme_share,
            "max_general_tag_share": max_general_tag_share,
        },
        "top_themes": [
            {"theme": theme, "count": count}
            for theme, count in theme_counter.most_common(12)
        ],
        "top_tags": [
            {"tag": tag, "count": count}
            for tag, count in tag_counter.most_common(12)
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit quality of historical_reference_cases.csv")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--max-median-age-years", type=int, default=4)
    parser.add_argument("--min-theme-count", type=int, default=12)
    parser.add_argument("--max-top-theme-share", type=float, default=0.26)
    parser.add_argument("--max-general-tag-share", type=float, default=0.70)
    parser.add_argument("--fail-on-degraded", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print report as JSON only")
    args = parser.parse_args()

    csv_path = Path(args.path)
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return 2

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))

    report = _build_quality_report(
        rows,
        max_median_age_years=int(args.max_median_age_years),
        min_theme_count=int(args.min_theme_count),
        max_top_theme_share=float(args.max_top_theme_share),
        max_general_tag_share=float(args.max_general_tag_share),
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print("Historical Seed Quality Report")
        print(f"Tier: {report['tier']} | Degraded: {report['degraded']}")
        print(
            "Rows: "
            f"{report['rows_total']} | Themes: {report['theme_count']} | "
            f"Median end year: {report['median_end_year']} | Median age: {report['median_age_years']}y"
        )
        print(
            "Win/ROI: "
            f"Win-rate 12m {report['global_win_rate_pct']}% | "
            f"Avg ROI 12m {report['global_avg_roi_12m_pct']}%"
        )
        print(
            "Concentration: "
            f"Top theme {report['top_theme']} ({report['top_theme_share'] * 100:.1f}%) | "
            f"General tag {report['general_tag_share'] * 100:.1f}%"
        )
        if report["issues"]:
            print("Issues:")
            for item in report["issues"]:
                print(f"- {item}")
        else:
            print("Issues: none")
        print("")
        print("Top themes:")
        for row in report["top_themes"][:8]:
            print(f"- {row['theme']}: {row['count']}")

    if args.fail_on_degraded and bool(report.get("degraded")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
