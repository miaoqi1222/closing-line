"""Stage 1 artifact: what did the loader load? Writes to results/, prints nothing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal GitHub-markdown table (avoids the tabulate dependency)."""
    df = df.reset_index() if df.index.name is not None else df

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def main() -> None:
    df = data.load_matches()
    RESULTS.mkdir(exist_ok=True)

    per_season = (
        df.groupby("season")
        .agg(
            matches=("result", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            teams=("home", "nunique"),
            home_win=("result", lambda s: (s == "H").mean()),
            draw=("result", lambda s: (s == "D").mean()),
            away_win=("result", lambda s: (s == "A").mean()),
        )
        .reset_index()
    )
    per_season["first_date"] = per_season["first_date"].dt.date
    per_season["last_date"] = per_season["last_date"].dt.date
    per_season.to_csv(RESULTS / "data_seasons.csv", index=False, float_format="%.4f")

    cov = data.coverage(df)
    cov.to_csv(RESULTS / "data_coverage.csv", index=False)

    wide = cov.pivot(index="book", columns="season", values="close_complete")
    wide_open = cov.pivot(index="book", columns="season", values="open_complete")

    lines = [
        "# Data summary",
        "",
        f"Source: football-data.co.uk, league {data.DEFAULT_LEAGUE}, "
        f"seasons {df.season.min()} to {df.season.max()}.",
        f"Rows: {len(df)}; date range {df.date.min().date()} to {df.date.max().date()}.",
        f"Odds columns: {len(data.odds_columns(df))} ({len(data.books(df))} books).",
        "",
        "## Matches per season",
        "",
        md_table(per_season),
        "",
        "## Closing-odds coverage (matches with a complete H/D/A triple, out of 380)",
        "",
        md_table(wide),
        "",
        "## Pre-closing ('open') odds coverage",
        "",
        md_table(wide_open),
        "",
    ]
    (RESULTS / "data_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
