"""Stage 1 artifact: what did the loader load? Writes to results/, prints nothing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data  # noqa: E402
from src.report import RESULTS, md_table, write_md  # noqa: E402


def main() -> None:
    df = data.load_matches()
    RESULTS.mkdir(exist_ok=True)

    share = lambda r: lambda s: (s == r).mean()  # noqa: E731
    seasons = (
        df.groupby("season")
        .agg(
            matches=("result", "size"),
            first_date=("date", lambda d: d.min().date()),
            last_date=("date", lambda d: d.max().date()),
            teams=("home", "nunique"),
            home_win=("result", share("H")),
            draw=("result", share("D")),
            away_win=("result", share("A")),
        )
        .reset_index()
    )
    seasons.to_csv(RESULTS / "data_seasons.csv", index=False, float_format="%.4f")

    cov = data.coverage(df)
    cov.to_csv(RESULTS / "data_coverage.csv", index=False)
    close = cov.pivot(index="book", columns="season", values="close_complete")
    open_ = cov.pivot(index="book", columns="season", values="open_complete")

    write_md("data_summary.md", [
        "# Data summary",
        "",
        f"Source: football-data.co.uk, league {data.DEFAULT_LEAGUE}, "
        f"seasons {df.season.min()} to {df.season.max()}.",
        f"Rows: {len(df)}; date range {df.date.min().date()} to {df.date.max().date()}.",
        f"Odds columns: {len(df.columns) - len(data.META_COLUMNS)} ({len(data.books(df))} books).",
        "",
        "## Matches per season", "", md_table(seasons), "",
        "## Closing-odds coverage (matches with a complete H/D/A triple, out of 380)", "",
        md_table(close), "",
        "## Pre-closing ('open') odds coverage", "", md_table(open_),
    ])  # fmt: skip


if __name__ == "__main__":
    main()
