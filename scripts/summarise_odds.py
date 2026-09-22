"""Stage 2 artifact: overround distribution, devig disagreement, benchmark composition."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, odds  # noqa: E402
from src.report import RESULTS, md_table, write_md  # noqa: E402

BOOKS = ["ps", "b365", "bw", "iw", "wh", "vc"]  # books with at least 5 seasons of prices
PAIRS = [("multiplicative", "shin"), ("multiplicative", "additive"), ("additive", "shin")]


def overround_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (book, stage, season) percentiles, and the same pooled across seasons."""
    long = pd.concat(
        pd.DataFrame(
            {"book": b, "stage": s, "season": df["season"], "overround": odds.overround(odds.odds_array(df, b, s))}
        )
        for b in BOOKS
        for s in ("open", "close")
    ).dropna()
    q = lambda k: lambda v: np.percentile(v, k)  # noqa: E731
    by_season = (
        long.groupby(["book", "stage", "season"], sort=False)["overround"]
        .agg(n="size", mean="mean", median="median", p5=q(5), p95=q(95), min="min", max="max")
        .reset_index()
    )
    pooled = (
        by_season.groupby(["book", "stage"])
        .apply(
            lambda g: pd.Series(
                {
                    "seasons": len(g),
                    "n": g["n"].sum(),
                    "mean": np.average(g["mean"], weights=g["n"]),
                    "min_season_mean": g["mean"].min(),
                    "max_season_mean": g["mean"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .astype({"seasons": int, "n": int})
    )
    return by_season, pooled


def disagreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise absolute differences between devig methods on closing odds, in probability."""
    rows = []
    for book in BOOKS:
        o = odds.odds_array(df, book, "close")
        o = o[np.isfinite(o[:, 0])]
        if not len(o):
            continue
        p = {m: odds.devig(o, m) for m in odds.DEVIG_METHODS}
        i = np.arange(len(o))
        for a, b in PAIRS:
            d = np.abs(p[a] - p[b])
            rows.append({
                "book": book, "pair": f"{a} vs {b}", "n": len(o),
                "mean_abs_diff_h": d[:, 0].mean(), "mean_abs_diff_d": d[:, 1].mean(), "mean_abs_diff_a": d[:, 2].mean(),
                "max_abs_diff": d.max(),
                "mean_abs_diff_favourite": d[i, p[a].argmax(axis=1)].mean(),
                "mean_abs_diff_longshot": d[i, p[a].argmin(axis=1)].mean(),
            })  # fmt: skip
    return pd.DataFrame(rows)


def disagreement_by_favourite(df: pd.DataFrame) -> pd.DataFrame:
    """Shin minus multiplicative on the favourite and the longshot, binned by favourite prob."""
    o = odds.odds_array(df, odds.BENCHMARK_BOOK, "close")
    o = o[np.isfinite(o[:, 0])]
    m, s = odds.devig(o, "multiplicative"), odds.devig(o, "shin")
    i, fav, dog = np.arange(len(o)), m.argmax(axis=1), m.argmin(axis=1)
    frame = pd.DataFrame({
        "fav_bin": pd.cut(m[i, fav], np.arange(0.3, 1.01, 0.1), right=False).astype(str),
        "shin_minus_mult_favourite": s[i, fav] - m[i, fav],
        "shin_minus_mult_longshot": s[i, dog] - m[i, dog],
    })  # fmt: skip
    return (
        frame.groupby("fav_bin")
        .agg(["size", "mean"])
        .pipe(
            lambda g: pd.DataFrame(
                {
                    "n": g[("shin_minus_mult_favourite", "size")],
                    "shin_minus_mult_favourite": g[("shin_minus_mult_favourite", "mean")],
                    "shin_minus_mult_longshot": g[("shin_minus_mult_longshot", "mean")],
                }
            )  # fmt: skip
        )
        .reset_index()
    )


def main() -> None:
    df = data.load_matches()
    RESULTS.mkdir(exist_ok=True)

    by_season, pooled = overround_tables(df)
    dis = disagreement_table(df)
    byfav = disagreement_by_favourite(df)
    bm = (
        odds.benchmark_probabilities(df)
        .join(df["season"])
        .groupby(["season", "mkt_source"])
        .agg(n=("mkt_h", "size"), mean_n_books=("mkt_n_books", "mean"))
        .reset_index()
    )

    by_season.to_csv(RESULTS / "odds_overround.csv", index=False, float_format="%.6f")
    dis.to_csv(RESULTS / "odds_devig_disagreement.csv", index=False, float_format="%.6f")
    byfav.to_csv(RESULTS / "odds_devig_by_favourite.csv", index=False, float_format="%.6f")
    bm.to_csv(RESULTS / "odds_benchmark_composition.csv", index=False, float_format="%.4f")
    write_md("odds_summary.md", [
        "# Odds summary",
        "",
        f"Benchmark: `{odds.BENCHMARK_BOOK}` closing odds, devigged with Shin's method; "
        "consensus of the other individual books where the benchmark book is absent.",
        "",
        "## Overround, pooled across seasons (fraction, e.g. 0.025 = 2.5%)", "", md_table(pooled, ".4f"), "",
        "Per-season detail with percentiles: `odds_overround.csv`.", "",
        "## Devig method disagreement on closing odds (absolute probability difference)", "",
        md_table(dis, ".4f"), "",
        "## Shin minus multiplicative, Pinnacle closing, by favourite strength", "", md_table(byfav, ".4f"), "",
        "## Benchmark composition", "", md_table(bm, ".2f"),
    ])  # fmt: skip


if __name__ == "__main__":
    main()
