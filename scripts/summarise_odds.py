"""Stage 2 artifact: overround distribution, devig disagreement, benchmark composition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarise_data import md_table  # noqa: E402
from src import data, odds  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
BOOKS = ["ps", "b365", "bw", "iw", "wh", "vc"]  # books with at least 5 seasons of prices
METHODS = list(odds.DEVIG_METHODS)


def overround_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for book in BOOKS:
        for stage in ("open", "close"):
            ov = odds.overround(odds.odds_array(df, book, stage))
            for season, idx in df.groupby("season").indices.items():
                v = ov[idx]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                rows.append(
                    {
                        "book": book,
                        "stage": stage,
                        "season": season,
                        "n": len(v),
                        "mean": v.mean(),
                        "median": np.median(v),
                        "p5": np.percentile(v, 5),
                        "p95": np.percentile(v, 95),
                        "min": v.min(),
                        "max": v.max(),
                    }
                )
    return pd.DataFrame(rows)


def disagreement_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise absolute differences between devig methods, in probability points."""
    rows = []
    pairs = [("multiplicative", "shin"), ("multiplicative", "additive"), ("additive", "shin")]
    for book in BOOKS:
        o = odds.odds_array(df, book, "close")
        ok = np.isfinite(o).all(axis=1)
        if ok.sum() == 0:
            continue
        p = {m: odds.devig(o[ok], m) for m in METHODS}
        for a, b in pairs:
            d = np.abs(p[a] - p[b])
            rows.append(
                {
                    "book": book,
                    "pair": f"{a} vs {b}",
                    "n": int(ok.sum()),
                    "mean_abs_diff_h": d[:, 0].mean(),
                    "mean_abs_diff_d": d[:, 1].mean(),
                    "mean_abs_diff_a": d[:, 2].mean(),
                    "max_abs_diff": d.max(),
                    "mean_abs_diff_favourite": d[np.arange(len(d)), p[a].argmax(axis=1)].mean(),
                    "mean_abs_diff_longshot": d[np.arange(len(d)), p[a].argmin(axis=1)].mean(),
                }
            )
    return pd.DataFrame(rows)


def disagreement_by_favourite(df: pd.DataFrame, book: str = odds.BENCHMARK_BOOK) -> pd.DataFrame:
    """Shin minus multiplicative on the favourite and the longshot, binned by favourite prob."""
    o = odds.odds_array(df, book, "close")
    ok = np.isfinite(o).all(axis=1)
    m, s = odds.devig(o[ok], "multiplicative"), odds.devig(o[ok], "shin")
    fav = m.argmax(axis=1)
    dog = m.argmin(axis=1)
    i = np.arange(len(m))
    frame = pd.DataFrame(
        {
            "fav_prob": m[i, fav],
            "shin_minus_mult_favourite": s[i, fav] - m[i, fav],
            "shin_minus_mult_longshot": s[i, dog] - m[i, dog],
        }
    )
    bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    frame["fav_bin"] = pd.cut(frame["fav_prob"], bins, right=False)
    out = (
        frame.groupby("fav_bin", observed=True)
        .agg(
            n=("fav_prob", "size"),
            shin_minus_mult_favourite=("shin_minus_mult_favourite", "mean"),
            shin_minus_mult_longshot=("shin_minus_mult_longshot", "mean"),
        )
        .reset_index()
    )
    out["fav_bin"] = out["fav_bin"].astype(str)
    return out


def benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    bm = odds.benchmark_probabilities(df)
    out = (
        pd.concat([df[["season"]], bm], axis=1)
        .groupby(["season", "mkt_source"])
        .agg(n=("mkt_h", "size"), mean_n_books=("mkt_n_books", "mean"))
        .reset_index()
    )
    return out


def main() -> None:
    df = data.load_matches()
    RESULTS.mkdir(exist_ok=True)

    ov = overround_table(df)
    ov.to_csv(RESULTS / "odds_overround.csv", index=False, float_format="%.6f")
    dis = disagreement_table(df)
    dis.to_csv(RESULTS / "odds_devig_disagreement.csv", index=False, float_format="%.6f")
    byfav = disagreement_by_favourite(df)
    byfav.to_csv(RESULTS / "odds_devig_by_favourite.csv", index=False, float_format="%.6f")
    bm = benchmark_table(df)
    bm.to_csv(RESULTS / "odds_benchmark_composition.csv", index=False, float_format="%.4f")

    ps_close = odds.odds_array(df, "ps", "close")
    ps_ok = np.isfinite(ps_close).all(axis=1)
    n_clipped = int(odds.additive_clipped(ps_close)[ps_ok].sum())
    n_ps = int(ps_ok.sum())

    ov_pooled = (
        ov.groupby(["book", "stage"])
        .apply(
            lambda g: pd.Series(
                {
                    "seasons": g["season"].nunique(),
                    "n": g["n"].sum(),
                    "mean": np.average(g["mean"], weights=g["n"]),
                    "min_season_mean": g["mean"].min(),
                    "max_season_mean": g["mean"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    ov_pooled["seasons"] = ov_pooled["seasons"].astype(int)
    ov_pooled["n"] = ov_pooled["n"].astype(int)

    lines = [
        "# Odds summary",
        "",
        f"Benchmark: `{odds.BENCHMARK_BOOK}` closing odds, devigged with Shin's method; "
        "consensus of the other individual books where the benchmark book is absent.",
        "",
        "## Overround, pooled across seasons (fraction, e.g. 0.025 = 2.5%)",
        "",
        md_table(ov_pooled, ".4f"),
        "",
        "Per-season detail with percentiles: `odds_overround.csv`.",
        "",
        "## Devig method disagreement on closing odds (absolute probability difference)",
        "",
        md_table(dis, ".4f"),
        "",
        f"Additive method needed flooring on {n_clipped} of {n_ps} Pinnacle closing rows.",
        "",
        "## Shin minus multiplicative, Pinnacle closing, by favourite strength",
        "",
        md_table(byfav, ".4f"),
        "",
        "## Benchmark composition",
        "",
        md_table(bm, ".2f"),
        "",
    ]
    (RESULTS / "odds_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
