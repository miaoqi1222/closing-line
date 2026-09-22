"""Stage 3 artifact: the harness scored on the closing line itself and a base-rate prior.

No model exists yet. These forecasters have known behaviour, so the numbers are a
check on the harness, not a result.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, odds  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.report import RESULTS, md_table, write_md  # noqa: E402

BOOK = odds.BENCHMARK_BOOK
MKT = ["mkt_h", "mkt_d", "mkt_a"]


def base_rate_forecast(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Prior for each season = outcome frequencies over all previous seasons; NaN in the first."""
    P = np.full((len(df), 3), np.nan)
    for i, s in enumerate(seasons := sorted(df["season"].unique())):
        if i:
            P[(df["season"] == s).to_numpy()] = ev.base_rate(y[df["season"].isin(seasons[:i])])
    return P


def main() -> None:
    df = data.load_matches()
    y = ev.outcome_index(df["result"])
    seasons = sorted(df["season"].unique())
    forecasters = {
        "close_shin": odds.benchmark_probabilities(df, "shin")[MKT].to_numpy(),
        "close_multiplicative": odds.benchmark_probabilities(df, "multiplicative")[MKT].to_numpy(),
        "close_additive": odds.benchmark_probabilities(df, "additive")[MKT].to_numpy(),
        "open_shin": odds.book_probabilities(df, BOOK, "open"),
        "base_rate": base_rate_forecast(df, y),
    }
    # Common set: every forecaster has a probability (drops season 1 and unpriced matches).
    ok = np.all([np.isfinite(P[:, 0]) for P in forecasters.values()], axis=0)
    F = {k: P[ok] for k, P in forecasters.items()}
    ye, close_p = y[ok], F["close_shin"]
    open_odds, close_odds = odds.odds_array(df, BOOK, "open")[ok], odds.odds_array(df, BOOK, "close")[ok]

    scores = pd.concat(ev.score_probabilities(P, ye, k) for k, P in F.items())
    diffs = pd.concat(
        ev.compare_probabilities(P, close_p, ye, f"{k} - close_shin") for k, P in F.items() if k != "close_shin"
    )
    bets = pd.concat(ev.score_bets(ev.bet_frame(P, ye, open_odds, close_odds, close_p), k) for k, P in F.items())
    reliability = pd.concat(
        ev.reliability(F[k], ye, outcome=o).assign(forecaster=k, outcome=tag)
        for k in ("close_shin", "base_rate")
        for o, tag in [(None, "pooled"), (0, "home"), (1, "draw"), (2, "away")]
    ).pipe(lambda t: t[["forecaster", "outcome", *t.columns[:5]]])
    season_of = df["season"].to_numpy()[ok]
    by_season = pd.DataFrame(
        {"season": s, "forecaster": k, **ev.bootstrap_ci(ev.log_loss(P[season_of == s], ye[season_of == s]))}
        for s in seasons[1:]
        for k, P in F.items()
    )
    move = np.log(open_odds / close_odds)
    movement = pd.DataFrame({
        "outcome": ["home", "draw", "away"],
        "n": np.isfinite(move).sum(axis=0),
        "mean_log_move": move.mean(axis=0),
        "mean_abs_log_move": np.abs(move).mean(axis=0),
        "p95_abs_log_move": np.percentile(np.abs(move), 95, axis=0),
        "mean_abs_prob_move": np.abs(F["open_shin"] - close_p).mean(axis=0),
        "share_unchanged": (move == 0).mean(axis=0),
    })  # fmt: skip

    RESULTS.mkdir(exist_ok=True)
    for name, table in [
        ("scores", scores),
        ("score_diffs", diffs),
        ("bets", bets),
        ("reliability", reliability),
        ("logloss_by_season", by_season),
        ("line_movement", movement),
    ]:
        table.to_csv(RESULTS / f"market_{name}.csv", index=False, float_format="%.6f")
    wide = scores.pivot(index="forecaster", columns="metric", values="value").reset_index()
    close_rel = reliability.query("forecaster == 'close_shin' and outcome == 'pooled'").iloc[:, :5]
    write_md("market_summary.md", [
        "# Harness check: closing line vs itself and vs base rate",
        "",
        f"Matches scored: {int(ok.sum())} (seasons {seasons[1]} to {seasons[-1]}; "
        f"{seasons[0]} is used only to seed the base-rate prior).",
        f"Bets are struck at {BOOK} pre-closing odds and settled at that price.",
        "",
        "## Proper scores and calibration", "",
        md_table(wide[["forecaster", "log_loss", "brier", "ece", "ece_h", "ece_d", "ece_a"]], ".4f"), "",
        "## Paired differences vs close_shin (negative = better than the closing line)", "", md_table(diffs, ".4f"), "",
        "## Betting metrics", "", md_table(bets, ".4f"), "",
        "## Closing line reliability (pooled over outcomes)", "", md_table(close_rel, ".4f"), "",
        "## Pre-closing to closing line movement (benchmark book)", "", md_table(movement, ".4f"),
    ])  # fmt: skip


if __name__ == "__main__":
    main()
