"""Stage 3 artifact: the harness scored on the closing line itself and a base-rate prior.

No model exists yet. This exercises every metric on forecasters whose behaviour
we know in advance, so the numbers here are a check on the harness, not a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarise_data import md_table  # noqa: E402
from src import data, odds  # noqa: E402
from src import evaluate as ev  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
BET_BOOK = odds.BENCHMARK_BOOK


def base_rate_forecast(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Per-match prior = outcome frequencies over all *previous seasons*. NaN in the first."""
    P = np.full((len(df), 3), np.nan)
    seasons = sorted(df["season"].unique())
    for i, s in enumerate(seasons[1:], start=1):
        prev = df["season"].isin(seasons[:i]).to_numpy()
        P[(df["season"] == s).to_numpy()] = ev.base_rate(y[prev])
    return P


def main() -> None:
    df = data.load_matches()
    y = ev.outcome_index(df["result"])
    seasons = sorted(df["season"].unique())
    eval_mask = (df["season"] != seasons[0]).to_numpy()  # first season is prior-only

    forecasters: dict[str, np.ndarray] = {
        "close_shin": odds.benchmark_probabilities(df, "shin")[
            ["mkt_h", "mkt_d", "mkt_a"]
        ].to_numpy(),
        "close_multiplicative": odds.benchmark_probabilities(df, "multiplicative")[
            ["mkt_h", "mkt_d", "mkt_a"]
        ].to_numpy(),
        "close_additive": odds.benchmark_probabilities(df, "additive")[
            ["mkt_h", "mkt_d", "mkt_a"]
        ].to_numpy(),
        "open_shin": odds.book_probabilities(df, BET_BOOK, "open", "shin"),
        "base_rate": base_rate_forecast(df, y),
    }
    # Score on the common set of matches where every forecaster has a probability.
    ok = eval_mask & np.all([np.isfinite(P).all(axis=1) for P in forecasters.values()], axis=0)
    ye = y[ok]

    scores = pd.concat(
        [ev.score_probabilities(P[ok], ye, name) for name, P in forecasters.items()],
        ignore_index=True,
    )
    diffs = pd.concat(
        [
            ev.compare_probabilities(
                P[ok], forecasters["close_shin"][ok], ye, f"{name} - close_shin"
            )
            for name, P in forecasters.items()
            if name != "close_shin"
        ],
        ignore_index=True,
    )

    # Betting: each forecaster bets at the benchmark book's pre-closing price.
    open_odds = odds.odds_array(df, BET_BOOK, "open")[ok]
    close_odds = odds.odds_array(df, BET_BOOK, "close")[ok]
    close_p = forecasters["close_shin"][ok]
    ledgers = {
        name: ev.bet_frame(P[ok], ye, open_odds, close_odds, close_p)
        for name, P in forecasters.items()
    }
    bets = pd.concat([ev.score_bets(led, name) for name, led in ledgers.items()], ignore_index=True)

    # Calibration tables for the closing line (pooled and per outcome).
    rel = []
    for outcome, tag in [(None, "pooled"), (0, "home"), (1, "draw"), (2, "away")]:
        t = ev.reliability(close_p, ye, outcome=outcome)
        t.insert(0, "outcome", tag)
        t.insert(0, "forecaster", "close_shin")
        rel.append(t)
        t = ev.reliability(forecasters["base_rate"][ok], ye, outcome=outcome)
        t.insert(0, "outcome", tag)
        t.insert(0, "forecaster", "base_rate")
        rel.append(t)
    reliability = pd.concat(rel, ignore_index=True)

    # Log loss by season.
    season_rows = []
    for s in seasons[1:]:
        m = ok & (df["season"] == s).to_numpy()
        for name, P in forecasters.items():
            ll = ev.log_loss(P[m], y[m])
            ci = ev.bootstrap_ci(ll)
            season_rows.append({"season": s, "forecaster": name, **ci.as_dict()})
    by_season = pd.DataFrame(season_rows)

    # Line movement between the pre-closing snapshot and close, benchmark book.
    move = np.log(open_odds / close_odds)
    p_open = forecasters["open_shin"][ok]
    movement = pd.DataFrame(
        {
            "outcome": ["home", "draw", "away"],
            "n": [int(np.isfinite(move[:, k]).sum()) for k in range(3)],
            "mean_log_move": move.mean(axis=0),
            "mean_abs_log_move": np.abs(move).mean(axis=0),
            "p95_abs_log_move": np.percentile(np.abs(move), 95, axis=0),
            "mean_abs_prob_move": np.abs(p_open - close_p).mean(axis=0),
            "share_unchanged": (move == 0).mean(axis=0),
        }
    )

    RESULTS.mkdir(exist_ok=True)
    scores.to_csv(RESULTS / "market_scores.csv", index=False, float_format="%.6f")
    diffs.to_csv(RESULTS / "market_score_diffs.csv", index=False, float_format="%.6f")
    bets.to_csv(RESULTS / "market_bets.csv", index=False, float_format="%.6f")
    reliability.to_csv(RESULTS / "market_reliability.csv", index=False, float_format="%.6f")
    by_season.to_csv(RESULTS / "market_logloss_by_season.csv", index=False, float_format="%.6f")
    movement.to_csv(RESULTS / "market_line_movement.csv", index=False, float_format="%.6f")

    wide = scores.pivot(index="forecaster", columns="metric", values="value")[
        ["log_loss", "brier", "ece", "ece_h", "ece_d", "ece_a"]
    ].reset_index()
    lines = [
        "# Harness check: closing line vs itself and vs base rate",
        "",
        f"Matches scored: {int(ok.sum())} (seasons {seasons[1]} to {seasons[-1]}; "
        f"{seasons[0]} is used only to seed the base-rate prior).",
        f"Bets are struck at {BET_BOOK} pre-closing odds and settled at that price.",
        "",
        "## Proper scores and calibration",
        "",
        md_table(wide, ".4f"),
        "",
        "## Paired differences vs close_shin (negative = better than the closing line)",
        "",
        md_table(diffs, ".4f"),
        "",
        "## Betting metrics",
        "",
        md_table(bets, ".4f"),
        "",
        "## Closing line reliability (pooled over outcomes)",
        "",
        md_table(
            reliability[
                (reliability.forecaster == "close_shin") & (reliability.outcome == "pooled")
            ].drop(columns=["forecaster", "outcome"]),
            ".4f",
        ),
        "",
        "## Pre-closing to closing line movement (benchmark book)",
        "",
        md_table(movement, ".4f"),
        "",
    ]
    (RESULTS / "market_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
