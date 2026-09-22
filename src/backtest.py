"""Walk-forward backtest: expanding training window, one season out-of-sample at a time.

For test season i the model is fitted on seasons 0..i-1 only, then predicts
season i in date order (updating its ratings on each result as it goes, which
uses only information available before kickoff). Nothing is refitted on
future data. The output is a per-match frame joining model probabilities,
the closing-line benchmark, the bet taken at the pre-closing price, stakes,
P&L and CLV.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src import evaluate as ev
from src import odds

MIN_TRAIN_SEASONS = 3
MODEL = ["model_h", "model_d", "model_a"]
MKT = ["mkt_h", "mkt_d", "mkt_a"]
PRIOR = ["prior_h", "prior_d", "prior_a"]


def walk_forward(df: pd.DataFrame, fit: Callable, min_train_seasons: int = MIN_TRAIN_SEASONS):
    """``(P, prior)``: model and base-rate probabilities per match; NaN in the training-only seasons."""
    seasons = sorted(df["season"].unique())
    P, prior = np.full((len(df), 3), np.nan), np.full((len(df), 3), np.nan)
    for i in range(min_train_seasons, len(seasons)):
        train = df[df["season"].isin(seasons[:i])]
        test = (df["season"] == seasons[i]).to_numpy()
        P[test] = fit(train).predict(df[test])
        prior[test] = ev.base_rate(ev.outcome_index(train["result"]))
    return P, prior


def run(df: pd.DataFrame, fit: Callable, book: str = odds.BENCHMARK_BOOK) -> pd.DataFrame:
    """Per-match backtest frame for the out-of-sample seasons.

    Bets are the model's, struck at ``book``'s pre-closing price and settled there;
    matches without that price are scored but not bet.
    """
    P, prior = walk_forward(df, fit)
    keep = np.isfinite(P[:, 0])
    df, P, prior = df[keep].reset_index(drop=True), P[keep], prior[keep]
    y = ev.outcome_index(df["result"])
    mkt = odds.benchmark_probabilities(df)
    open_odds, close_odds = odds.odds_array(df, book, "open"), odds.odds_array(df, book, "close")
    ledger = ev.bet_frame(P, y, open_odds, close_odds, mkt[MKT].to_numpy())
    out = pd.concat(
        [
            df[["season", "date", "home", "away", "result"]],
            pd.DataFrame(P, columns=MODEL),
            mkt,
            pd.DataFrame(prior, columns=PRIOR),
            pd.DataFrame(open_odds, columns=[f"{book}_open_{o}" for o in "hda"]),
            pd.DataFrame(close_odds, columns=[f"{book}_close_{o}" for o in "hda"]),
            ledger.drop(columns=["model_p", "close_p", "open_odds"]),
        ],
        axis=1,
    )
    out["bet"] = out["bet"].map({-1: "", 0: "H", 1: "D", 2: "A"})
    return out
