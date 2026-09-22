"""Stage 4 artifact: fit Elo on the first three seasons, score the rest against the market."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, model, odds  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.report import RESULTS, md_table, write_md  # noqa: E402

N_TRAIN_SEASONS = 3


def main() -> None:
    df = data.load_matches()
    seasons = sorted(df["season"].unique())
    is_train = df["season"].isin(seasons[:N_TRAIN_SEASONS])
    train, test = df[is_train], df[~is_train]

    elo = model.fit(train)
    y = ev.outcome_index(test["result"])
    F = {
        "elo": elo.predict(test),
        "close_shin": odds.benchmark_probabilities(test)[["mkt_h", "mkt_d", "mkt_a"]].to_numpy(),
        "base_rate": np.tile(ev.base_rate(ev.outcome_index(train["result"])), (len(test), 1)),
    }
    scores = pd.concat(ev.score_probabilities(P, y, k) for k, P in F.items())
    diff = ev.compare_probabilities(F["elo"], F["close_shin"], y, "elo - close_shin")
    params = pd.DataFrame([{"k": elo.k, "hfa": elo.hfa, "draw": elo.draw}])
    ratings = (
        pd.Series(elo.ratings, name="rating").sort_values(ascending=False).round(1).rename_axis("team").reset_index()
    )

    RESULTS.mkdir(exist_ok=True)
    pd.concat([scores, diff]).to_csv(RESULTS / "elo_check_scores.csv", index=False, float_format="%.6f")
    params.to_csv(RESULTS / "elo_check_params.csv", index=False, float_format="%.6f")
    ratings.to_csv(RESULTS / "elo_check_ratings.csv", index=False, float_format="%.1f")
    write_md("elo_check.md", [
        "# Elo check (single split; the walk-forward backtest is stage 5)",
        "",
        f"Train: {seasons[0]} to {seasons[N_TRAIN_SEASONS - 1]} ({len(train)} matches). "
        f"Test: {seasons[N_TRAIN_SEASONS]} to {seasons[-1]} ({len(test)} matches).",
        "",
        "## Fitted on train only", "", md_table(params, ".3f"), "",
        "## Test scores", "", md_table(pd.concat([scores, diff]), ".4f"), "",
        "## Final ratings, top and bottom five", "", md_table(pd.concat([ratings.head(5), ratings.tail(5)]), ".1f"),
    ])  # fmt: skip


if __name__ == "__main__":
    main()
