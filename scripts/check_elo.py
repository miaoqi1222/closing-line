"""Stage 4 artifact: fit Elo on the first three seasons, score the rest against the market."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarise_data import md_table  # noqa: E402
from src import data, model, odds  # noqa: E402
from src import evaluate as ev  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_TRAIN_SEASONS = 3


def main() -> None:
    df = data.load_matches()
    seasons = sorted(df["season"].unique())
    is_train = df["season"].isin(seasons[:N_TRAIN_SEASONS])
    train, test = df[is_train], df[~is_train]

    elo = model.fit(train)
    y = ev.outcome_index(test["result"])
    forecasts = {
        "elo": elo.predict(test),
        "close_shin": odds.benchmark_probabilities(test)[["mkt_h", "mkt_d", "mkt_a"]].to_numpy(),
        "base_rate": np.tile(ev.base_rate(ev.outcome_index(train["result"])), (len(test), 1)),
    }

    rows = []
    for name, p in forecasts.items():
        for metric, fn in [("log_loss", ev.log_loss), ("brier", ev.brier)]:
            rows.append(
                {"forecaster": name, "metric": metric, **ev.bootstrap_ci(fn(p, y)).as_dict()}
            )
        rows.append({"forecaster": name, "metric": "ece", "value": ev.ece(p, y), "n": len(y)})
    diff = ev.paired_diff_ci(
        ev.log_loss(forecasts["elo"], y), ev.log_loss(forecasts["close_shin"], y)
    )
    rows.append({"forecaster": "elo - close_shin", "metric": "log_loss_diff", **diff.as_dict()})
    scores = pd.DataFrame(rows)

    params = pd.DataFrame([{"k": elo.k, "hfa": elo.hfa, "draw": elo.draw}])
    top = pd.Series(elo.ratings).sort_values(ascending=False).round(1)
    ratings = top.rename_axis("team").reset_index(name="rating")

    scores.to_csv(RESULTS / "elo_check_scores.csv", index=False, float_format="%.6f")
    params.to_csv(RESULTS / "elo_check_params.csv", index=False, float_format="%.6f")
    ratings.to_csv(RESULTS / "elo_check_ratings.csv", index=False, float_format="%.1f")
    (RESULTS / "elo_check.md").write_text(
        "\n".join(
            [
                "# Elo check (single split; the walk-forward backtest is stage 5)",
                "",
                f"Train: {seasons[0]} to {seasons[N_TRAIN_SEASONS - 1]} ({len(train)} matches). "
                f"Test: {seasons[N_TRAIN_SEASONS]} to {seasons[-1]} ({len(test)} matches).",
                "",
                "## Fitted on train only",
                "",
                md_table(params, ".3f"),
                "",
                "## Test scores",
                "",
                md_table(scores, ".4f"),
                "",
                "## Final ratings, top and bottom five",
                "",
                md_table(pd.concat([ratings.head(5), ratings.tail(5)]), ".1f"),
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
