"""Stage 5 artifact: walk-forward backtest of the Elo baseline against the closing line."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import backtest, data, model  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.report import RESULTS, md_table, write_md  # noqa: E402


def main() -> None:
    df = data.load_matches()
    bt = backtest.run(df, model.fit)
    y = ev.outcome_index(bt["result"])
    F = {
        k: bt[cols].to_numpy()
        for k, cols in [("elo", backtest.MODEL), ("close", backtest.MKT), ("base_rate", backtest.PRIOR)]
    }

    scores = pd.concat(ev.score_probabilities(P, y, k) for k, P in F.items())
    diffs = pd.concat(ev.compare_probabilities(P, F["close"], y, f"{k} - close") for k, P in F.items() if k != "close")
    bets = ev.score_bets(bt.assign(bet=bt["bet"].map({"H": 0, "D": 1, "A": 2}).fillna(-1)), "elo")
    reliability = pd.concat(
        ev.reliability(P, y).assign(forecaster=k).pipe(lambda t: t[["forecaster", *t.columns[:5]]])
        for k, P in F.items()
    )
    seasons = bt["season"].to_numpy()
    by_season = pd.DataFrame(
        {"season": s, "forecaster": k, **ev.bootstrap_ci(ev.log_loss(P[seasons == s], y[seasons == s]))}
        for s in sorted(set(seasons))
        for k, P in F.items()
    )

    RESULTS.mkdir(exist_ok=True)
    bt.to_csv(RESULTS / "backtest_matches.csv", index=False, float_format="%.6f")
    for name, table in [
        ("scores", scores),
        ("score_diffs", diffs),
        ("bets", bets),
        ("reliability", reliability),
        ("logloss_by_season", by_season),
    ]:
        table.to_csv(RESULTS / f"backtest_{name}.csv", index=False, float_format="%.6f")
    wide = scores.pivot(index="forecaster", columns="metric", values="value").reset_index()
    write_md("backtest_summary.md", [
        "# Walk-forward backtest: Elo vs closing line",
        "",
        f"Out-of-sample matches: {len(bt)} (seasons {seasons.min()} to {seasons.max()}); "
        f"training window expands one season at a time from {backtest.MIN_TRAIN_SEASONS} seasons.",
        f"Bets: {int((bt['bet'] != '').sum())}, struck at Pinnacle pre-closing odds and settled there.",
        "",
        "## Proper scores and calibration", "",
        md_table(wide[["forecaster", "log_loss", "brier", "ece", "ece_h", "ece_d", "ece_a"]], ".4f"), "",
        "## Paired differences vs the closing line (negative = better than the close)", "", md_table(diffs, ".4f"), "",
        "## Betting", "", md_table(bets, ".4f"), "",
        "## Log loss by season", "",
        md_table(by_season.pivot(index="season", columns="forecaster", values="value").reset_index(), ".4f"),
    ])  # fmt: skip


if __name__ == "__main__":
    main()
