"""Four plain figures from the backtest artifacts. Default matplotlib style, two colours plus grey."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.report import FIGS, RESULTS  # noqa: E402

MODEL = dict(color="black", marker="o", linestyle="-", label="Elo (model)")
MARKET = dict(color="tab:blue", marker="s", linestyle="--", label="Closing line (Pinnacle, Shin devig)")
GREY = dict(color="grey", linewidth=1)


def reliability() -> None:
    rel = pd.read_csv(RESULTS / "backtest_reliability.csv")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], **GREY)
    for name, style in [("elo", MODEL), ("close", MARKET)]:
        t = rel[(rel["forecaster"] == name) & (rel["n"] > 0)]
        ax.plot(t["mean_pred"], t["mean_obs"], **style)
    ax.set(xlabel="Forecast probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1),
           title="Reliability, pooled over home/draw/away (10 bins)")  # fmt: skip
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "reliability.png", dpi=120)


def clv_hist() -> None:
    bt = pd.read_csv(RESULTS / "backtest_matches.csv")
    clv = bt.loc[bt["bet"].notna(), "clv"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clv, bins=np.arange(-0.4, 0.42, 0.02), color="black")
    ax.axvline(0, **GREY)
    ax.axvline(clv.mean(), color="tab:blue", linestyle="--", label=f"mean = {clv.mean():+.3f} (n = {len(clv)})")
    ax.set(xlabel="CLV of model bets: open / close - 1", ylabel="Bets", title="Closing line value distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "clv_hist.png", dpi=120)


def cumulative_pnl() -> None:
    bt = pd.read_csv(RESULTS / "backtest_matches.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, **GREY)
    ax.plot(bt["date"], bt["pnl_flat"].cumsum() / 0.01, color="black", label="Flat stakes (units of one stake)")
    ax.plot(
        bt["date"],
        bt["pnl_kelly"].cumsum() / 0.01,
        color="tab:blue",
        linestyle="--",
        label="Quarter Kelly, 2% cap (units of 1% bankroll)",
    )
    ax.set(xlabel="Date", ylabel="Cumulative P&L", title="Walk-forward P&L at Pinnacle pre-closing prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "cumulative_pnl.png", dpi=120)


def logloss_by_season() -> None:
    t = pd.read_csv(RESULTS / "backtest_logloss_by_season.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(t["season"].nunique())
    for name, style in [
        ("elo", MODEL),
        ("close", MARKET),
        ("base_rate", dict(color="grey", marker="^", linestyle=":", label="Base rate")),
    ]:
        s = t[t["forecaster"] == name]
        ax.errorbar(x, s["value"], yerr=[s["value"] - s["ci_lo"], s["ci_hi"] - s["value"]], capsize=3, **style)
    ax.set(xticks=x, xticklabels=sorted(t["season"].unique()), ylabel="Log loss (lower is better)",
           title="Log loss by season, 95% bootstrap intervals")  # fmt: skip
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "logloss_by_season.png", dpi=120)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    reliability()
    clv_hist()
    cumulative_pnl()
    logloss_by_season()


if __name__ == "__main__":
    main()
