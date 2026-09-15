"""Scoring: proper scores, calibration, CLV, bet selection, staking, and bootstrap CIs.

Everything here takes arrays so it is agnostic to where the probabilities came
from. Conventions:

* ``P`` is an ``(n, 3)`` array of probabilities ordered home / draw / away.
* ``y`` is an ``(n,)`` int array of realised outcomes, 0 / 1 / 2 in the same order.
* Odds are decimal, the book's own price (vig included). Bets are settled at
  that price, never at a devigged "fair" value.
* Stakes are fractions of a constant notional bankroll of 1.0. There is no
  compounding: P&L is additive so that bootstrap intervals are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTCOME_INDEX: dict[str, int] = {"H": 0, "D": 1, "A": 2}
N_BINS = 10
N_BOOT = 2000
SEED = 0
FLAT_STAKE = 0.01  # 1% of bankroll per bet
KELLY_MULT = 0.25  # quarter Kelly
KELLY_CAP = 0.02  # never more than 2% of bankroll on one bet
EPS = 1e-12


def outcome_index(result: pd.Series | np.ndarray) -> np.ndarray:
    """'H'/'D'/'A' -> 0/1/2."""
    return pd.Series(result).map(OUTCOME_INDEX).to_numpy(dtype=int)


def _check(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: P {P.shape}, y {y.shape}")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to 1 per row")
    return P, y


# --- proper scoring rules -------------------------------------------------------


def log_loss(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match negative log likelihood of the realised outcome (natural log)."""
    P, y = _check(P, y)
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))


def brier(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match multiclass Brier score: sum_k (p_k - 1[y = k])^2. Range 0 to 2."""
    P, y = _check(P, y)
    onehot = np.eye(3)[y]
    return ((P - onehot) ** 2).sum(axis=1)


def base_rate(y: np.ndarray) -> np.ndarray:
    """Outcome frequencies as a (3,) prior."""
    y = np.asarray(y, dtype=int)
    return np.bincount(y, minlength=3) / len(y)


# --- calibration ------------------------------------------------------------------


def reliability(
    P: np.ndarray, y: np.ndarray, n_bins: int = N_BINS, outcome: int | None = None
) -> pd.DataFrame:
    """Reliability table with equal-width probability bins.

    ``outcome=None`` pools all three outcome probabilities (3n forecast/event
    pairs); an int restricts to that outcome. Columns: bin_lo, bin_hi, n,
    mean_pred, mean_obs.
    """
    P, y = _check(P, y)
    onehot = np.eye(3)[y]
    if outcome is None:
        p, o = P.ravel(), onehot.ravel()
    else:
        p, o = P[:, outcome], onehot[:, outcome]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": int(m.sum()),
                "mean_pred": p[m].mean() if m.any() else np.nan,
                "mean_obs": o[m].mean() if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ece(P: np.ndarray, y: np.ndarray, n_bins: int = N_BINS, outcome: int | None = None) -> float:
    """Expected calibration error: n-weighted mean |mean_obs - mean_pred| over bins."""
    tab = reliability(P, y, n_bins, outcome)
    tab = tab[tab["n"] > 0]
    w = tab["n"] / tab["n"].sum()
    return float((w * (tab["mean_obs"] - tab["mean_pred"]).abs()).sum())


# --- bootstrap -----------------------------------------------------------------


@dataclass(frozen=True)
class CI:
    value: float
    lo: float
    hi: float
    n: int

    @property
    def spans_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "value": self.value,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "n": self.n,
            "spans_zero": self.spans_zero,
        }


def bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> CI:
    """Percentile bootstrap CI for the mean of ``x``. Deterministic for a given seed."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return CI(np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(float(x.mean()), float(lo), float(hi), n)


def bootstrap_ratio_ci(
    num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05
) -> CI:
    """Bootstrap CI for ``sum(num) / sum(den)`` (e.g. ROI = profit / stake)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = len(num)
    if n == 0 or den.sum() == 0:
        return CI(np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(float(num.sum() / den.sum()), float(lo), float(hi), n)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, **kw: int | float) -> CI:
    """Bootstrap CI for mean(a - b) over identical matches."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    return bootstrap_ci(a - b, **kw)


# --- bets ----------------------------------------------------------------------


def expected_value(P: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Expected return per unit stake at the book's price: p * odds - 1."""
    return np.asarray(P, dtype=float) * np.asarray(odds, dtype=float) - 1.0


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Full-Kelly fraction ``(p * odds - 1) / (odds - 1)``, floored at zero."""
    p = np.asarray(p, dtype=float)
    odds = np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (p * odds - 1.0) / (odds - 1.0)
    return np.where(np.isfinite(f), np.maximum(f, 0.0), 0.0)


def select_bets(P: np.ndarray, odds: np.ndarray, min_edge: float = 0.0) -> np.ndarray:
    """At most one bet per match: the outcome with the largest EV, if EV > ``min_edge``.

    Returns an ``(n,)`` int array of outcome indices, -1 where no bet is taken.
    Matches with any NaN odds are never bet.
    """
    ev = expected_value(P, odds)
    ev = np.where(np.isfinite(ev), ev, -np.inf)
    best = ev.argmax(axis=1)
    take = ev[np.arange(len(best)), best] > min_edge
    return np.where(take, best, -1)


def stake_sizes(
    P: np.ndarray,
    odds: np.ndarray,
    bet: np.ndarray,
    method: str = "flat",
    flat: float = FLAT_STAKE,
    kelly_mult: float = KELLY_MULT,
    cap: float = KELLY_CAP,
) -> np.ndarray:
    """Stake per match as a fraction of bankroll; zero where ``bet == -1``."""
    n = len(bet)
    taken = bet >= 0
    idx = np.where(taken, bet, 0)
    if method == "flat":
        s = np.full(n, flat)
    elif method == "kelly":
        p = P[np.arange(n), idx]
        o = odds[np.arange(n), idx]
        s = np.minimum(kelly_mult * kelly_fraction(p, o), cap)
    else:
        raise ValueError(f"unknown staking method {method!r}")
    return np.where(taken, s, 0.0)


def settle(bet: np.ndarray, odds: np.ndarray, stake: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Profit per match at the book's price: ``stake * (odds - 1)`` if won, else ``-stake``."""
    n = len(bet)
    taken = bet >= 0
    idx = np.where(taken, bet, 0)
    o = odds[np.arange(n), idx]
    won = taken & (idx == y)
    return np.where(taken, np.where(won, stake * (o - 1.0), -stake), 0.0)


def clv(open_odds: np.ndarray, close_odds: np.ndarray) -> np.ndarray:
    """Closing line value of a bet struck at ``open_odds``: ``open / close - 1``.

    Positive means the price moved against the bettor's side after they bet,
    i.e. the market later agreed with them. Both prices from the same book.
    """
    return np.asarray(open_odds, dtype=float) / np.asarray(close_odds, dtype=float) - 1.0


def bet_frame(
    P: np.ndarray,
    y: np.ndarray,
    open_odds: np.ndarray,
    close_odds: np.ndarray,
    close_probs: np.ndarray,
    min_edge: float = 0.0,
) -> pd.DataFrame:
    """Per-match bet ledger for a forecaster betting at ``open_odds``.

    Columns: bet (outcome index or -1), model_p, open_odds, close_odds,
    close_p (devigged closing probability of the bet side), ev_open (model EV
    at the price taken), clv (open/close - 1), ev_at_close (close_p * open_odds
    - 1: the bet's expected return if the closing line is the truth),
    stake_flat, stake_kelly, pnl_flat, pnl_kelly, won.
    """
    P, y = _check(P, y)
    n = len(y)
    bet = select_bets(P, open_odds, min_edge)
    taken = bet >= 0
    idx = np.where(taken, bet, 0)
    r = np.arange(n)
    out = pd.DataFrame(
        {
            "bet": bet,
            "model_p": np.where(taken, P[r, idx], np.nan),
            "open_odds": np.where(taken, open_odds[r, idx], np.nan),
            "close_odds": np.where(taken, close_odds[r, idx], np.nan),
            "close_p": np.where(taken, close_probs[r, idx], np.nan),
        }
    )
    out["ev_open"] = out["model_p"] * out["open_odds"] - 1.0
    out["clv"] = clv(out["open_odds"], out["close_odds"])
    out["ev_at_close"] = out["close_p"] * out["open_odds"] - 1.0
    out["stake_flat"] = stake_sizes(P, open_odds, bet, "flat")
    out["stake_kelly"] = stake_sizes(P, open_odds, bet, "kelly")
    out["pnl_flat"] = settle(bet, open_odds, out["stake_flat"].to_numpy(), y)
    out["pnl_kelly"] = settle(bet, open_odds, out["stake_kelly"].to_numpy(), y)
    out["won"] = np.where(taken, idx == y, False)
    return out


# --- headline scoring ---------------------------------------------------------------


def score_probabilities(
    P: np.ndarray, y: np.ndarray, label: str, seed: int = SEED, n_bins: int = N_BINS
) -> pd.DataFrame:
    """Long-form table: log loss, Brier, ECE (pooled and per outcome) with bootstrap CIs."""
    P, y = _check(P, y)
    rows = [
        {
            "forecaster": label,
            "metric": "log_loss",
            **bootstrap_ci(log_loss(P, y), seed=seed).as_dict(),
        },
        {"forecaster": label, "metric": "brier", **bootstrap_ci(brier(P, y), seed=seed).as_dict()},
    ]
    # ECE is a functional of the whole sample; bootstrap it by resampling matches.
    rng = np.random.default_rng(seed)
    n = len(y)
    for name, outcome in [("ece", None), ("ece_h", 0), ("ece_d", 1), ("ece_a", 2)]:
        point = ece(P, y, n_bins, outcome)
        boots = np.empty(N_BOOT // 4)  # ECE is slow-ish; 500 resamples is plenty for a CI
        for i in range(len(boots)):
            idx = rng.integers(0, n, size=n)
            boots[i] = ece(P[idx], y[idx], n_bins, outcome)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(
            {"forecaster": label, "metric": name, **CI(point, float(lo), float(hi), n).as_dict()}
        )
    return pd.DataFrame(rows)


def score_bets(ledger: pd.DataFrame, label: str, seed: int = SEED) -> pd.DataFrame:
    """Long-form table of betting metrics for one ledger from ``bet_frame``."""
    b = ledger[ledger["bet"] >= 0]
    rows = []

    def add(metric: str, ci: CI) -> None:
        rows.append({"forecaster": label, "metric": metric, **ci.as_dict()})

    add("n_bets", CI(float(len(b)), np.nan, np.nan, len(b)))
    add("bet_rate", CI(len(b) / max(len(ledger), 1), np.nan, np.nan, len(ledger)))
    if len(b) == 0:
        return pd.DataFrame(rows)
    add("clv_mean", bootstrap_ci(b["clv"].to_numpy(), seed=seed))
    add("ev_at_close_mean", bootstrap_ci(b["ev_at_close"].to_numpy(), seed=seed))
    add("hit_rate", bootstrap_ci(b["won"].to_numpy(dtype=float), seed=seed))
    add(
        "roi_flat",
        bootstrap_ratio_ci(b["pnl_flat"].to_numpy(), b["stake_flat"].to_numpy(), seed=seed),
    )
    add(
        "roi_kelly",
        bootstrap_ratio_ci(b["pnl_kelly"].to_numpy(), b["stake_kelly"].to_numpy(), seed=seed),
    )
    add("pnl_flat_total", CI(float(b["pnl_flat"].sum()), np.nan, np.nan, len(b)))
    add("pnl_kelly_total", CI(float(b["pnl_kelly"].sum()), np.nan, np.nan, len(b)))
    add("mean_odds_taken", CI(float(b["open_odds"].mean()), np.nan, np.nan, len(b)))
    return pd.DataFrame(rows)


def compare_probabilities(
    P_a: np.ndarray, P_b: np.ndarray, y: np.ndarray, label: str, seed: int = SEED
) -> pd.DataFrame:
    """Paired differences (a minus b) in log loss and Brier, with CIs. Negative favours a."""
    rows = []
    for metric, fn in [("log_loss", log_loss), ("brier", brier)]:
        ci = paired_diff_ci(fn(P_a, y), fn(P_b, y), seed=seed)
        rows.append({"forecaster": label, "metric": f"{metric}_diff", **ci.as_dict()})
    return pd.DataFrame(rows)
