"""Scoring: proper scores, calibration, bootstrap CIs, bets, CLV.

Conventions: ``P`` is ``(n, 3)`` probabilities home / draw / away; ``y`` is ``(n,)``
ints 0 / 1 / 2; odds are the book's own decimal price (vig included) and bets
settle at that price. Stakes are fractions of a constant bankroll of 1.0, no
compounding, so P&L is additive and bootstrap intervals are meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10
N_BOOT = 2000
SEED = 0
FLAT_STAKE = 0.01  # 1% of bankroll per bet
KELLY_MULT = 0.25  # quarter Kelly
KELLY_CAP = 0.02  # never more than 2% of bankroll on one bet


def outcome_index(result: pd.Series) -> np.ndarray:
    """'H'/'D'/'A' -> 0/1/2."""
    return pd.Series(result).map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)


def _check(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    if P.shape != (len(y), 3) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("P must be (n, 3) rows summing to 1, y must be (n,)")
    return P, y


# --- proper scoring rules and calibration ---------------------------------------


def log_loss(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match negative log likelihood of the realised outcome."""
    P, y = _check(P, y)
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def brier(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-match multiclass Brier score, sum_k (p_k - 1[y = k])^2; range 0 to 2."""
    P, y = _check(P, y)
    return ((P - np.eye(3)[y]) ** 2).sum(axis=1)


def base_rate(y: np.ndarray) -> np.ndarray:
    """Outcome frequencies as a (3,) prior."""
    return np.bincount(np.asarray(y), minlength=3) / len(y)


def _bins(P: np.ndarray, y: np.ndarray, n_bins: int, outcome: int | None):
    """Forecasts, events and bin ids, pooled over outcomes or for one outcome."""
    P, y = _check(P, y)
    p, o = (P.ravel(), np.eye(3)[y].ravel()) if outcome is None else (P[:, outcome], (y == outcome) * 1.0)
    return p, o, np.minimum((p * n_bins).astype(int), n_bins - 1)


def reliability(P: np.ndarray, y: np.ndarray, n_bins: int = N_BINS, outcome: int | None = None):
    """Equal-width reliability table: bin_lo, bin_hi, n, mean_pred, mean_obs."""
    p, o, b = _bins(P, y, n_bins, outcome)
    n = np.bincount(b, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        pred = np.bincount(b, p, n_bins) / n
        obs = np.bincount(b, o, n_bins) / n
    edges = np.linspace(0, 1, n_bins + 1)
    return pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "n": n, "mean_pred": pred, "mean_obs": obs})


def ece(P: np.ndarray, y: np.ndarray, n_bins: int = N_BINS, outcome: int | None = None) -> float:
    """Expected calibration error: sum_b |sum_b(obs) - sum_b(pred)| / N."""
    p, o, b = _bins(P, y, n_bins, outcome)
    return float(np.abs(np.bincount(b, o, n_bins) - np.bincount(b, p, n_bins)).sum() / len(p))


# --- bootstrap -----------------------------------------------------------------


def _ci(value: float, samples: np.ndarray, n: int) -> dict:
    lo, hi = np.percentile(samples, [2.5, 97.5]) if len(samples) else (np.nan, np.nan)
    return {"value": float(value), "ci_lo": float(lo), "ci_hi": float(hi), "n": n, "spans_zero": bool(lo <= 0 <= hi)}


def _resample(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def bootstrap_ci(x: np.ndarray, seed: int = SEED) -> dict:
    """Percentile bootstrap CI for the mean of ``x``: value, ci_lo, ci_hi, n, spans_zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return _ci(np.nan, np.array([]), 0)
    return _ci(x.mean(), x[_resample(len(x), seed)].mean(axis=1), len(x))


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, seed: int = SEED) -> dict:
    """Bootstrap CI for ``sum(num) / sum(den)``, e.g. ROI = profit / stake."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if len(num) == 0:
        return _ci(np.nan, np.array([]), 0)
    idx = _resample(len(num), seed)
    return _ci(num.sum() / den.sum(), num[idx].sum(axis=1) / den[idx].sum(axis=1), len(num))


def bootstrap_ece_ci(P: np.ndarray, y: np.ndarray, outcome: int | None = None, seed: int = SEED) -> dict:
    """ECE with a bootstrap CI over matches, computed as one matrix product.

    Per match, per bin: S = sum(obs) - sum(pred). Resampling matches with multiplicity W
    gives every replicate's ECE as ``|W @ S|.sum(axis=1) / N``.
    """
    p, o, b = _bins(P, y, N_BINS, outcome)
    n, per = len(y), len(p) // len(y)  # `per` = 3 pooled, 1 for a single outcome
    match = np.arange(len(p)) // per
    S = np.bincount(match * N_BINS + b, o - p, n * N_BINS).reshape(n, N_BINS)
    idx = _resample(n, seed)
    W = np.bincount((idx + n * np.arange(N_BOOT)[:, None]).ravel(), minlength=N_BOOT * n)
    boots = np.abs(W.reshape(N_BOOT, n) @ S).sum(axis=1) / len(p)
    return _ci(np.abs(S.sum(axis=0)).sum() / len(p), boots, n)


# --- bets ----------------------------------------------------------------------


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Full-Kelly fraction ``(p * odds - 1) / (odds - 1)``, floored at zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (p * odds - 1) / (odds - 1)
    return np.where(np.isfinite(f), np.maximum(f, 0), 0.0)


def select_bets(P: np.ndarray, odds: np.ndarray, min_edge: float = 0.0) -> np.ndarray:
    """At most one bet per match: the max-EV outcome if its EV exceeds ``min_edge``, else -1."""
    ev = np.where(np.isfinite(odds), P * odds - 1, -np.inf)
    best = ev.argmax(axis=1)
    return np.where(ev[np.arange(len(best)), best] > min_edge, best, -1)


def bet_frame(
    P: np.ndarray, y: np.ndarray, open_odds: np.ndarray, close_odds: np.ndarray, close_probs: np.ndarray
) -> pd.DataFrame:
    """Per-match ledger for a forecaster betting at ``open_odds`` and settling there.

    ``clv`` is open / close - 1 at the same book; ``ev_at_close`` is close_p * open_odds - 1,
    the bet's expected return if the closing line is the truth.
    """
    P, y = _check(P, y)
    bet = select_bets(P, open_odds)
    taken, r, k = bet >= 0, np.arange(len(y)), np.maximum(bet, 0)
    pick = lambda a: np.where(taken, a[r, k], np.nan)  # noqa: E731
    out = pd.DataFrame({"bet": bet, "model_p": pick(P), "open_odds": pick(open_odds), "close_p": pick(close_probs)})
    out["clv"] = out["open_odds"] / pick(close_odds) - 1
    out["ev_at_close"] = out["close_p"] * out["open_odds"] - 1
    out["stake_flat"] = np.where(taken, FLAT_STAKE, 0.0)
    out["stake_kelly"] = np.where(
        taken, np.minimum(KELLY_MULT * kelly_fraction(pick(P), pick(open_odds)), KELLY_CAP), 0.0
    )
    out["won"] = taken & (k == y)
    payoff = np.where(out["won"], out["open_odds"].fillna(1) - 1, -1.0)
    out["pnl_flat"] = out["stake_flat"] * payoff
    out["pnl_kelly"] = out["stake_kelly"] * payoff
    return out


# --- tables ----------------------------------------------------------------------


def _rows(label: str, **metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([{"forecaster": label, "metric": m, **ci} for m, ci in metrics.items()])


def score_probabilities(P: np.ndarray, y: np.ndarray, label: str, seed: int = SEED) -> pd.DataFrame:
    """Long table: log loss, Brier, ECE (pooled and per outcome), each with a bootstrap CI."""
    return _rows(
        label,
        log_loss=bootstrap_ci(log_loss(P, y), seed),
        brier=bootstrap_ci(brier(P, y), seed),
        ece=bootstrap_ece_ci(P, y, None, seed),
        ece_h=bootstrap_ece_ci(P, y, 0, seed),
        ece_d=bootstrap_ece_ci(P, y, 1, seed),
        ece_a=bootstrap_ece_ci(P, y, 2, seed),
    )


def score_bets(ledger: pd.DataFrame, label: str, seed: int = SEED) -> pd.DataFrame:
    """Long table of betting metrics for one ledger from ``bet_frame``."""
    b = ledger[ledger["bet"] >= 0]
    return _rows(
        label,
        n_bets=_ci(len(b), np.array([]), len(b)),
        clv_mean=bootstrap_ci(b["clv"], seed),
        ev_at_close_mean=bootstrap_ci(b["ev_at_close"], seed),
        hit_rate=bootstrap_ci(b["won"].astype(float), seed),
        roi_flat=bootstrap_ratio_ci(b["pnl_flat"], b["stake_flat"], seed),
        roi_kelly=bootstrap_ratio_ci(b["pnl_kelly"], b["stake_kelly"], seed),
        pnl_flat_total=_ci(b["pnl_flat"].sum(), np.array([]), len(b)),
        pnl_kelly_total=_ci(b["pnl_kelly"].sum(), np.array([]), len(b)),
    )


def compare_probabilities(P_a: np.ndarray, P_b: np.ndarray, y: np.ndarray, label: str, seed: int = SEED):
    """Paired differences (a minus b) in log loss and Brier with CIs. Negative favours a."""
    return _rows(
        label,
        log_loss_diff=bootstrap_ci(log_loss(P_a, y) - log_loss(P_b, y), seed),
        brier_diff=bootstrap_ci(brier(P_a, y) - brier(P_b, y), seed),
    )
