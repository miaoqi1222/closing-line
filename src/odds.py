"""The market side: implied probabilities, overround removal, consensus and benchmark.

Odds are decimal ``(n, 3)`` arrays ordered home / draw / away; probabilities come
back in the same shape and NaN rows stay NaN. Devig methods share ``fn(odds) -> P``:

* ``multiplicative``: divide raw implied probabilities by their sum.
* ``additive``: subtract the same amount from each (floored so long shots stay > 0).
* ``shin``: Shin (1993) insider-trading model. Solves the insider fraction ``z``
  per row so the probabilities sum to one; shifts probability towards favourites
  relative to multiplicative (the favourite-longshot bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("h", "d", "a")
BENCHMARK_BOOK = "ps"
AGGREGATES = {"max", "avg"}  # market-wide columns, not books; excluded from the consensus


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Raw ``1 / odds``; rows sum to more than one at a real book."""
    odds = np.asarray(odds, dtype=float)
    if np.nanmin(odds, initial=np.inf) < 1.0:
        raise ValueError("decimal odds below 1.0 are not valid")
    return 1.0 / odds


def overround(odds: np.ndarray) -> np.ndarray:
    """``sum(1 / odds) - 1`` per row; zero for a fair book."""
    return implied_probabilities(odds).sum(axis=1) - 1.0


def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    r = implied_probabilities(odds)
    return r / r.sum(axis=1, keepdims=True)


def devig_additive(odds: np.ndarray) -> np.ndarray:
    r = implied_probabilities(odds)
    p = np.maximum(r - (r.sum(axis=1, keepdims=True) - 1.0) / r.shape[1], 1e-4)
    return p / p.sum(axis=1, keepdims=True)


def _shin(r: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    z = z[:, None]
    return (np.sqrt(z**2 + 4 * (1 - z) * r**2 / b) - z) / (2 * (1 - z))


def shin_z(odds: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """Shin's insider fraction per row by bisection: ``sum_i p_i(z) - 1`` is decreasing in z."""
    r = implied_probabilities(odds)
    b = r.sum(axis=1, keepdims=True)
    lo, hi = np.zeros(len(r)), np.full(len(r), 1 - 1e-9)
    with np.errstate(invalid="ignore"):
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            above = _shin(r, b, mid).sum(axis=1) > 1
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    z = np.where(b[:, 0] <= 1, 0.0, (lo + hi) / 2)  # fair book: no insiders
    return np.where(np.isfinite(r).all(axis=1), z, np.nan)


def devig_shin(odds: np.ndarray) -> np.ndarray:
    r = implied_probabilities(odds)
    return _shin(r, r.sum(axis=1, keepdims=True), shin_z(odds))


DEVIG_METHODS = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "shin": devig_shin,
}


def devig(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    return DEVIG_METHODS[method](odds)


# --- frame-level helpers -------------------------------------------------------


def odds_array(df: pd.DataFrame, book: str, stage: str) -> np.ndarray:
    """``(n, 3)`` odds for one book/stage; a row is all-NaN unless all three prices exist."""
    cols = [f"{book}_{stage}_{o}" for o in OUTCOMES]
    if not set(cols) <= set(df.columns):
        return np.full((len(df), 3), np.nan)
    arr = df[cols].to_numpy(dtype=float, copy=True)
    arr[np.isnan(arr).any(axis=1)] = np.nan
    return arr


def books_with_stage(df: pd.DataFrame, stage: str) -> list[str]:
    """Individual books (no aggregates) with a full H/D/A triple of columns for ``stage``."""
    codes = {c.split("_")[0] for c in df.columns if f"_{stage}_" in c} - AGGREGATES
    return sorted(b for b in codes if set(f"{b}_{stage}_{o}" for o in OUTCOMES) <= set(df))


def book_probabilities(df: pd.DataFrame, book: str, stage: str = "close", method: str = "shin"):
    """Devigged ``(n, 3)`` probabilities for one book; NaN rows where it is missing."""
    return devig(odds_array(df, book, stage), method)


def consensus_probabilities(
    df: pd.DataFrame, stage: str = "close", method: str = "shin", books: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of devigged probabilities across the books priced for each match; ``(P, n_books)``."""
    books = books_with_stage(df, stage) if books is None else books
    stack = np.stack([book_probabilities(df, b, stage, method) for b in books])
    n_books = np.isfinite(stack[:, :, 0]).sum(axis=0)
    with np.errstate(invalid="ignore"):
        probs = np.nansum(stack, axis=0) / n_books[:, None]
    probs[n_books == 0] = np.nan
    return probs, n_books


def benchmark_probabilities(df: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """The closing-line benchmark: Pinnacle closing where priced, else the consensus.

    Columns ``mkt_h, mkt_d, mkt_a``, ``mkt_source`` ('ps' / 'consensus' / 'none') and
    ``mkt_n_books`` (books behind the number; 1 for Pinnacle).
    """
    primary = book_probabilities(df, BENCHMARK_BOOK, "close", method)
    have = np.isfinite(primary[:, 0])
    others = [b for b in books_with_stage(df, "close") if b != BENCHMARK_BOOK]
    cons, n_books = consensus_probabilities(df, "close", method, others)
    out = pd.DataFrame(np.where(have[:, None], primary, cons), columns=["mkt_h", "mkt_d", "mkt_a"])
    out["mkt_source"] = np.where(have, BENCHMARK_BOOK, np.where(n_books > 0, "consensus", "none"))
    out["mkt_n_books"] = np.where(have, 1, n_books)
    return out.set_index(df.index)
