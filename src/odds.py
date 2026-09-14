"""The market side: implied probabilities, overround removal, consensus and benchmark.

All functions take decimal odds as an ``(n, 3)`` array ordered home / draw / away
and return probabilities in the same shape. Rows containing NaN propagate NaN.

Devig methods share one interface, ``fn(odds) -> probs``:

* ``multiplicative``: divide each raw implied probability by their sum.
* ``additive``: subtract the same amount from each raw implied probability. This
  can push a long shot below zero; we floor at ``ADDITIVE_FLOOR`` and renormalise,
  and ``additive_clipped`` reports how often that happens.
* ``shin``: Shin (1993) insider-trading model. The bookmaker's prices are the
  solution of a market with a fraction ``z`` of informed bettors; ``z`` is solved
  per row so the devigged probabilities sum to one. Moves more probability
  towards favourites than multiplicative normalisation does (the favourite-
  longshot bias).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

OUTCOMES: tuple[str, str, str] = ("h", "d", "a")
ADDITIVE_FLOOR = 1e-4
BENCHMARK_BOOK = "ps"
# Individual bookmakers only. ``max`` and ``avg`` are market aggregates, not books.
AGGREGATE_BOOKS: frozenset[str] = frozenset({"max", "avg"})


def _as_odds(odds: np.ndarray | pd.DataFrame) -> np.ndarray:
    arr = np.asarray(odds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"odds must be (n, k>=2), got {arr.shape}")
    if np.nanmin(arr, initial=np.inf) < 1.0:
        raise ValueError("decimal odds below 1.0 are not valid")
    return arr


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Raw implied probabilities ``1 / odds`` (rows sum to more than one at a real book)."""
    return 1.0 / _as_odds(odds)


def overround(odds: np.ndarray) -> np.ndarray:
    """Book overround per row: ``sum(1 / odds) - 1``. Zero for a fair book."""
    return implied_probabilities(odds).sum(axis=1) - 1.0


def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    r = implied_probabilities(odds)
    return r / r.sum(axis=1, keepdims=True)


def devig_additive(odds: np.ndarray) -> np.ndarray:
    r = implied_probabilities(odds)
    k = r.shape[1]
    p = r - (r.sum(axis=1, keepdims=True) - 1.0) / k
    p = np.where(np.isnan(p), np.nan, np.maximum(p, ADDITIVE_FLOOR))
    return p / p.sum(axis=1, keepdims=True)


def additive_clipped(odds: np.ndarray) -> np.ndarray:
    """Boolean per row: did the additive method need flooring?"""
    r = implied_probabilities(odds)
    p = r - (r.sum(axis=1, keepdims=True) - 1.0) / r.shape[1]
    return (p < ADDITIVE_FLOOR).any(axis=1)


def _shin_probs(r: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    zz = z[:, None]
    return (np.sqrt(zz**2 + 4.0 * (1.0 - zz) * r**2 / b) - zz) / (2.0 * (1.0 - zz))


def shin_z(odds: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """Shin's insider fraction ``z`` per row, by vectorised bisection.

    With ``r`` the raw implied probabilities and ``B`` their sum, the devigged
    probabilities are ``p_i(z) = (sqrt(z^2 + 4 (1 - z) r_i^2 / B) - z) / (2 (1 - z))``
    and ``z`` is the root of ``sum_i p_i(z) - 1``. That function is strictly
    decreasing in ``z`` and positive at ``z = 0`` whenever ``B > 1``, so bisection
    on ``[0, 1)`` is exact to machine precision after ``n_iter`` halvings. Rows
    with ``B <= 1`` (a fair or negative-margin book) get ``z = 0``.
    """
    r = implied_probabilities(odds)
    if r.shape[1] < 3:
        raise ValueError("Shin's model needs at least three outcomes")
    b = r.sum(axis=1, keepdims=True)
    ok = np.isfinite(r).all(axis=1)
    lo = np.zeros(r.shape[0])
    hi = np.full(r.shape[0], 1.0 - 1e-9)
    with np.errstate(invalid="ignore"):
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            f = _shin_probs(r, b, mid).sum(axis=1) - 1.0
            pos = f > 0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
    z = 0.5 * (lo + hi)
    z = np.where(b[:, 0] <= 1.0, 0.0, z)
    return np.where(ok, z, np.nan)


def devig_shin(odds: np.ndarray) -> np.ndarray:
    r = implied_probabilities(odds)
    b = r.sum(axis=1, keepdims=True)
    return _shin_probs(r, b, shin_z(odds))


DEVIG_METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "shin": devig_shin,
}


def devig(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    """Remove the overround with the named method. See ``DEVIG_METHODS``."""
    try:
        fn = DEVIG_METHODS[method]
    except KeyError:
        raise ValueError(f"unknown devig method {method!r}") from None
    return fn(odds)


# --- frame-level helpers -------------------------------------------------------


def odds_array(df: pd.DataFrame, book: str, stage: str) -> np.ndarray:
    """``(n, 3)`` decimal odds for one book/stage; all-NaN rows where the book is absent."""
    cols = [f"{book}_{stage}_{o}" for o in OUTCOMES]
    if not all(c in df.columns for c in cols):
        return np.full((len(df), 3), np.nan)
    arr = df[cols].to_numpy(dtype=float, copy=True)
    incomplete = np.isnan(arr).any(axis=1)
    arr[incomplete] = np.nan
    return arr


def books_with_stage(df: pd.DataFrame, stage: str, include_aggregates: bool = False) -> list[str]:
    """Book codes with a full H/D/A triple of columns for ``stage``."""
    cols = set(df.columns)
    out = []
    for code in sorted({c.split("_")[0] for c in cols if c.count("_") == 2}):
        if not include_aggregates and code in AGGREGATE_BOOKS:
            continue
        if all(f"{code}_{stage}_{o}" in cols for o in OUTCOMES):
            out.append(code)
    return out


def book_probabilities(
    df: pd.DataFrame, book: str, stage: str = "close", method: str = "shin"
) -> np.ndarray:
    """Devigged ``(n, 3)`` probabilities for one book; NaN rows where it is missing."""
    return devig(odds_array(df, book, stage), method)


def consensus_probabilities(
    df: pd.DataFrame,
    stage: str = "close",
    method: str = "shin",
    books: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of devigged probabilities across the individual books priced for each match.

    Returns ``(probs, n_books)``. ``probs`` is NaN where no book prices the match.
    Aggregate columns (``max``, ``avg``) are excluded: they are derived from the
    same books and would double count.
    """
    if books is None:
        books = books_with_stage(df, stage)
    stack = np.stack([book_probabilities(df, b, stage, method) for b in books], axis=0)
    present = ~np.isnan(stack).any(axis=2)
    n_books = present.sum(axis=0)
    with np.errstate(invalid="ignore"):
        probs = np.nansum(stack, axis=0) / n_books[:, None]
    probs[n_books == 0] = np.nan
    return probs, n_books


def benchmark_probabilities(
    df: pd.DataFrame, method: str = "shin", book: str = BENCHMARK_BOOK
) -> pd.DataFrame:
    """The closing-line benchmark: ``book`` closing where priced, else the consensus.

    Columns: ``mkt_h, mkt_d, mkt_a`` (devigged probabilities), ``mkt_source``
    (``book`` or ``"consensus"``), ``mkt_n_books`` (books behind the number:
    1 for the primary book), and ``mkt_overround`` of the book actually used.
    """
    primary = book_probabilities(df, book, "close", method)
    have_primary = ~np.isnan(primary).any(axis=1)
    others = [b for b in books_with_stage(df, "close") if b != book]
    cons, n_books = consensus_probabilities(df, "close", method, others)
    probs = np.where(have_primary[:, None], primary, cons)
    source = np.where(have_primary, book, np.where(n_books > 0, "consensus", "none"))
    n = np.where(have_primary, 1, n_books)
    ov_primary = overround(odds_array(df, book, "close"))
    ov_stack = np.stack([overround(odds_array(df, b, "close")) for b in others], axis=0)
    ov_count = np.isfinite(ov_stack).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ov_others = np.nansum(ov_stack, axis=0) / ov_count
    ov_others[ov_count == 0] = np.nan
    ov = np.where(have_primary, ov_primary, ov_others)
    out = pd.DataFrame(probs, columns=[f"mkt_{o}" for o in OUTCOMES], index=df.index)
    out["mkt_source"] = source
    out["mkt_n_books"] = n
    out["mkt_overround"] = ov
    return out
