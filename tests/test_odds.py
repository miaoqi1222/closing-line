import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from src import odds

FAIR = np.array([[2.0, 4.0, 4.0], [1.25, 10.0, 10.0]])  # 1/odds sums to exactly 1
BOOK = np.array([[2.0, 3.5, 4.0]])  # r = .5, .285714, .25 ; sum 1.035714
METHODS = list(odds.DEVIG_METHODS)


def test_implied_and_overround():
    np.testing.assert_allclose(odds.implied_probabilities(BOOK), [[0.5, 1 / 3.5, 0.25]])
    np.testing.assert_allclose(odds.overround(BOOK), [0.5 + 1 / 3.5 + 0.25 - 1])
    np.testing.assert_allclose(odds.overround(FAIR), [0.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("method", METHODS)
def test_probabilities_sum_to_one(method):
    rng = np.random.default_rng(0)
    p = rng.dirichlet([2, 2, 2], size=500)
    o = 1.0 / (p * (1 + rng.uniform(0.01, 0.12, size=(500, 1))))  # 1-12% overround
    out = odds.devig(o, method)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)
    assert (out > 0).all()


@pytest.mark.parametrize("method", METHODS)
def test_fair_book_is_identity(method):
    np.testing.assert_allclose(odds.devig(FAIR, method), 1.0 / FAIR, atol=1e-10)


def test_worked_examples():
    np.testing.assert_allclose(odds.devig(BOOK, "multiplicative"), [[0.482759, 0.275862, 0.241379]], atol=1e-6)
    # additive: subtract 0.035714 / 3 = 0.011905 from each
    np.testing.assert_allclose(odds.devig(BOOK, "additive"), [[0.488095, 0.273810, 0.238095]], atol=1e-6)


def test_shin_matches_root_finder():
    r = 1.0 / BOOK[0]
    b = r.sum()
    p_of = lambda z: (np.sqrt(z**2 + 4 * (1 - z) * r**2 / b) - z) / (2 * (1 - z))  # noqa: E731
    z_star = brentq(lambda z: p_of(z).sum() - 1.0, 1e-12, 0.5)
    np.testing.assert_allclose(odds.shin_z(BOOK), [z_star], atol=1e-9)
    np.testing.assert_allclose(odds.devig(BOOK, "shin"), [p_of(z_star)], atol=1e-9)
    assert 0.005 < z_star < 0.03  # a ~3.6% overround 3-way book: z a little over 1%


def test_shin_favours_favourite_relative_to_multiplicative():
    o = np.array([[1.3, 5.5, 11.0]])
    m, s = odds.devig(o, "multiplicative"), odds.devig(o, "shin")
    assert s[0, 0] > m[0, 0] and s[0, 2] < m[0, 2]


def test_additive_floor_on_extreme_longshot():
    p = odds.devig(np.array([[1.01, 15.0, 1000.0]]), "additive")  # r_a = .001 < overround/3
    assert p[0, 2] > 0 and np.isclose(p.sum(), 1.0)


@pytest.mark.parametrize("method", METHODS)
def test_nan_rows_propagate(method):
    out = odds.devig(np.array([[2.0, 3.5, 4.0], [np.nan, 3.5, 4.0]]), method)
    assert np.isfinite(out[0]).all() and np.isnan(out[1]).all()


def test_rejects_odds_below_one():
    with pytest.raises(ValueError):
        odds.devig(np.array([[0.9, 3.0, 3.0]]))


def _frame() -> pd.DataFrame:
    nan = np.nan
    return pd.DataFrame(
        {
            "season": ["s"] * 3, "result": ["H", "D", "A"],
            "ps_close_h": [2.0, nan, nan], "ps_close_d": [3.5, nan, nan], "ps_close_a": [4.0, nan, nan],
            "b365_close_h": [2.1, 2.1, nan], "b365_close_d": [3.4, 3.4, nan], "b365_close_a": [3.9, 3.9, nan],
            "wh_close_h": [1.9, 1.9, nan], "wh_close_d": [3.6, 3.6, nan], "wh_close_a": [4.2, 4.2, nan],
            "avg_close_h": [2.0] * 3, "avg_close_d": [3.5] * 3, "avg_close_a": [4.0] * 3,
        }
    )  # fmt: skip


def test_books_with_stage_excludes_aggregates():
    assert odds.books_with_stage(_frame(), "close") == ["b365", "ps", "wh"]
    assert odds.books_with_stage(_frame(), "open") == []


def test_consensus_is_mean_of_devigged_books():
    probs, n = odds.consensus_probabilities(_frame(), "close", "multiplicative", ["b365", "wh"])
    expect = (odds.devig([[2.1, 3.4, 3.9]], "multiplicative") + odds.devig([[1.9, 3.6, 4.2]], "multiplicative")) / 2
    np.testing.assert_allclose(probs[0], expect[0])
    assert n.tolist() == [2, 2, 0] and np.isnan(probs[2]).all()


def test_benchmark_falls_back_to_consensus():
    out = odds.benchmark_probabilities(_frame(), "multiplicative")
    assert out["mkt_source"].tolist() == ["ps", "consensus", "none"]
    assert out["mkt_n_books"].tolist() == [1, 2, 0]
    np.testing.assert_allclose(
        out.loc[0, ["mkt_h", "mkt_d", "mkt_a"]].astype(float), odds.devig(BOOK, "multiplicative")[0]
    )
    assert np.isnan(out.loc[2, "mkt_h"])
