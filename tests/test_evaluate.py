import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev

P = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3]])
Y = np.array([0, 2, 1])


def test_outcome_index():
    assert ev.outcome_index(pd.Series(["H", "D", "A"])).tolist() == [0, 1, 2]


def test_scores_known_values():
    np.testing.assert_allclose(ev.log_loss(P, Y), [-np.log(0.5), -np.log(0.5), np.log(3)])
    np.testing.assert_allclose(ev.brier(P, Y)[0], 0.38)  # (0.5-1)^2 + 0.3^2 + 0.2^2
    perfect = np.eye(3)[Y]
    np.testing.assert_allclose(ev.brier(perfect, Y), 0.0)
    np.testing.assert_allclose(ev.log_loss(perfect, Y), 0.0, atol=1e-9)
    with pytest.raises(ValueError):
        ev.log_loss(np.array([[0.5, 0.5, 0.5]]), np.array([0]))


def test_base_rate():
    np.testing.assert_allclose(ev.base_rate(np.array([0, 0, 1, 2])), [0.5, 0.25, 0.25])


def test_calibrated_synthetic_has_small_ece():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([2, 2, 2], size=60000)
    y = (rng.random(len(p))[:, None] > p.cumsum(axis=1)).sum(axis=1)
    tab = ev.reliability(p, y)
    assert tab["n"].sum() == 3 * len(p)
    big = tab["n"] > 200
    assert (np.abs(tab.loc[big, "mean_obs"] - tab.loc[big, "mean_pred"]) < 0.02).all()
    assert ev.ece(p, y) < 0.01 and ev.ece(p, y, outcome=0) < 0.02
    ci = ev.bootstrap_ece_ci(p, y)
    assert ci["ci_lo"] <= ci["value"] <= ci["ci_hi"] and ci["n"] == len(p)


def test_reliability_bins_cover_edges():
    tab = ev.reliability(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), np.array([2, 0]))
    assert tab["n"].sum() == 6 and tab["n"].iloc[0] == 4 and tab["n"].iloc[-1] == 2


def test_ece_bootstrap_point_matches_direct():
    rng = np.random.default_rng(2)
    p = rng.dirichlet([1, 1, 1], size=500)
    y = rng.integers(0, 3, 500)
    for outcome in (None, 0, 1, 2):
        np.testing.assert_allclose(ev.bootstrap_ece_ci(p, y, outcome)["value"], ev.ece(p, y, outcome=outcome))


def test_bootstrap_ci():
    x = np.random.default_rng(3).normal(1.0, 1.0, size=500)
    a, b = ev.bootstrap_ci(x, seed=7), ev.bootstrap_ci(x, seed=7)
    assert a == b and a["ci_lo"] < a["value"] < a["ci_hi"] and a["n"] == 500 and not a["spans_zero"]
    assert ev.bootstrap_ci(x - x.mean())["spans_zero"]
    assert ev.bootstrap_ci(np.array([np.nan, 1.0, 3.0]))["n"] == 2
    assert ev.bootstrap_ci(np.array([]))["n"] == 0
    same = ev.bootstrap_ci(np.arange(10.0) - np.arange(10.0))
    assert same["value"] == same["ci_lo"] == same["ci_hi"] == 0.0


def test_bootstrap_ratio_ci():
    ci = ev.bootstrap_ratio_ci(np.array([1.0, -1.0, 2.0]), np.ones(3))
    np.testing.assert_allclose(ci["value"], 2 / 3)
    assert ci["ci_lo"] <= ci["value"] <= ci["ci_hi"]


def test_kelly_fraction():
    np.testing.assert_allclose(ev.kelly_fraction(np.array([0.6, 0.4]), np.array([2.0, 2.0])), [0.2, 0.0])


def test_select_bets_picks_max_ev_only_if_positive():
    odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [np.nan, 3.0, 4.0]])
    p = np.array([[0.6, 0.2, 0.2], [0.45, 0.30, 0.25], [0.9, 0.05, 0.05]])
    # row 0: EVs 0.2, -0.4, -0.2 -> home; row 1: -0.1, -0.1, 0.0 -> none; row 2: NaN side never bet
    assert ev.select_bets(p, odds).tolist() == [0, -1, -1]
    assert ev.select_bets(p, odds, min_edge=0.3).tolist() == [-1, -1, -1]


def test_bet_frame_and_scores():
    n, y = 6, np.array([0, 1, 2, 0, 1, 2])
    p = np.tile([0.6, 0.2, 0.2], (n, 1))
    open_odds, close_odds = np.tile([2.0, 3.5, 4.0], (n, 1)), np.tile([1.9, 3.6, 4.2], (n, 1))
    close_p = np.tile([0.5, 0.27, 0.23], (n, 1))
    led = ev.bet_frame(p, y, open_odds, close_odds, close_p)
    assert (led["bet"] == 0).all() and led["won"].sum() == 2
    np.testing.assert_allclose(led["clv"], 2.0 / 1.9 - 1)
    np.testing.assert_allclose(led["ev_at_close"], 0.5 * 2.0 - 1)
    np.testing.assert_allclose(led["stake_flat"], 0.01)
    np.testing.assert_allclose(led["stake_kelly"], 0.02)  # quarter of 0.2 = 0.05, capped at 0.02
    np.testing.assert_allclose(led["pnl_flat"].sum(), 2 * 0.01 - 4 * 0.01)
    tab = ev.score_bets(led, "x").set_index("metric")
    assert tab.loc["n_bets", "value"] == 6
    np.testing.assert_allclose(tab.loc["roi_flat", "value"], -1 / 3)
    assert set(ev.score_probabilities(p, y, "x")["metric"]) == {"log_loss", "brier", "ece", "ece_h", "ece_d", "ece_a"}
    assert (ev.compare_probabilities(p, p, y, "x")["value"] == 0).all()


def test_no_bet_when_no_odds():
    led = ev.bet_frame(P, Y, np.full((3, 3), np.nan), np.full((3, 3), np.nan), P)
    assert (led["bet"] == -1).all() and led["pnl_flat"].sum() == 0
    assert ev.score_bets(led, "x").set_index("metric").loc["n_bets", "value"] == 0
