import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev

P = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3]])
Y = np.array([0, 2, 1])


def test_outcome_index():
    assert ev.outcome_index(pd.Series(["H", "D", "A"])).tolist() == [0, 1, 2]


def test_log_loss_values():
    np.testing.assert_allclose(ev.log_loss(P, Y), [-np.log(0.5), -np.log(0.5), np.log(3)])


def test_brier_values():
    # row 0: (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
    np.testing.assert_allclose(ev.brier(P, Y)[0], 0.38)
    perfect = np.eye(3)[Y]
    np.testing.assert_allclose(ev.brier(perfect, Y), 0.0)
    np.testing.assert_allclose(ev.log_loss(perfect, Y), 0.0, atol=1e-9)


def test_rejects_non_normalised():
    with pytest.raises(ValueError):
        ev.log_loss(np.array([[0.5, 0.5, 0.5]]), np.array([0]))


def test_base_rate():
    np.testing.assert_allclose(ev.base_rate(np.array([0, 0, 1, 2])), [0.5, 0.25, 0.25])


def test_reliability_perfectly_calibrated_synthetic():
    rng = np.random.default_rng(1)
    n = 60000
    p = rng.dirichlet([2, 2, 2], size=n)
    y = np.array([rng.choice(3, p=row) for row in p])
    tab = ev.reliability(p, y)
    assert tab["n"].sum() == 3 * n
    ok = tab["n"] > 200
    assert (np.abs(tab.loc[ok, "mean_obs"] - tab.loc[ok, "mean_pred"]) < 0.02).all()
    assert ev.ece(p, y) < 0.01
    assert ev.ece(p, y, outcome=0) < 0.02


def test_reliability_bins_cover_edges():
    p = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    tab = ev.reliability(p, np.array([2, 0]))
    assert tab["n"].sum() == 6
    assert tab.iloc[0]["n"] == 4 and tab.iloc[-1]["n"] == 2


def test_bootstrap_ci_deterministic_and_contains_mean():
    x = np.random.default_rng(3).normal(1.0, 1.0, size=500)
    a = ev.bootstrap_ci(x, seed=7)
    b = ev.bootstrap_ci(x, seed=7)
    assert a == b
    assert a.lo < a.value < a.hi
    assert a.n == 500
    assert not a.spans_zero
    z = ev.bootstrap_ci(x - x.mean(), seed=7)
    assert z.spans_zero


def test_bootstrap_ci_ignores_nan_and_empty():
    assert ev.bootstrap_ci(np.array([np.nan, 1.0, 3.0])).n == 2
    assert ev.bootstrap_ci(np.array([])).n == 0


def test_paired_diff_identical_is_zero():
    x = np.arange(10.0)
    ci = ev.paired_diff_ci(x, x)
    assert ci.value == 0.0 and ci.lo == 0.0 and ci.hi == 0.0


def test_bootstrap_ratio_ci():
    num = np.array([1.0, -1.0, 2.0])
    den = np.array([1.0, 1.0, 1.0])
    ci = ev.bootstrap_ratio_ci(num, den)
    np.testing.assert_allclose(ci.value, 2 / 3)
    assert ci.lo <= ci.value <= ci.hi


def test_kelly_fraction_worked_example():
    # p = 0.6 at odds 2.0: (1.2 - 1) / 1 = 0.2
    np.testing.assert_allclose(ev.kelly_fraction(np.array([0.6]), np.array([2.0])), [0.2])
    np.testing.assert_allclose(ev.kelly_fraction(np.array([0.4]), np.array([2.0])), [0.0])


def test_select_bets_picks_max_ev_only_if_positive():
    odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [np.nan, 3.0, 4.0]])
    p = np.array([[0.6, 0.2, 0.2], [0.45, 0.30, 0.25], [0.9, 0.05, 0.05]])
    bet = ev.select_bets(p, odds)
    # row 0: EVs 0.2, -0.4, -0.2 -> home; row 1: -0.1, -0.1, 0.0 -> none (not > 0)
    assert bet.tolist() == [0, -1, -1]
    assert ev.select_bets(p, odds, min_edge=0.3).tolist() == [-1, -1, -1]


def test_stakes_and_settlement():
    odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    p = np.array([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]])
    bet = np.array([0, 0])
    flat = ev.stake_sizes(p, odds, bet, "flat")
    np.testing.assert_allclose(flat, [0.01, 0.01])
    kelly = ev.stake_sizes(p, odds, bet, "kelly")
    np.testing.assert_allclose(kelly, [min(0.25 * 0.2, 0.02)] * 2)  # 0.05 -> capped at 0.02
    pnl = ev.settle(bet, odds, flat, np.array([0, 1]))
    np.testing.assert_allclose(pnl, [0.01, -0.01])
    assert ev.settle(np.array([-1]), odds[:1], np.array([0.01]), np.array([0]))[0] == 0.0


def test_clv():
    np.testing.assert_allclose(ev.clv(np.array([2.2]), np.array([2.0])), [0.1])
    np.testing.assert_allclose(ev.clv(np.array([2.0]), np.array([2.2])), [2.0 / 2.2 - 1])


def test_bet_frame_and_scores():
    n = 6
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.tile([0.6, 0.2, 0.2], (n, 1))
    open_odds = np.tile([2.0, 3.5, 4.0], (n, 1))
    close_odds = np.tile([1.9, 3.6, 4.2], (n, 1))
    close_p = np.tile([0.5, 0.27, 0.23], (n, 1))
    led = ev.bet_frame(p, y, open_odds, close_odds, close_p)
    assert (led["bet"] == 0).all()
    np.testing.assert_allclose(led["clv"], 2.0 / 1.9 - 1)
    np.testing.assert_allclose(led["ev_at_close"], 0.5 * 2.0 - 1)
    assert led["won"].sum() == 2
    np.testing.assert_allclose(led["pnl_flat"].sum(), 2 * 0.01 - 4 * 0.01)
    tab = ev.score_bets(led, "x", seed=0)
    assert tab.set_index("metric").loc["n_bets", "value"] == 6
    np.testing.assert_allclose(tab.set_index("metric").loc["roi_flat", "value"], -1 / 3)
    probs = ev.score_probabilities(p, y, "x")
    assert set(probs["metric"]) == {"log_loss", "brier", "ece", "ece_h", "ece_d", "ece_a"}
    diff = ev.compare_probabilities(p, p, y, "x")
    assert (diff["value"] == 0).all()
