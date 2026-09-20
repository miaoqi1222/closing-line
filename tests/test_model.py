import numpy as np
import pandas as pd

from src.model import Elo, fit


def _matches(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    pairs = [rng.choice(teams, 2, replace=False) for _ in range(n)]
    return pd.DataFrame(
        {
            "home": [p[0] for p in pairs],
            "away": [p[1] for p in pairs],
            "result": rng.choice(["H", "D", "A"], n),
        }
    )


def test_probabilities_are_valid():
    p = Elo().predict(_matches())
    assert np.allclose(p.sum(axis=1), 1) and (p > 0).all()


def test_home_advantage_and_rating_direction():
    m = Elo()
    p = m.probs("A", "B")
    assert p[0] > p[2]  # equal ratings: home side favoured
    m.predict(pd.DataFrame({"home": ["A"], "away": ["B"], "result": ["H"]}))
    assert m.ratings["A"] > 1500 > m.ratings["B"]
    assert np.isclose(m.ratings["A"] + m.ratings["B"], 3000)  # zero-sum update


def _leaks(model_cls, cut: int = 150) -> bool:
    """True if predictions up to and including ``cut`` change when later results change."""
    a = _matches()
    b = a.copy()
    flip = {"H": "A", "A": "H", "D": "H"}
    b.loc[cut:, "result"] = b.loc[cut:, "result"].map(flip)  # alter match `cut` and all after
    pa, pb = model_cls().predict(a), model_cls().predict(b)
    assert not np.allclose(pa[cut + 1 :], pb[cut + 1 :])  # later predictions must react
    return not np.array_equal(pa[: cut + 1], pb[: cut + 1])


def test_no_lookahead():
    assert not _leaks(Elo)


def test_leakage_check_has_teeth():
    class LeakyElo(Elo):
        """Updates on the result *before* predicting: the bug the check must catch."""

        def predict(self, matches):
            out = np.empty((len(matches), 3))
            for i in range(len(matches)):
                super().predict(matches.iloc[[i]])
                out[i] = self.probs(matches["home"].iloc[i], matches["away"].iloc[i])
            return out

    assert _leaks(LeakyElo)


def test_fit_recovers_signal_and_is_deterministic():
    # Strong teams win: a fitted model must beat the untrained default on the same data.
    rng = np.random.default_rng(1)
    strength = {f"T{i}": i for i in range(10)}
    m = _matches(600, seed=2)
    edge = m["home"].map(strength) - m["away"].map(strength) + 1
    m["result"] = np.where(
        edge + rng.normal(0, 3, len(m)) > 1.5,
        "H",
        np.where(edge + rng.normal(0, 3, len(m)) < -1.5, "A", "D"),
    )
    a, b = fit(m), fit(m)
    assert (a.k, a.hfa, a.draw) == (b.k, b.hfa, b.draw)
    assert a.ratings["T9"] > a.ratings["T0"]
