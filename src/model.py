"""Baseline model: Elo with home advantage and an ordered-logit draw band.

Interface shared by all models: ``fit(train) -> model`` and
``model.predict(matches) -> (n, 3)`` probabilities, home / draw / away.

``predict`` walks forward in time. For each match it first predicts from the
current ratings, then updates the ratings with that match's result. A match
therefore never sees its own result or any later one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}
START = 1500.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Elo:
    k: float = 20.0  # update speed
    hfa: float = 60.0  # home advantage, in rating points
    draw: float = 0.6  # half-width of the draw band on the logit scale
    ratings: dict[str, float] = field(default_factory=dict)

    def probs(self, home: str, away: str) -> np.ndarray:
        diff = self.ratings.get(home, START) + self.hfa - self.ratings.get(away, START)
        s = diff * np.log(10) / 400
        p_home, p_away = _sigmoid(s - self.draw), _sigmoid(-s - self.draw)
        return np.array([p_home, 1 - p_home - p_away, p_away])

    def predict(self, matches: pd.DataFrame) -> np.ndarray:
        """Predict each match in order, then update on its result (if it has one)."""
        out = np.empty((len(matches), 3))
        rows = zip(matches["home"], matches["away"], matches["result"], strict=True)
        for i, (home, away, result) in enumerate(rows):
            p = out[i] = self.probs(home, away)
            if result in SCORE:
                delta = self.k * (SCORE[result] - (p[0] + 0.5 * p[1]))
                self.ratings[home] = self.ratings.get(home, START) + delta
                self.ratings[away] = self.ratings.get(away, START) - delta
        return out


def fit(train: pd.DataFrame) -> Elo:
    """Choose k, hfa and draw by minimising log loss on ``train`` only."""
    y = train["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

    def loss(theta: np.ndarray) -> float:
        p = Elo(abs(theta[0]), theta[1], abs(theta[2])).predict(train)
        return -np.log(p[np.arange(len(y)), y]).mean()

    best = minimize(loss, x0=[20.0, 60.0, 0.6], method="Nelder-Mead").x
    model = Elo(abs(best[0]), best[1], abs(best[2]))
    model.predict(train)  # leave the ratings as they stand at the end of training
    return model
