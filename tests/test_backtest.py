import numpy as np
import pandas as pd

from src import backtest, model


def _matches(n_seasons: int = 5, per: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    for s in range(n_seasons):
        for j in range(per):
            h, a = rng.choice(teams, 2, replace=False)
            rows.append({"season": f"S{s}", "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=365 * s + j),
                         "home": h, "away": a, "result": rng.choice(["H", "D", "A"]),
                         "ps_open_h": 2.5, "ps_open_d": 3.3, "ps_open_a": 3.0,
                         "ps_close_h": 2.4, "ps_close_d": 3.4, "ps_close_a": 3.1})  # fmt: skip
    return pd.DataFrame(rows)


def test_walk_forward_is_out_of_sample():
    df = _matches()
    P, prior = backtest.walk_forward(df, model.fit, min_train_seasons=2)
    assert np.isnan(P[df["season"].isin(["S0", "S1"])]).all()
    assert np.isfinite(P[~df["season"].isin(["S0", "S1"])]).all()
    # Changing every result in S4 must not change predictions or priors for S2 and S3.
    other = df.copy()
    other.loc[other["season"] == "S4", "result"] = "A"
    P2, prior2 = backtest.walk_forward(other, model.fit, min_train_seasons=2)
    early = df["season"].isin(["S2", "S3"]).to_numpy()
    assert np.array_equal(P[early], P2[early]) and np.array_equal(prior[early], prior2[early])
    assert not np.array_equal(P[~early & np.isfinite(P[:, 0])], P2[~early & np.isfinite(P2[:, 0])])


def test_run_frame():
    out = backtest.run(_matches(), model.fit)
    assert set(out["season"]) == {"S3", "S4"}
    assert np.allclose(out[backtest.MODEL].sum(axis=1), 1)
    assert out["bet"].isin(["", "H", "D", "A"]).all()
    assert (out.loc[out["bet"] == "", "stake_flat"] == 0).all()
    assert out["mkt_source"].eq("ps").all()
