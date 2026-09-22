import numpy as np
import pandas as pd
import pytest

from src import data


@pytest.mark.parametrize(
    "col,expected",
    [
        ("PSH", ("PS", "open", "h")),
        ("PSCD", ("PS", "close", "d")),
        ("B365CA", ("B365", "close", "a")),
        ("1XBCH", ("1XB", "close", "h")),
        ("BFECH", ("BFE", "close", "h")),
        ("BFD", ("BF", "open", "d")),  # Betfair draw ...
        ("BFDH", ("BFD", "open", "h")),  # ... vs Betfred home
        ("PH", ("PS", "open", "h")),  # old Pinnacle alias
        ("PPD", ("PP", "open", "d")),
        ("AvgCH", ("Avg", "close", "h")),
        ("FTHG", None),
        ("B365AH", None),  # Asian handicap size
        ("PAHH", None),
        ("B365>2.5", None),
        ("BbAvH", None),
        ("AHh", None),
        ("Date", None),
    ],
)
def test_parse_odds_column(col, expected):
    assert data.parse_odds_column(col) == expected


def test_season_code_and_label():
    assert data.season_code(2016) == "1617"
    assert data.season_label(2019) == "2019-20"


def _raw(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Div": ["E0"] * n, "Date": dates, "HomeTeam": ["A"] * n, "AwayTeam": ["B"] * n,
            "FTHG": [1] * n, "FTAG": [0] * n, "FTR": ["H"] * n,
            "PSH": [2.0] * n, "PSD": [3.5] * n, "PSA": [4.0] * n,
            "PSCH": [1.9] * n, "PSCD": [3.6] * n, "PSCA": [4.2] * n, "B365AH": [-0.5] * n,
        }
    )  # fmt: skip


def test_tidy_season_both_date_formats():
    out = data.tidy_season(_raw(["13/08/16", "14/08/2016"]), 2016)
    assert list(out["date"]) == [pd.Timestamp("2016-08-13"), pd.Timestamp("2016-08-14")]
    assert out["season"].iloc[0] == "2016-17"
    assert out["time"].isna().all()
    assert data.books(out) == ["ps"]
    assert out["ps_close_h"].iloc[0] == 1.9 and "b365_open_h" not in out


def test_tidy_season_rejects_bad_result():
    raw = _raw(["13/08/16"])
    raw.loc[0, "FTR"] = "X"
    with pytest.raises(ValueError):
        data.tidy_season(raw, 2016)


def test_tidy_season_odds_below_one_become_nan():
    raw = _raw(["13/08/16"])
    raw.loc[0, "PSH"] = 0.5
    assert np.isnan(data.tidy_season(raw, 2016)["ps_open_h"].iloc[0])


def test_load_matches_cached(tmp_path):
    r1 = _raw(["13/08/16", "14/08/16"])
    r2 = _raw(["12/08/2017"]).assign(B365CH=1.8, B365CD=3.7, B365CA=4.5)
    r1.to_csv(tmp_path / "E0_1617.csv", index=False)
    r2.to_csv(tmp_path / "E0_1718.csv", index=False)
    df = data.load_matches([2016, 2017], raw_dir=tmp_path)
    assert len(df) == 3 and df["date"].is_monotonic_increasing
    assert df.columns[: len(data.META_COLUMNS)].tolist() == data.META_COLUMNS
    assert df["b365_close_h"].isna().sum() == 2
    assert data.books(df) == ["b365", "ps"]
    cov = data.coverage(df).set_index(["season", "book"])
    assert cov.loc[("2016-17", "b365"), "close_complete"] == 0
    assert cov.loc[("2017-18", "b365"), "close_complete"] == 1


@pytest.mark.skipif(
    not all(data.raw_path(y).exists() for y in data.DEFAULT_START_YEARS), reason="raw cache not present"
)
def test_real_cache_shape():
    df = data.load_matches()
    assert df.groupby("season").size().eq(380).all() and len(df) == 3800
    assert df["result"].isin(["H", "D", "A"]).all()
    ps = df[["ps_close_h", "ps_close_d", "ps_close_a"]].notna().all(axis=1)
    avg = df[["avg_close_h", "avg_close_d", "avg_close_a"]].notna().all(axis=1)
    assert (ps | avg).all()  # every match has a closing price from somewhere
