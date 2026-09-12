import numpy as np
import pandas as pd
import pytest

from src import data


@pytest.mark.parametrize(
    "col,expected",
    [
        ("PSH", ("PS", "open", "h")),
        ("PSCD", ("PS", "close", "d")),
        ("B365A", ("B365", "open", "a")),
        ("B365CA", ("B365", "close", "a")),
        ("1XBCH", ("1XB", "close", "h")),
        ("BFEH", ("BFE", "open", "h")),
        ("BFECH", ("BFE", "close", "h")),
        ("BFD", ("BF", "open", "d")),  # Betfair draw ...
        ("BFDH", ("BFD", "open", "h")),  # ... vs Betfred home
        ("BFDCA", ("BFD", "close", "a")),
        ("PH", ("PS", "open", "h")),  # old Pinnacle alias
        ("PPD", ("PP", "open", "d")),
        ("AvgCH", ("Avg", "close", "h")),
        ("MaxA", ("Max", "open", "a")),
        ("FTHG", None),
        ("HTHG", None),
        ("B365AH", None),  # Asian handicap size
        ("B365AHH", None),
        ("PAHH", None),
        ("B365>2.5", None),
        ("BbAvH", None),
        ("AHh", None),
        ("HomeTeam", None),
        ("Date", None),
    ],
)
def test_parse_odds_column(col, expected):
    assert data.parse_odds_column(col) == expected


def test_season_code_and_label():
    assert data.season_code(2016) == "1617"
    assert data.season_code(2025) == "2526"
    assert data.season_label(2019) == "2019-20"


def _raw(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Div": ["E0"] * n,
            "Date": dates,
            "HomeTeam": ["A"] * n,
            "AwayTeam": ["B"] * n,
            "FTHG": [1] * n,
            "FTAG": [0] * n,
            "FTR": ["H"] * n,
            "PSH": [2.0] * n,
            "PSD": [3.5] * n,
            "PSA": [4.0] * n,
            "PSCH": [1.9] * n,
            "PSCD": [3.6] * n,
            "PSCA": [4.2] * n,
            "B365AH": [-0.5] * n,
        }
    )


def test_tidy_season_both_date_formats():
    out = data.tidy_season(_raw(["13/08/16", "14/08/2016"]), 2016)
    assert list(out["date"]) == [pd.Timestamp("2016-08-13"), pd.Timestamp("2016-08-14")]
    assert out["season"].iloc[0] == "2016-17"
    assert out["time"].isna().all()
    assert set(data.odds_columns(out)) == {
        "ps_open_h",
        "ps_open_d",
        "ps_open_a",
        "ps_close_h",
        "ps_close_d",
        "ps_close_a",
    }
    assert out["ps_close_h"].iloc[0] == 1.9


def test_tidy_season_rejects_bad_result():
    raw = _raw(["13/08/16"])
    raw.loc[0, "FTR"] = "X"
    with pytest.raises(ValueError):
        data.tidy_season(raw, 2016)


def test_tidy_season_odds_below_one_become_nan():
    raw = _raw(["13/08/16"])
    raw.loc[0, "PSH"] = 0.5
    out = data.tidy_season(raw, 2016)
    assert np.isnan(out["ps_open_h"].iloc[0])


def test_load_matches_cached(tmp_path):
    # Build two fake cached seasons and check concat / sort / column union behaviour.
    r1 = _raw(["13/08/16", "14/08/16"])
    r2 = _raw(["12/08/2017"])
    r2["B365CH"] = 1.8
    r2["B365CD"] = 3.7
    r2["B365CA"] = 4.5
    r1.to_csv(tmp_path / "E0_1617.csv", index=False)
    r2.to_csv(tmp_path / "E0_1718.csv", index=False)
    df = data.load_matches([2016, 2017], raw_dir=tmp_path)
    assert len(df) == 3
    assert df["date"].is_monotonic_increasing
    assert df.columns[: len(data.META_COLUMNS)].tolist() == data.META_COLUMNS
    assert df["b365_close_h"].isna().sum() == 2
    assert data.books(df) == ["b365", "ps"]
    cov = data.coverage(df)
    row = cov[(cov.season == "2016-17") & (cov.book == "b365")].iloc[0]
    assert row["close_complete"] == 0 and row["matches"] == 2


@pytest.mark.skipif(
    not all(data.raw_path(y).exists() for y in data.DEFAULT_START_YEARS),
    reason="raw cache not present",
)
def test_real_cache_shape():
    df = data.load_matches()
    assert len(df) == 380 * len(data.DEFAULT_START_YEARS)
    assert df["result"].isin(["H", "D", "A"]).all()
    assert df.groupby("season").size().eq(380).all()
    # every match has at least one complete closing triple
    close = df[[c for c in data.odds_columns(df, "close") if c.startswith("avg_")]]
    ps = df[["ps_close_h", "ps_close_d", "ps_close_a"]]
    assert (close.notna().all(axis=1) | ps.notna().all(axis=1)).all()
