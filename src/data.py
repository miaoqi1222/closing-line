"""Download, cache and normalise football-data.co.uk season files.

Source: https://www.football-data.co.uk/ (schema in notes.txt on that site).
Each season is a single CSV with one row per match, results, and 1X2 odds from
several bookmakers. Two odds snapshots exist per book:

* the plain columns (e.g. ``PSH``) are the *pre-closing* price. Per notes.txt these
  are collected on Friday afternoon (weekend fixtures) or Tuesday afternoon
  (midweek fixtures). They are NOT true opening lines; we call them ``open`` here
  for brevity but the README must disclose what they actually are.
* columns with a ``C`` inserted before the outcome letter (e.g. ``PSCH``) are the
  *closing* price, available from 2019/20 for most books and from much earlier
  for Pinnacle (``PSCH``).

Column naming in the tidy frame: ``{book}_{open|close}_{h|d|a}`` with the book
code lower-cased, e.g. ``ps_close_h``, ``b365_open_d``, ``avg_close_a``.
"""

from __future__ import annotations

import io
import ssl
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_LEAGUE = "E0"
# Ten most recent complete EPL seasons as of September 2026.
DEFAULT_START_YEARS: tuple[int, ...] = tuple(range(2016, 2026))

# Bookmaker column prefixes from notes.txt. Aliases map old prefixes onto the current one.
BOOK_CODES: dict[str, str] = {
    "1XB": "1XBet",
    "B365": "Bet365",
    "BFE": "Betfair Exchange",
    "BFD": "Betfred",
    "BF": "Betfair",
    "BMGM": "BetMGM",
    "BV": "BetVictor",
    "BS": "Blue Square",
    "BW": "Bet&Win",
    "CL": "Coral",
    "GB": "Gamebookers",
    "IW": "Interwetten",
    "LB": "Ladbrokes",
    "PP": "Paddy Power",
    "PS": "Pinnacle",
    "SK": "Skybet",
    "SO": "Sporting Odds",
    "SB": "Sportingbet",
    "SJ": "Stan James",
    "SY": "Stanleybet",
    "VC": "VC Bet",
    "WH": "William Hill",
    "Max": "Market maximum",
    "Avg": "Market average",
}
BOOK_ALIASES: dict[str, str] = {"P": "PS"}
_OUTCOMES = {"H": "h", "D": "d", "A": "a"}
META_COLUMNS = ["season", "date", "time", "home", "away", "home_goals", "away_goals", "result"]


def season_code(start_year: int) -> str:
    """2016 -> '1617' (the football-data.co.uk URL fragment)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2016 -> '2016-17'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def raw_path(start_year: int, league: str = DEFAULT_LEAGUE, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{league}_{season_code(start_year)}.csv"


def _ssl_context() -> ssl.SSLContext:
    """Default context, falling back to the OS bundle when Python ships no CA certs."""
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats()["x509_ca"] == 0:
        for cafile in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
            if Path(cafile).exists():
                ctx = ssl.create_default_context(cafile=cafile)
                break
    return ctx


def download_season(
    start_year: int,
    league: str = DEFAULT_LEAGUE,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
) -> Path:
    """Fetch one season CSV into ``raw_dir`` unless it is already cached."""
    path = raw_path(start_year, league, raw_dir)
    if path.exists() and not force:
        return path
    url = BASE_URL.format(season=season_code(start_year), league=league)
    req = urllib.request.Request(url, headers={"User-Agent": "closing-line/0.0.1"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        body = resp.read()
    if not body.lstrip(b"\xef\xbb\xbf").startswith(b"Div,"):
        raise ValueError(f"unexpected payload from {url}: {body[:80]!r}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def parse_odds_column(col: str) -> tuple[str, str, str] | None:
    """Map a source column name to ``(book_code, stage, outcome)`` or ``None``.

    ``PSH`` -> ('PS', 'open', 'h'); ``B365CA`` -> ('B365', 'close', 'a').
    Anything that is not a 1X2 price (Asian handicap, over/under, Betbrain
    aggregates, match stats) returns ``None``.
    """
    codes = sorted(list(BOOK_CODES) + list(BOOK_ALIASES), key=len, reverse=True)
    for code in codes:
        if not col.startswith(code):
            continue
        rest = col[len(code) :]
        book = BOOK_ALIASES.get(code, code)
        if rest in _OUTCOMES:
            return book, "open", _OUTCOMES[rest]
        if len(rest) == 2 and rest[0] == "C" and rest[1] in _OUTCOMES:
            return book, "close", _OUTCOMES[rest[1]]
        # e.g. "BFD" matched Betfred ("BFD") with nothing left: fall through to Betfair draw.
    return None


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Source uses dd/mm/yy in some seasons and dd/mm/yyyy in others."""
    s = raw.astype(str).str.strip()
    four = s.str.len() == 10
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[four] = pd.to_datetime(s[four], format="%d/%m/%Y")
    out[~four] = pd.to_datetime(s[~four], format="%d/%m/%y")
    return out


def read_raw_season(path: Path) -> pd.DataFrame:
    """Read a cached season CSV as-is, dropping fully blank rows."""
    body = path.read_bytes().lstrip(b"\xef\xbb\xbf")  # some seasons carry a UTF-8 BOM
    raw = pd.read_csv(io.BytesIO(body), encoding="latin-1")
    raw = raw.dropna(how="all")
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    return raw.reset_index(drop=True)


def tidy_season(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Normalise one raw season frame to the tidy schema."""
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"season {start_year}: missing columns {sorted(missing)}")
    out = pd.DataFrame(
        {
            "season": season_label(start_year),
            "date": _parse_dates(raw["Date"]),
            "time": raw["Time"].astype("string")
            if "Time" in raw
            else pd.Series(pd.NA, index=raw.index, dtype="string"),
            "home": raw["HomeTeam"].astype("string").str.strip(),
            "away": raw["AwayTeam"].astype("string").str.strip(),
            "home_goals": raw["FTHG"].astype(int),
            "away_goals": raw["FTAG"].astype(int),
            "result": raw["FTR"].astype("string").str.strip(),
        }
    )
    for col in raw.columns:
        parsed = parse_odds_column(col)
        if parsed is None:
            continue
        book, stage, outcome = parsed
        vals = pd.to_numeric(raw[col], errors="coerce").astype(float)
        vals = vals.where(vals >= 1.0, np.nan)  # decimal odds below 1 are data errors
        out[f"{book.lower()}_{stage}_{outcome}"] = vals
    bad = ~out["result"].isin(["H", "D", "A"])
    if bad.any():
        raise ValueError(f"season {start_year}: {int(bad.sum())} rows with bad FTR")
    return out


def load_matches(
    start_years: Iterable[int] = DEFAULT_START_YEARS,
    league: str = DEFAULT_LEAGUE,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Return one tidy frame of matches across seasons, sorted by date/home/away.

    Downloads any season not already cached in ``raw_dir``. Odds columns absent
    from a season are NaN for that season's rows.
    """
    frames = []
    for year in start_years:
        path = download_season(year, league, raw_dir)
        frames.append(tidy_season(read_raw_season(path), year))
    df = pd.concat(frames, ignore_index=True, sort=False)
    odds_cols = sorted(c for c in df.columns if c not in META_COLUMNS)
    df = df[META_COLUMNS + odds_cols]
    df = df.sort_values(["date", "home", "away"], kind="mergesort").reset_index(drop=True)
    return df


def odds_columns(df: pd.DataFrame, stage: str | None = None) -> list[str]:
    """Odds column names in ``df``, optionally filtered to 'open' or 'close'."""
    cols = [c for c in df.columns if c not in META_COLUMNS]
    if stage is not None:
        cols = [c for c in cols if c.split("_")[1] == stage]
    return cols


def books(df: pd.DataFrame) -> list[str]:
    """Lower-cased book codes present in ``df``."""
    return sorted({c.split("_")[0] for c in odds_columns(df)})


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per season and book: number of matches with a complete H/D/A triple, open and close."""
    rows = []
    for season, grp in df.groupby("season", sort=True):
        for book in books(df):
            rec: dict[str, object] = {"season": season, "book": book, "matches": len(grp)}
            for stage in ("open", "close"):
                cols = [f"{book}_{stage}_{o}" for o in "hda"]
                present = [c for c in cols if c in grp]
                rec[f"{stage}_complete"] = (
                    int(grp[present].notna().all(axis=1).sum()) if len(present) == 3 else 0
                )
            rows.append(rec)
    return pd.DataFrame(rows)
