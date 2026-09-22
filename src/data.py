"""Download, cache and normalise football-data.co.uk season files.

Source: https://www.football-data.co.uk/ (schema in notes.txt on that site).
One CSV per season, one row per match, with 1X2 odds from several bookmakers:

* plain columns (``PSH``) are the *pre-closing* price, collected Friday afternoon
  for weekend fixtures and Tuesday afternoon for midweek ones. They are not true
  opening lines; we call them ``open`` for brevity and disclose this in the README.
* columns with a ``C`` before the outcome letter (``PSCH``) are the *closing* price.

Tidy frame columns: ``{book}_{open|close}_{h|d|a}``, e.g. ``ps_close_h``.
"""

from __future__ import annotations

import ssl
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.report import ROOT

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_LEAGUE = "E0"
DEFAULT_START_YEARS = tuple(range(2016, 2026))  # ten complete EPL seasons as of Sept 2026

# Bookmaker column prefixes from notes.txt, longest first so "B365" wins over "B".
BOOK_CODES = sorted(
    ["1XB", "B365", "BFE", "BFD", "BF", "BMGM", "BV", "BS", "BW", "CL", "GB", "IW", "LB", "PP",
     "PS", "P", "SK", "SO", "SB", "SJ", "SY", "VC", "WH", "Max", "Avg"],
    key=len, reverse=True,
)  # fmt: skip
BOOK_ALIASES = {"P": "PS"}  # old Pinnacle prefix
META_COLUMNS = ["season", "date", "time", "home", "away", "home_goals", "away_goals", "result"]


def season_code(start_year: int) -> str:
    """2016 -> '1617' (the URL fragment)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2016 -> '2016-17'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def raw_path(start_year: int, league: str = DEFAULT_LEAGUE, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{league}_{season_code(start_year)}.csv"


def _ssl_context() -> ssl.SSLContext:
    """Default context, falling back to the OS bundle when Python ships no CA certs."""
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats()["x509_ca"] == 0 and Path("/etc/ssl/cert.pem").exists():
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ctx


def download_season(start_year: int, league: str = DEFAULT_LEAGUE, raw_dir: Path = RAW_DIR) -> Path:
    """Fetch one season CSV into ``raw_dir`` unless it is already cached."""
    path = raw_path(start_year, league, raw_dir)
    if path.exists():
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
    """'PSH' -> ('PS', 'open', 'h'); 'B365CA' -> ('B365', 'close', 'a'); else None."""
    for code in BOOK_CODES:
        if col.startswith(code):
            rest = col[len(code) :]
            book = BOOK_ALIASES.get(code, code)
            if rest in "HDA" and rest:
                return book, "open", rest.lower()
            if len(rest) == 2 and rest[0] == "C" and rest[1] in "HDA":
                return book, "close", rest[1].lower()
            # e.g. "BFD" matched Betfred with nothing left: keep looking (Betfair draw).
    return None


def read_raw_season(path: Path) -> pd.DataFrame:
    """Read a cached season CSV, dropping blank rows and unnamed trailing columns."""
    raw = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace").dropna(how="all")
    return raw.loc[:, ~raw.columns.str.startswith("Unnamed")].reset_index(drop=True)


def tidy_season(raw: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Normalise one raw season frame to the tidy schema."""
    date = raw["Date"].astype(str).str.strip()
    date = date.where(date.str.len() == 10, date.str[:6] + "20" + date.str[6:])  # dd/mm/yy
    out = pd.DataFrame(
        {
            "season": season_label(start_year),
            "date": pd.to_datetime(date, format="%d/%m/%Y"),
            "time": raw["Time"] if "Time" in raw else pd.NA,
            "home": raw["HomeTeam"].str.strip(),
            "away": raw["AwayTeam"].str.strip(),
            "home_goals": raw["FTHG"].astype(int),
            "away_goals": raw["FTAG"].astype(int),
            "result": raw["FTR"].str.strip(),
        }
    )
    for col in raw.columns:
        if parsed := parse_odds_column(col):
            book, stage, outcome = parsed
            vals = pd.to_numeric(raw[col], errors="coerce")
            out[f"{book.lower()}_{stage}_{outcome}"] = vals.where(vals >= 1.0, np.nan)
    if not out["result"].isin(["H", "D", "A"]).all():
        raise ValueError(f"season {start_year}: bad FTR values")
    return out


def load_matches(
    start_years: Iterable[int] = DEFAULT_START_YEARS,
    league: str = DEFAULT_LEAGUE,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """One tidy frame across seasons, sorted by date/home/away. Downloads what is not cached."""
    frames = [tidy_season(read_raw_season(download_season(y, league, raw_dir)), y) for y in start_years]
    df = pd.concat(frames, ignore_index=True)
    df = df[META_COLUMNS + sorted(c for c in df.columns if c not in META_COLUMNS)]
    return df.sort_values(["date", "home", "away"], kind="mergesort").reset_index(drop=True)


def books(df: pd.DataFrame) -> list[str]:
    """Lower-cased book codes present in ``df``."""
    return sorted({c.split("_")[0] for c in df.columns if c not in META_COLUMNS})


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per season and book: matches with a complete H/D/A triple, open and close."""
    rows = []
    for season, g in df.groupby("season"):
        for book in books(df):
            rec = {"season": season, "book": book, "matches": len(g)}
            for stage in ("open", "close"):
                cols = [f"{book}_{stage}_{o}" for o in "hda"]
                have = set(cols) <= set(g.columns)
                rec[f"{stage}_complete"] = int(g[cols].notna().all(axis=1).sum()) if have else 0
            rows.append(rec)
    return pd.DataFrame(rows)
