"""Where artifacts go, and a markdown table writer (avoids the tabulate dependency)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    df = df.reset_index() if df.index.name is not None else df
    fmt = lambda v: format(v, floatfmt) if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(fmt, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, *body])


def write_md(name: str, lines: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text("\n".join(lines) + "\n")
