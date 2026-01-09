from __future__ import annotations

from pathlib import Path
import pandas as pd

DOWNLOAD_URL = "https://live.euronext.com/nb/markets/oslo/equities/list"

COL_NAME = "Name"
COL_SYMBOL = "Symbol"
COL_MARKET = "Market"

REQUIRED_COLS = (COL_SYMBOL, COL_NAME, COL_MARKET)


def get_tickers(path: str | Path = "data/tickers.csv") -> pd.DataFrame:
    """
    Return Oslo Børs equity tickers from a locally saved Euronext export file.

    The file is a semicolon-separated CSV with a header row, followed by three metadata rows.
    Required columns are defined by REQUIRED_COLS. The returned DataFrame contains OUT_COLS,
    and Symbol values have '.OL' appended.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n\n"
            f"Download it from:\n  {DOWNLOAD_URL}\n\n"
            f"Then save it as:\n  {path}\n"
        )

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=[1, 2, 3],
        encoding="utf-8-sig",
        dtype=str,
    )

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df[COL_NAME] = df[COL_NAME].str.strip()
    df[COL_SYMBOL] = df[COL_SYMBOL].str.strip() + ".OL"
    df[COL_MARKET] = df[COL_MARKET].str.strip()

    return df.loc[:, REQUIRED_COLS].reset_index(drop=True)


if __name__ == "__main__":
    tickers = get_tickers()
    print(tickers.head())
    print(f"Fetched {len(tickers)} tickers")
