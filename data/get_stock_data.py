from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import yfinance as yf


def _ensure_utc_naive(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df = df.copy()
        df.index = idx.tz_convert("UTC").tz_localize(None)
    return df


@dataclass(slots=True)
class OHLCV:
    ticker: str
    interval: str
    df: pd.DataFrame
    msg: str = "OK"
    failed: bool = False

    def __len__(self) -> int:
        return int(len(self.df))

    def __bool__(self) -> bool:
        return not self.failed

    def __str__(self) -> str:
        start, end = self._range()
        return (
            f"{self.ticker} [{self.interval}] - {self.msg} | count={len(self)} | range={start}..{end} | na={self._na()}"
        )

    def __repr__(self) -> str:
        return f"OHLCV({str(self)})"

    def _range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if self.df.empty:
            return None, None
        idx = self.df.index
        return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())

    def _na(self) -> int:
        if self.df.empty or "Close" not in self.df.columns:
            return 0
        return int(self.df["Close"].isna().sum())


@dataclass(slots=True)
class OHLCVBatch:
    items: dict[str, OHLCV]

    def __len__(self) -> int:
        return int(len(self.items))

    def __iter__(self):
        return iter(self.items.values())

    def __getitem__(self, ticker: str) -> OHLCV:
        return self.items[ticker]

    def __str__(self) -> str:
        ok = sum(1 for v in self.items.values() if v)
        failed = len(self.items) - ok
        return f"OHLCVBatch(ok={ok}, failed={failed}, total={len(self.items)})"

    def __repr__(self) -> str:
        return str(self)

    def failed(self) -> dict[str, OHLCV]:
        return {t: v for t, v in self.items.items() if v.failed}

    def to_pandas(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for ticker, v in self.items.items():
            if v.df.empty:
                continue

            tmp = v.df.copy()
            idx_name = tmp.index.name or "index"
            tmp = tmp.reset_index().rename(columns={idx_name: "Datetime"})
            tmp.insert(0, "Ticker", ticker)

            tmp = tmp.sort_values("Datetime").reset_index(drop=True)
            frames.append(tmp)

        if not frames:
            return pd.DataFrame(columns=["Ticker", "Datetime"])

        return pd.concat(frames, axis=0, ignore_index=False)

    def drop(self, tickers: list[str]) -> "OHLCVBatch":
        drop_set = set(tickers)
        return OHLCVBatch(items={t: v for t, v in self.items.items() if t not in drop_set})

    def drop_illiquid(self, min_median_nok_volume: float = 1_000_000) -> "OHLCVBatch":
        df = self.to_pandas()
        if df.empty:
            return self

        nok_vol = df["Volume"] * df["Close"]
        med = nok_vol.groupby(df["Ticker"]).median()

        to_drop = med[med < min_median_nok_volume].index.tolist()
        return self.drop(to_drop)


def get_ohlcv(
    ticker: str,
    *,
    path: str | Path = "data/ohlcv",
    interval: str = "1h",
    update: bool = True,
) -> OHLCV:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / f"{ticker.split()[0]}_{interval}.csv"
    t = yf.Ticker(ticker)

    # No cache
    if not csv_path.exists():
        if not update:
            return OHLCV(ticker=ticker, interval=interval, df=pd.DataFrame(), msg="No cache (update=False)")
        df = _ensure_utc_naive(t.history(period="729d", interval=interval, raise_errors=True).sort_index())
        if not df.empty:
            df.to_csv(csv_path)
        return OHLCV(ticker=ticker, interval=interval, df=df, msg="OK")

    # Cache exists
    existing = _ensure_utc_naive(pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index())
    if not update:
        return OHLCV(ticker=ticker, interval=interval, df=existing, msg="OK")

    # Skip update if latest cached bar is today (UTC)
    last = existing.index.max()
    if pd.Timestamp(last).normalize() == pd.Timestamp.utcnow().normalize():
        return OHLCV(ticker=ticker, interval=interval, df=existing, msg="Already up to date")

    na = existing["Close"].isna()
    start = existing.index[na].min() if bool(na.any()) else existing.index.max()

    new = _ensure_utc_naive(t.history(start=start, interval=interval, raise_errors=True).sort_index())

    if new.empty:
        return OHLCV(ticker=ticker, interval=interval, df=existing, msg="No new data")

    merged = pd.concat([existing, new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.to_csv(csv_path)
    return OHLCV(ticker=ticker, interval=interval, df=cast(pd.DataFrame, merged), msg="OK")


def get_ohlcv_batch(
    tickers: list[str],
    *,
    path: str | Path = "data/ohlcv",
    interval: str = "1h",
    update: bool = True,
) -> OHLCVBatch:
    from tqdm import tqdm

    items: dict[str, OHLCV] = {}

    total = len(tickers)
    failed = 0

    with tqdm(total=total, desc="Fetching", unit="ticker", leave=True) as pbar:
        for ticker in tickers:
            try:
                items[ticker] = get_ohlcv(ticker, path=path, interval=interval, update=update)
            except Exception as e:
                items[ticker] = OHLCV(
                    ticker=ticker,
                    interval=interval,
                    df=pd.DataFrame(),
                    msg=str(e) or type(e).__name__,
                    failed=True,
                )
                failed += 1

            pbar.set_description_str(f"Fetching OHLCV [{ticker}])")
            pbar.set_postfix_str(f"FAILED={failed}", refresh=True)
            pbar.update(1)

    return OHLCVBatch(items=items)
