from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

if TYPE_CHECKING:
    from data.get_stock_data import OHLCV, OHLCVBatch


class ER(NamedTuple):
    """Expected return, standard error, and whiteness p-value."""

    mean: float
    se: float
    p_white: float


@dataclass(slots=True)
class LLTrendModel:
    ohlcv: OHLCV
    model: UnobservedComponents
    res: Any
    idx: pd.Index

    @classmethod
    def fit(cls, ohlcv: OHLCV) -> "LLTrendModel":
        if ohlcv.failed or ohlcv.df.empty:
            raise ValueError(f"Cannot fit model: {ohlcv.ticker} - {ohlcv.msg}")

        log_close = np.log(ohlcv.df["Close"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = UnobservedComponents(log_close, level="lltrend")
            res = mod.fit(disp=False, warn_convergence=False)

        return cls(ohlcv=ohlcv, model=mod, res=res, idx=log_close.index)

    def _idx(self) -> pd.Index:
        return self.idx

    def _state_var(self, i: int) -> pd.Series:
        idx = self._idx()
        v = np.sqrt(self.res.filtered_state_cov[i, i, :])
        return pd.Series(v, index=idx, name="se")

    def _filtered(self, name: str) -> pd.Series:
        idx = self._idx()

        if name == "level":
            s = self.res.level["filtered"]
        elif name == "trend":
            s = self.res.trend["filtered"]
        else:
            raise ValueError(f"Unknown state name: {name!r}")

        return pd.Series(np.asarray(s), index=idx)

    def unlogged(self, z: float = 2.0) -> pd.DataFrame:
        lvl = self._filtered("level")
        tr = self._filtered("trend")
        lvl_se = self._state_var(0)
        tr_se = self._state_var(1)

        return pd.DataFrame(
            {
                "level": lvl.pipe(np.exp),
                "level_se": lvl_se,
                "level_lo": (lvl - z * lvl_se).pipe(np.exp),
                "level_hi": (lvl + z * lvl_se).pipe(np.exp),
                "trend": tr.pipe(np.exp),
                "trend_se": tr_se,
                "trend_lo": (tr - z * tr_se).pipe(np.exp),
                "trend_hi": (tr + z * tr_se).pipe(np.exp),
            },
            index=lvl.index,
        )

    def _p_white(self, lags: int = 24) -> float:
        """Test whether residuals are white"""
        lb = self.res.test_serial_correlation(method="ljungbox", lags=lags)
        pvals = np.asarray(lb)[0, 1, :]
        return float(np.nanmin(pvals))

    def er(self, lags: int = 24) -> ER:
        tr = self._filtered("trend")
        se_tr = self._state_var(1)

        m = float(tr.iloc[-1])
        s = float(se_tr.iloc[-1])
        v = s * s

        er_mean = float(np.exp(m + 0.5 * v) - 1.0)
        er_var = float((np.exp(v) - 1.0) * np.exp(2.0 * m + v))
        er_se = float(np.sqrt(er_var))

        p_white = self._p_white(lags=lags)
        return ER(mean=er_mean, se=er_se, p_white=p_white)


@dataclass(slots=True)
class LLTrendBatch:
    batch: OHLCVBatch
    items: dict[str, LLTrendModel]
    errors: dict[str, str]

    @classmethod
    def fit(cls, batch: OHLCVBatch) -> "LLTrendBatch":
        items: dict[str, LLTrendModel] = {}
        errors: dict[str, str] = {}

        total = len(batch.items)
        ok = 0
        failed = 0

        for ticker, ohlcv in batch.items.items():
            try:
                items[ticker] = LLTrendModel.fit(ohlcv)
                ok += 1
            except Exception as e:
                errors[ticker] = str(e) or type(e).__name__
                failed += 1

            msg = f"Fitted {ok}/{total} (failed {failed})"
            print("\r" + msg.ljust(40), end="", flush=True)

        print()
        return cls(batch=batch, items=items, errors=errors)

    def er(self, lags: int = 24) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for ticker, m in self.items.items():
            er = m.er(lags=lags)
            rows.append({"Ticker": ticker, "er": er.mean, "er_se": er.se, "p_white": er.p_white})
        return pd.DataFrame(rows).set_index("Ticker").sort_index()
