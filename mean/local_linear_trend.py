from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from mean.base import MeanModel, PredictValues


if TYPE_CHECKING:
    from data.get_stock_data import OHLCV


@dataclass(slots=True)
class LLTModel(MeanModel):
    model: UnobservedComponents
    ohlcv: OHLCV
    res: Any
    idx: pd.Index

    @classmethod
    def fit(cls, ohlcv: OHLCV) -> "LLTModel":
        if ohlcv.failed or ohlcv.df.empty:
            raise ValueError(f"Cannot fit model: {ohlcv.ticker} - {ohlcv.msg}")

        log_close = np.log(ohlcv.df["Close"])
        mod = UnobservedComponents(log_close, level="lltrend")
        res = mod.fit(disp=False)

        return cls(ohlcv=ohlcv, model=mod, res=res, idx=log_close.index)

    def history(self, z: float = 2.0) -> pd.DataFrame:
        level = self.res.level["filtered"]
        trend = self.res.trend["filtered"]
        level_se = np.sqrt(self.res.filtered_state_cov[0, 0, :])
        trend_se = np.sqrt(self.res.filtered_state_cov[1, 1, :])

        return pd.DataFrame(
            {
                "level": np.exp(level),
                "level_se": level_se,
                "level_lo": np.exp(level - z * level_se),
                "level_hi": np.exp(level + z * level_se),
                "trend": np.expm1(trend),
                "trend_se": trend_se * np.exp(trend),
                "trend_lo": np.expm1(trend - z * trend_se),
                "trend_hi": np.expm1(trend + z * trend_se),
            },
            index=self.idx,
        )

    def predict(self) -> PredictValues:
        return {
            "ticker": self.ohlcv.ticker,
            "ret": float(self.res.trend["filtered"][-1]),
            "ret_se": float(np.sqrt(self.res.filtered_state_cov[1, 1, :])[-1]),
            "p_white": round(float(np.min(np.asarray(self.res.test_serial_correlation(method=None))[0, 1, :])), 2),
            "p_norm": round(float(np.asarray(self.res.test_normality(method=None))[0][1]), 2),
            "p_heterod": round(float(np.asarray(self.res.test_heteroskedasticity(method=None))[0][1]), 2),
        }

    def plot(
        self,
        skip: int = 0,
        z: float = 2.0,
        figsize: Tuple[float, float] = (15, 15),
    ):
        import matplotlib.pyplot as plt

        hist = self.history(z=z).iloc[skip:]
        ticker = self.ohlcv.ticker
        close = self.ohlcv.df["Close"].reindex(self.idx).iloc[skip:]
        vol = self.ohlcv.df["Volume"].reindex(self.idx).iloc[skip:].fillna(0)
        log_return = np.log(close).diff().iloc[skip:]

        innov = self.res.filter_results.standardized_forecasts_error
        innov = innov[0] if np.ndim(innov) == 2 else innov
        innov = pd.Series(innov, index=pd.Index(self.idx)).iloc[skip:]
        innov = innov.reindex(hist.index)

        fig, (ax_close, ax_trend, ax_vol, ax_innov) = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={"height_ratios": (2, 2, 1, 1)},
        )

        er = (1 + hist["trend"].iloc[-1]) ** (8 * 250) - 1
        lo = (1 + hist["trend_lo"].iloc[-1]) ** (8 * 250) - 1
        hi = (1 + hist["trend_hi"].iloc[-1]) ** (8 * 250) - 1

        fig.suptitle(f"Local Linear Trend — {ticker} — ER (annualized): {er:.2%} [{lo:.2%}, {hi:.2%}] — z={z:.2f}")

        ax_close.fill_between(hist.index, hist["level_lo"], hist["level_hi"], alpha=0.2, zorder=1)
        ax_close.plot(hist.index, hist["level"], zorder=3, label="level")
        ax_close.plot(close.index, close.to_numpy(), zorder=2, linewidth=1.5, label="close")
        ax_close.set_title("Close Price (and level)")

        ax_trend.fill_between(hist.index, hist["trend_lo"], hist["trend_hi"], alpha=0.2)
        ax_trend.plot(hist.index, hist["trend"], label="trend")
        ax_trend.axhline(0, linestyle=":", linewidth=1)

        ax_ret = ax_trend.twinx()
        ax_ret.plot(log_return.index, log_return, linewidth=1.0, alpha=0.3, label="returns")
        ax_ret.axhline(0, linestyle=":", linewidth=1)
        ax_ret.set_title("Trend")

        ax_vol.bar(vol.index, vol.to_numpy())
        ax_vol.set_title("Volume")

        ax_innov.plot(innov.index, innov.to_numpy(), linewidth=1.0, alpha=0.6)
        ax_innov.axhline(0, linestyle=":", linewidth=1)
        ax_innov.set_title("Standardized residuals")

        fig.tight_layout()
        return fig, (ax_close, ax_trend, ax_vol, ax_innov)
