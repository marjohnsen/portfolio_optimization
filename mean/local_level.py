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
class LLModel(MeanModel):
    ohlcv: OHLCV
    model: UnobservedComponents
    res: Any
    idx: pd.Index

    @classmethod
    def fit(cls, ohlcv: OHLCV) -> "LLModel":
        if ohlcv.failed or ohlcv.df.empty:
            raise ValueError(f"Cannot fit model: {ohlcv.ticker} - {ohlcv.msg}")

        log_close = np.log(ohlcv.df["Close"].astype(float))
        log_return = log_close.diff().fillna(0.0)
        mod = UnobservedComponents(log_return, level="llevel")
        res = mod.fit(disp=False)

        return cls(ohlcv=ohlcv, model=mod, res=res, idx=log_return.index)

    def history(self, z: float = 2.0) -> pd.DataFrame:
        level = self.res.level["filtered"]
        level_se = np.sqrt(self.res.filtered_state_cov[0, 0, :])

        return pd.DataFrame(
            {
                "level": np.expm1(level),
                "level_se": np.exp(level) * level_se,
                "level_lo": np.expm1(level - z * level_se),
                "level_hi": np.expm1(level + z * level_se),
            },
            index=self.idx,
        )

    def predict(self) -> PredictValues:
        level = np.asarray(self.res.level["filtered"]).ravel()
        level_se = np.sqrt(np.asarray(self.res.filtered_state_cov[0, 0, :])).ravel()

        return {
            "ticker": self.ohlcv.ticker,
            "ret": float(level[-1]),
            "ret_se": float(level_se[-1]),
            "p_white": round(float(np.min(np.asarray(self.res.test_serial_correlation(method=None))[0, 1, :])), 2),
            "p_norm": round(float(np.asarray(self.res.test_normality(method=None))[0][1]), 2),
            "p_heterod": round(float(np.asarray(self.res.test_heteroskedasticity(method=None))[0][1]), 2),
        }

    def plot(self, skip: int = 0, z: float = 2.0, figsize: Tuple[float, float] = (15, 15)):
        import matplotlib.pyplot as plt

        idx = pd.Index(self.idx)

        hist = self.history(z=z).reindex(idx).iloc[skip:]
        ticker = self.ohlcv.ticker
        close = self.ohlcv.df["Close"].reindex(self.idx).iloc[skip:]
        vol = self.ohlcv.df["Volume"].reindex(idx).iloc[skip:].fillna(0)

        log_return = np.log(close).diff()

        innov = self.res.filter_results.standardized_forecasts_error
        innov = innov[0] if np.ndim(innov) == 2 else innov
        innov = pd.Series(innov, index=idx).iloc[skip:]

        fig, (ax_close, ax_lvl, ax_vol, ax_innov) = plt.subplots(
            4, 1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": (2, 2, 1, 1)}
        )

        er = (1 + hist["level"].iloc[-1]) ** (8 * 250) - 1
        lo = (1 + hist["level_lo"].iloc[-1]) ** (8 * 250) - 1
        hi = (1 + hist["level_hi"].iloc[-1]) ** (8 * 250) - 1

        fig.suptitle(f"Local Level — {ticker} — ER (annualized): {er:.2%} [{lo:.2%}, {hi:.2%}] — z={z:.2f}")

        ax_close.plot(close.index, close, linewidth=1.5)
        ax_close.set_title("Close Price")

        ax_lvl.fill_between(hist.index, hist["level_lo"], hist["level_hi"], alpha=0.2)
        ax_lvl.plot(hist.index, hist["level"], label="level")

        ax_ret = ax_lvl.twinx()
        ax_ret.plot(log_return.index, log_return, linewidth=1.0, alpha=0.3, label="returns")

        for ax in (ax_lvl, ax_ret):
            lo, hi = ax.get_ylim()
            m = max(abs(lo), abs(hi))
            ax.set_ylim(-m, m)
            ax.axhline(0, linestyle=":", linewidth=1, color="k", alpha=0.7)

        ax_lvl.set_title("Level")

        ax_vol.bar(vol.index, vol.to_numpy())
        ax_vol.set_title("Volume")

        ax_innov.plot(innov.index, innov.to_numpy(), linewidth=1.0, alpha=0.6)
        ax_innov.axhline(0, linestyle=":", linewidth=1)
        ax_innov.set_title("Standardized innovations")

        fig.tight_layout()
        return fig, (ax_close, ax_ret, ax_vol, ax_innov)
