from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from data.get_stock_data import OHLCV, OHLCVBatch


class PredictValues(TypedDict):
    ticker: str
    ret: float
    ret_se: float
    p_white: float
    p_norm: float
    p_heterod: float


@runtime_checkable
class MeanModel(Protocol):
    ohlcv: OHLCV

    @classmethod
    def fit(cls, ohlcv: OHLCV) -> "MeanModel": ...

    def history(self, z: float = 2.0) -> pd.DataFrame: ...

    def predict(self) -> PredictValues: ...


@dataclass(slots=True)
class MeanModelBatch:
    batch: OHLCVBatch
    models: dict[str, MeanModel]
    errors: dict[str, str]

    @classmethod
    def fit(cls, batch: OHLCVBatch, model: type[MeanModel]) -> "MeanModelBatch":
        import warnings

        from tqdm import tqdm

        models: dict[str, MeanModel] = {}
        errors: dict[str, str] = {}

        total = len(batch.items)
        failed = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with tqdm(total=total, desc="Fitting", unit="ticker", leave=True, dynamic_ncols=True) as pbar:
                for ticker, ohlcv in batch.items.items():
                    try:
                        models[ticker] = model.fit(ohlcv)
                    except Exception as e:
                        errors[ticker] = str(e) or type(e).__name__
                        failed += 1

                    pbar.set_description_str(f"Fitting [{ticker}]")
                    pbar.set_postfix_str(f"FAILED={failed}", refresh=True)
                    pbar.update(1)

        return cls(batch=batch, models=models, errors=errors)

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame([m.predict() for m in self.models.values()])
        return df.set_index("ticker").sort_index()

    def ER(self) -> pd.DataFrame:
        return self.summary().loc[:, ["ret"]]
