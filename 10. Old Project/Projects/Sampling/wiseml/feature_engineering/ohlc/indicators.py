from typing import Optional

import ta
import pandas as pd
import numpy as np

from wiseml.feature_engineering.ohlc.base import OHLCFeatureEngineer


class OHLCIndicatorsFeatureEngineer(OHLCFeatureEngineer):
    """
    Technical indicators features
    """
    def __init__(self, volatility_period: Optional[int], momentum_period: Optional[int], trend_period: Optional[int],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.volatility_period = volatility_period
        self.momentum_period = momentum_period
        self.trend_period = trend_period

    def add_volatility_indicators(self, fillna: bool = False):
        return ta.add_volatility_ta(self._data, self.high, self.low, self.close, fillna=fillna)

    def add_trend_indicators(self, fillna: bool = False):
        return ta.add_trend_ta(self._data, self.high, self.low, self.close, fillna=fillna)

    def add_volume_indicators(self, fillna: bool = False):
        return ta.add_volume_ta(self._data, self.high, self.low, self.close, self.volume, fillna=fillna)

    def add_momentum_indicators(self, fillna: bool = False):
        return ta.add_momentum_ta(self._data, self.high, self.low, self.close, self.volume, fillna=fillna)

    def add_returns(self, fillna: bool = False):
        return ta.add_others_ta(self._data, self.close, fillna=fillna)

    def add_all_features(self, fillna: bool = False):
        return ta.add_all_ta_features(self._data, self.open, self.high, self.low, self.close, self.volume, fillna=fillna)
