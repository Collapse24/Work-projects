from typing import Union

import pandas as pd
import numpy as np


class OHLCFeatureEngineer:
    """
    Base class for OHLC feature engneering
    """
    def __init__(self, source_data: Union[str, pd.DataFrame], open_column: str = 'open', high_column: str = 'high',
                 low_column: str = 'low', close_column: str = 'close', volume_column: str = 'volume'):
        """

        :param source_data:     source OHLC(V) pandas dataframe or path to csv
        :param open_column:     name of open column
        :param high_column:     name of high column
        :param low_column:      name of low column
        :param close_column:    name of close column
        :param volume_column:   name of volume column
        """
        if type(source_data) == str:
            self._data = pd.read_csv(source_data)
        else:
            self._data = source_data
        self.open: str = open_column
        self.high: str = high_column
        self.low: str = low_column
        self.close: str = close_column
        self.volume: str = volume_column
        self.ohlc_list = [self.open, self.high, self.low, self.close]
