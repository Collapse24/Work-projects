from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from wiseml.feature_engineering.ohlc.base import OHLCFeatureEngineer


class OHLCStandardFeatureEngineer(OHLCFeatureEngineer):

    default_period_list = [10, 30, 60]
    default_ma_window_list = [10, 80, 120]  # default moving average period list
    default_z_score_rolling_window = 1440  # one day

    def __init__(self, period: Tuple[str, str] = ('1', 'min'), period_list: Optional[List[int]] = None,
                 moving_average_window_list: Optional[List[int]] = None, z_score_rolling_window: Optional[int] = None,
                 *args, **kwargs):
        """

        :param period:      tuple that describe period of one observation in data (1 min or 1000 volume for example)
        :param period_list: list of periods for creating features based on n-prev observation period
                            (n its value from period list)
        :param moving_average_window_list:  list of periods (number of observations) for creating moving averages features
        :param z_score_rolling_window:  number of prev observations for z-scoring (we can`t take info from future)
        """
        super().__init__(*args, **kwargs)

        self.features: pd.DataFrame = pd.DataFrame(index=self._data.index).sort_index()
        self.features_count: int = 0
        self.features_description: Dict[str, str] = {}
        self.period: Tuple[str, int] = period
        self._period_str = f"{self.period[0]} {self.period[1]}"
        self.feature_name_generator = self.__feature_name_generator()
        self.period_list: List[int] = period_list or self.default_period_list
        self.moving_average_window_list = moving_average_window_list or self.default_ma_window_list
        self.z_score_rolling_window = z_score_rolling_window or self.default_z_score_rolling_window

    def __feature_name_generator(self):
        while True:
            yield f"s_f{self.features_count}"
            self.features_count += 1

    @property
    def _next_feature_name(self):
        return next(self.feature_name_generator)

    def add_return_features(self) -> None:
        feature_name = self._next_feature_name
        feature_description = f"{self._period_str} return like 'close / open - 1'"
        self.features[feature_name] = self._data[self.close] / self._data[self.open] - 1
        self.features_description[feature_name] = feature_description

        for period in self.period_list:
            feature_name = self._next_feature_name
            feature_description = f"{period} prev observations close return: " \
                f"'curr_close / prev{period}observations_close - 1'"
            self.features[feature_name] = self._data[self.close] / self._data[self.close].shift(period) - 1
            self.features_description[feature_name] = feature_description

    def add_pct_change_features(self) -> None:
        for period in self.period_list:
            for column in self.ohlc_list:
                feature_name = self._next_feature_name
                feature_description = f"{column} pct change feature at {period} prev observations"
                self.features[feature_name] = self._data[column].pct_change(period)
                self.features_description[feature_name] = feature_description

    def add_moving_average_features(self) -> None:
        for window in self.moving_average_window_list:
            for column in self.ohlc_list:
                feature_name = self._next_feature_name
                feature_description = f"{column} moving average based on {window} prev observations"
                self.features[feature_name] = self._data[column] / self._data[column].rolling(window).mean() - 1
                self.features_description[feature_name] = feature_description

    def add_exp_moving_average_features(self) -> None:
        for window in self.moving_average_window_list:
            for column in self.ohlc_list:
                feature_name = self._next_feature_name
                feature_description = f"{column} exp moving average based on {window} prev observations"
                self.features[feature_name] = self._data[column] / self._data[column].ewm(window).mean() - 1
                self.features_description[feature_name] = feature_description

    def add_z_score_features(self) -> None:
        window = self.z_score_rolling_window
        for column in self.ohlc_list:
            feature_name = self._next_feature_name
            feature_description = f"{column} z-score by {window} prev observations"
            self.features[feature_name] = (self._data[column] - self._data[column].rolling(window).mean()) / self._data[column].rolling(window).std()
            self.features_description[feature_name] = feature_description

    def add_volume_features(self) -> None:
        """
        Volume features
        """
        # log
        feature_name = self._next_feature_name
        feature_description = "Volume log"
        self.features[feature_name] = self._data[self.volume].apply(np.log)
        self.features_description[feature_name] = feature_description

        # rate of change
        for period in self.period_list:
            feature_name = self._next_feature_name
            feature_description = f"Rate of change volume by {period} prev observations"
            self.features[feature_name] = self._data[self.volume].pct_change(period)
            self.features_description[feature_name] = feature_description

        # features based on moving average
        for window in self.moving_average_window_list:
            moving_average = self._data[self.volume].rolling(window).mean()
            # log moving average of volume
            feature_name = self._next_feature_name
            feature_description = f"Log of moving average of volume by {window} prev observations"
            self.features[feature_name] = moving_average.apply(np.log)
            self.features_description[feature_name] = feature_description
            # minute volume vs window moving average
            feature_name = self._next_feature_name
            feature_description = f"volume vs moving average by {window} prev observations"
            self.features[feature_name] = self._data[self.volume] / moving_average - 1
            self.features_description[feature_name] = feature_description

        # exp moving average
        for window in self.moving_average_window_list:
            feature_name = self._next_feature_name
            feature_description = f"exp moving average of volume by {window} prev observations"
            self.features[feature_name] = self._data[self.volume].ewm(window)
            self.features_description[feature_name] = feature_description

        # z-score
        window = self.z_score_rolling_window
        feature_name = self._next_feature_name
        feature_description = f"Z-score of volume by {self.z_score_rolling_window} prev observations"
        self.features[feature_name] = (self._data[self.volume] - self._data[self.volume].rolling(window).mean()) \
                                      / self._data[self.volume].rolling(window).std()
        self.features_description[feature_name] = feature_description

    def add_all_features(self) -> pd.DataFrame:
        self.add_return_features()
        self.add_pct_change_features()
        self.add_moving_average_features()
        self.add_exp_moving_average_features()
        self.add_z_score_features()
        self.add_volume_features()

        return pd.concat([self._data, self.features], axis=1)
