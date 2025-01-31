from pickle import load
import pandas as pd
import numpy as np

from tsfresh.feature_extraction.settings import from_columns
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

def add_returns(data: pd.DataFrame, intervals: list,  res_col_name: str, column: str = 'close'):
    """
        :param data:            source pandas dataframe
        :param intervals        intervals between bars for returns calculation
        :param column:          name of column at which returns will be calculated
        :param res_col_name:    the row with which the result column name begins
        :return:                source dataframe with returns
    """
    for i in intervals:
        data[f'{res_col_name}_{i}'] = data[column].pct_change(i)

    return data

def add_lags(data: pd.DataFrame, shifts: list,  res_col_name: str, column: str = 'close'):
    """
        :param data:            source pandas dataframe
        :param shifts           lag shifts
        :param column:          name of column at which lags will be calculated
        :param res_col_name:    the row with which the result column name begins
        :return:                source dataframe with lags
    """
    for i in shifts:
        data[f'{res_col_name}_{i}'] = data[column].shift(i)

    return data

def add_log_returns(data: pd.DataFrame, res_col_name: str, column: str = 'close'):
    """
        :param data:            source pandas dataframe
        :param res_col_name:    the row with which the result column name begins
        :param column:          name of column at which log returns will be calculated
        :return:                source dataframe with log returns
    """
    data[f'{res_col_name}_log'] = (np.log(data[column].pct_change() + 1) / np.log(data[column]))
    data.loc[0, f'{res_col_name}_log'] = 0

    return data

def add_time_features(data: pd.DataFrame, column: str = 'datetime'):
    """
        :param data:            source pandas dataframe
        :param column:          name of column at which time features will be calculated
        :return:                source dataframe with time features
    """
    data['datetime'] = pd.to_datetime(data['datetime'])

    data['month'] = data[column].dt.month
    data['day'] = data[column].dt.day
    data['hour'] = data[column].dt.hour
    data['minute'] = data[column].dt.minute

    return data

def add_tsfresh_features(data: pd.DataFrame, tsfresh_feat: list, column: str = 'close_returns_log'):
    """
        :param data:            source pandas dataframe
        :param tsfresh_feat     list of selected features generated by tsfresh package
        :param column:          name of column at which features will be calculated
        :return:                source dataframe with features generated by tsfresh package
    """
    settings_ts = from_columns(tsfresh_feat)

    df_shift, y = make_forecasting_frame(data[column], kind="price", max_timeshift=100, rolling_direction=1)
    X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value", impute_function=impute,
                         n_jobs=0, default_fc_parameters=settings_ts['value'], disable_progressbar=True)
    X = X[tsfresh_feat]
    data = data.join(X)

    return data
