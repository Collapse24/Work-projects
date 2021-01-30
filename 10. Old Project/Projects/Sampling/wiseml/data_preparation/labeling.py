import pandas as pd
import numpy as np

def add_fthm_label(data: pd.DataFrame, target_column: str, thresholds: list, window_size: int):
    """
    Fixed Time-Horizon model labeling

    0 - keep
    1 - up
    2 - down
    :param data:            source pandas dataframe
    :param target_column:   name of price column at which data will be labeled
    :param thresholds:       'keep' threshold, thresholds[0] - upper limit, thresholds[1] - lower limit
    :param window_size:     how many observations in the future we predict
    :return:                pandas dataframe with target label `target`
    """
    data['price_target'] = data[target_column].shift(-window_size)
    data.dropna(inplace=True)

    data['target'] = 0
    data.loc[data[target_column] + (data[target_column] * thresholds[0]) < data['price_target'], 'target'] = 1
    data.loc[data[target_column] - (data[target_column] * thresholds[1]) > data['price_target'], 'target'] = 2

    data.drop(['price_target'], axis=1, inplace=True)

    return data

# Нужно объеденить с функцией выше или переделать (дублирование кода)
def add_fthm_bin_label(data: pd.DataFrame, target_column: str, threshold: float, window_size: int):
    """
    Fixed Time-Horizon model binary labeling

    0 - keep
    1 - up
    2 - down
    :param data:            source pandas dataframe
    :param target_column:   name of price column at which data will be labeled
    :param threshold:       'keep' threshold
    :param window_size:     how many observations in the future we predict
    :return:                pandas dataframe with target label `target`
    """
    data['price_target'] = data[target_column].shift(-window_size)
    data.dropna(inplace=True)

    data['target'] = 0
    data.loc[data[target_column] + (data[target_column] * threshold) < data['price_target'], 'target'] = 1
    data.loc[data[target_column] - (data[target_column] * threshold) > data['price_target'], 'target'] = 1

    data.drop(['price_target'], axis=1, inplace=True)

    return data


def add_tbm_label(data: pd.DataFrame, target_column: str, thresholds: list, window_size: int):
    """
    Triple-Barrier Model labeling

    0 - keep
    1 - up
    2 - down
    :param data:            source pandas data frame
    :param target_column:   name of price column at which data will be labeled
    :param thresholds:       'keep' threshold, if > 1.0 then will calculate volatility threshold, thresholds[0] - upper limit, thresholds[1] - lower limit
    :param window_size:     how many observations in the future we predict
    :return:                pandas data frame with target label `target`
    """
    upper_bound = np.inf
    lower_bound = -np.inf

    if thresholds[0] > 0.0:
        upper_bound = thresholds[0]
    if thresholds[1] > 0.0:
        lower_bound = -thresholds[1]

    data['target'] = 0

    for i in range(window_size, 0, -1):
        data['wt_pct'] = data[target_column].pct_change(periods=i).shift(-i)
        data.loc[data['wt_pct'] > upper_bound, 'target'] = 1
        data.loc[data['wt_pct'] < lower_bound, 'target'] = 2

    data.drop(columns='wt_pct', axis=1, inplace=True)

    data = data.iloc[:-window_size, :]

    return data

# 1. Необходимо добавить расчет волатильности;
# 2. Добавить документацию.
def vol_horiz(data: pd.DataFrame, price_column='close', horizon=10):
    data['price_target'] = data[price_column].shift(-horizon)
    data.dropna(inplace=True)

    data['target_1'] = 0
    data.loc[((data['price_target'] / data[price_column] - 1) > data['volatility_wt']), 'target_1'] = 1
    data.loc[((data['price_target'] / data[price_column] - 1) < -data['volatility_wt']), 'target_1'] = 2

    return data
