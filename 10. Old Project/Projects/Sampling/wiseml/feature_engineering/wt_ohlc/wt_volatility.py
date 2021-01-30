import numpy as np
import pandas as pd

def add_volatility(data: pd.DataFrame, target_column: str, window: int, span0: float):
    """
        :param data:            source pandas dataframe
        :param target_column:   name of price column at which data will be labeled
        :param window:          how many observations use for volatility estimation
        :param span0:           decay in terms of span
        :return:                pandas dataframe with volatility 'wt_volatility'
    """
    pct_price_changes = np.abs(data[target_column].pct_change())
    std_pct = pct_price_changes.ewm(span=span0, min_periods=(window - 1)).std()
    mean_pct = pct_price_changes.ewm(span=span0, min_periods=(window - 1)).mean()
    data['wt_volatility'] = std_pct + mean_pct

    data.dropna(inplace=True, axis=0)

    return data