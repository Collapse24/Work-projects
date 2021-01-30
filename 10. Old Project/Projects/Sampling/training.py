import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from wiseml.validation.purged_k_fold import PurgedKFold


from typing import Union

from imblearn.over_sampling import SMOTE


def train(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: list, y_val: list, params: dict, weights=None):
    """
    Training lightgbm Gradient Boosting model

    :param X_train:                 train set
    :param X_val:                   validation set
    :param y_train:                 answers for train set
    :param y_val:                   answers for validation set
    :param params:                  Gradient Boosting algorithm parameters
    :param num_boost_round:         Number of trees
    :param early_stopping_rounds:    Number of iterations for early stopping
    :param verbose_eval:            Number of iterations before intermediate result prints
    :param weights:                 Weights for classes, if there is class disbalance
    :return:                        lightgbm booster
    """
    if weights is None:
        dtrain = lgb.Dataset(X_train, label=y_train)
    else:
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
    dvalid = lgb.Dataset(X_val, label=y_val)

    params_t = params.copy()
    num_boost_round = params_t.pop('num_boost_round')
    early_stopping_rounds = params_t.pop('early_stopping_rounds')
    verbose_eval = params_t.pop('verbose_eval')
    model = lgb.train(params_t,
                      dtrain,
                      num_boost_round,
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=verbose_eval
                      )

    return model


# Добавить веса
def lgb_train(X: pd.DataFrame,
              y: pd.DataFrame,
              optim_params: dict,
              model_params: dict,
              task_params: dict,
              device_params: dict,
              train_part: float = 0.7, tuning: bool = False, n_splits: int = 5, overlap: Union[int, float] = 60
              ):
    params = dict(**model_params, **task_params, **device_params, **optim_params)

    params['max_depth'] = int(params['max_depth'])
    params['min_child_samples'] = int(params['min_child_samples'])

    if tuning:
        metric_val = []

        pkf = PurgedKFold(5, 210)
        splitter = pkf.split(X, y)

        for (train_idx, test_idx) in splitter:

            X_train, y_train = X.iloc[train_idx], list(y.iloc[train_idx])
            X_valid, y_valid = X.iloc[test_idx], list(y.iloc[test_idx])

            model = train(X_train, X_valid, y_train, y_valid, params)

            predictions = model.predict(X_valid, num_iteration=model.best_iteration)

            classes_predictions = list(map(lambda x: np.where(x == np.max(x))[0][0], predictions))

            y_pred_conf = []
            y_conf = []

            min_confidence = 0
            for i in range(len(predictions)):
                if np.max(predictions[i]) > min_confidence:
                    y_conf.append(y_valid[i])
                    y_pred_conf.append(classes_predictions[i])

            pre_score = accuracy_score(y_conf, y_pred_conf)
            #pre_score = model.best_score['valid'][params['metric']]
            metric_val.append(pre_score)

        return np.mean(metric_val)

    else:

        X_train, y_train = X[:int(len(X) * train_part)], list(y[:int(len(y) * train_part)])
        X_valid, y_valid = X[int(len(X) * train_part):], list(y[int(len(y) * train_part):])

        model = train(X_train, X_valid, y_train, y_valid, params)

        return model
