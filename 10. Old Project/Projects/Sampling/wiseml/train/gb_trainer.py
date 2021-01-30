from typing import Optional

import numpy as np

from wiseml.train.train import Trainer
from wiseml.models.types.model_type import GBModelType
from wiseml.models.gb_models.lgbm import LGBModel
from wiseml.train.metric import Metric, metric_func


class GBTrainer(Trainer):

    model_task_mapper = {
        GBModelType.LGBM: LGBModel
    }

    def __init__(self, num_iterations: Optional[int] = None, early_stopping_rounds: Optional[int] = None,
                 verbose: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_iterations = num_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self._model = self.model_task_mapper[self.model_type]()

    def train(self, X, y, X_test=None):

        if self.validator:
            splitter = self.validator.split(X, y)
            i = 0
            scores = []
            for (train_idx, valid_idx) in splitter:
                print(f"##### Fold {i} #####")
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
                model = self._model.fit(X, y, valid_set=(X_val, y_val), params=self.params, num_iterations=self.num_iterations,
                                        early_stopping_rounds=self.early_stopping_rounds, verbose=self.verbose,
                                        eval_metric=self.metric)
                predict = model.predict(X_val)
                score = metric_func[Metric[self.metric]](y_val, predict)
                scores.append(score)
                print(f"Score on {i}th fold: {score}")
                i += 1

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"Mean score: {mean_score}    Std score: {std_score}")

        model = self._model.fit(X, y, params=self.params, early_stopping_rounds=self.early_stopping_rounds,
                                num_iterations=self.num_iterations, verbose=self.verbose)

        if X_test:
            predict = model.predict(X_test)
            return {
                'model': model,
                'predict': predict
            }
