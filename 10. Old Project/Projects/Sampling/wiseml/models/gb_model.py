from typing import Callable, ClassVar, Dict, Optional, Union, Tuple

import numpy as np

from wiseml.train.metric import Metric
from wiseml.models.model import Model
from wiseml.models.types.model_type import GBModelType
from wiseml.models.types.task_type import TaskType


class GBModel(Model):
    """
        Gradient boosting model base class
    """

    def __init__(self, task_type: TaskType):
        super().__init__()

        self.task_type = task_type

        # optional for detect num of classes in classification problem
        self.num_class: Optional[int] = None

        # class or method for create dataset (lgb.Dataset for example)
        self.dataset_class: Union[None, Callable, ClassVar] = None

        # method for training in specific framework
        self.train_function: Optional[Callable] = None

        # alias of early stopping rounds parameter in specific framework
        self.early_stopping_round_alias: Optional[str] = None

        # alias of eval metric parameter in specific framework
        self.eval_metric_alias: Optional[str] = None

        # alias of num rounds parameter in specific framework
        self.num_rounds_alias: Optional[str] = None

        # alias of valid set parameter in specific framework
        self.valid_set_alias: Optional[str] = None

        # alias of verbose parameter in specific framework
        self.verbose_alias: Optional[str] = None

    def fit(self, X, y, params: Dict, num_iterations: int, eval_metric: Metric, early_stopping_rounds: int,
            verbose: Union[bool, int], valid_set: Optional[Tuple] = None):
        specific_parameters = {}    # dictionary for mapping standard parameters to specific framework`s parameters
        train_set = self.dataset_class(X, y)

        if self.task_type == TaskType.Classification:
            self.num_class = len(np.unique(y))

        if valid_set:
            valid_set = self.dataset_class(valid_set[0], valid_set[1])
            specific_parameters[self.valid_set_alias] = valid_set
        specific_parameters[self.num_rounds_alias] = num_iterations
        specific_parameters[self.eval_metric_alias] = eval_metric
        specific_parameters[self.early_stopping_round_alias] = early_stopping_rounds
        specific_parameters[self.verbose_alias] = verbose
        specific_parameters['params'] = params

        train_result = self.train_function(**specific_parameters)