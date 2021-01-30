from abc import ABC, abstractmethod, abstractclassmethod
from typing import Dict, Optional

import pandas as pd
import numpy as np

from wiseml.models.types.task_type import TaskType
from wiseml.models.types.model_type import ModelType


class TrainSet:
    def __init__(self, X: pd.DataFrame, y: pd.Series):

        if X.shape[0] != y.shape[0]:
            raise ValueError("Len of X and y should be equal")

        self.indices = np.arange(X.shape[0])
        self.X = X
        self.y = y

    def __iter__(self):
        for i in self.indices:
            yield self.X.iloc[i], self.y.iloc[i]


class Model(ABC):

    def __init__(self):
        self.task_type: Optional[TaskType] = None
        self.model_type: Optional[ModelType] = None

    @abstractmethod
    def fit(self, *args, **kwargs): pass

    @abstractmethod
    def predict(self, y_true, y_pred, *args, **kwargs): pass

    @classmethod
    @abstractmethod
    def save(cls, path): pass

    @classmethod
    @abstractmethod
    def load(cls, path): pass
