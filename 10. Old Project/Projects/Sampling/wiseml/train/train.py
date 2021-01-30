from abc import ABC, abstractmethod
from typing import Dict, Optional

from wiseml.models.types.task_type import TaskType
from wiseml.models.types.model_type import ModelType
from wiseml.validation.validator import Validator


class Trainer(ABC):

    def __init__(self, task_type: TaskType, model_type: ModelType, params: Dict, metric: str,
                 validator: Optional[Validator] = None):
        self.task_type = task_type
        self.model_type = model_type
        self.params = params
        self.metric = metric
        self.validator = validator

    @abstractmethod
    def train(self, X, y, X_train): pass
