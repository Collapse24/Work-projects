from abc import ABC, abstractmethod
from typing import Generator, Union, Tuple

import pandas as pd
import numpy as np


class Validator(ABC):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    @abstractmethod
    def split(self, X, y, return_splitted_data) -> Generator[Tuple, None, None]:
        pass
