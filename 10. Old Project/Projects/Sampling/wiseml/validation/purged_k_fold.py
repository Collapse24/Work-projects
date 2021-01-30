from typing import Optional, Union

import pandas as pd
import numpy as np

from wiseml.validation.validator import Validator


class PurgedKFold(Validator):

    def __init__(self, n_splits: int = 5,  overlap: Union[int, float] = 0.0001):

        super().__init__(n_splits)
        self.overlap = overlap

    def split(self, X, y=None, return_splitted_data=False):
        indices = np.arange(X.shape[0], dtype=int)

        if type(self.overlap) == float:
            overlap_count = int(X.shape[0] * self.overlap)
        else:
            overlap_count = self.overlap

        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, j in test_ranges:

            test_indices = indices[i:j]
            left_train = []
            if i != 0:
                left_train = indices[0:i - overlap_count]

            # right part of train
            right_train = []
            if j != X.shape[0]:
                right_train = indices[j + overlap_count:X.shape[0]]

            train_indices = np.concatenate((np.array(left_train, int), np.array(right_train, int)))

            if not return_splitted_data:
                yield train_indices, test_indices

            else:
                yield X.iloc[train_indices], y.iloc[test_indices]
