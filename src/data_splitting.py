from typing import Tuple

import pandas as pd
from abc import ABC, abstractmethod


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Abstract method for splitting data into train and test sets.
        :param:
            data (pd.DataFrame): Data to be split.
        :return:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The train and test sets.
        """
        pass


class DateDataSplittingStrategy(DataSplittingStrategy):
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into train and test sets using a date range.
        :param:
            data (pd.DataFrame): Data to be split.
        :return:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The train and test sets.
        """

        train = data.loc[(data['date'] < '2017-10-01'), :]
        val = data.loc[(data['date'] >= '2017-10-01'), :]

        cols_to_drop = ['date', 'sales']

        X_train = train.drop(cols_to_drop, axis=1)
        y_train = train['sales']

        X_val = val.drop(cols_to_drop, axis=1)
        y_val = val['sales']

        return X_train, X_val, y_train, y_val


class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter class with a specific strategy.

        :param:
            strategy (DataSplittingStrategy): The strategy to split the data into train and test sets.
        """
        self._strategy = strategy

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return self._strategy.split_data(data)
