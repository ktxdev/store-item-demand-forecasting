from abc import ABC, abstractmethod

import pandas as pd


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for feature engineering strategy.
        :param:
            data (pd.DataFrame): The dataframe to engineer features for.
        :return:
        """
        pass


class DateFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['month'] = df_copy.date.dt.month
        df_copy['day_of_month'] = df_copy.date.dt.day
        df_copy['day_of_year'] = df_copy.date.dt.dayofyear
        df_copy['day_of_week'] = df_copy.date.dt.dayofweek
        df_copy['year'] = df_copy.date.dt.year
        df_copy["is_weekend"] = df_copy.date.dt.weekday // 4
        df_copy['is_month_start'] = df_copy.date.dt.is_month_start.astype(int)
        df_copy['is_month_end'] = df_copy.date.dt.is_month_end.astype(int)
        return df_copy


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initialize the feature engineer with a specific feature engineering strategy.
        :param:
            strategy (FeatureEngineeringStrategy): The feature engineering strategy to be use
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets the feature engineering strategy.
        :param:
            strategy (FeatureEngineeringStrategy): The new feature engineering strategy to be use
        """
        self._strategy = strategy

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering on a dataframe using the given strategy.
        :param:
            df (pd.DataFrame): The dataframe to engineer features for.
        :return:
            pd.DataFrame: The dataframe after feature engineering.
        """
        return self._strategy.engineer_features(df)
