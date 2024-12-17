from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


# Utility function to add random noise
def add_random_noise(size: int, scale: float = 1.6) -> np.ndarray:
    return np.random.normal(scale=scale, size=size)


# Base class for feature engineering strategies
class FeatureEngineeringStrategy(ABC):
    def __init__(self, scale_noise: float = 1.6):
        self.scale_noise = scale_noise

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# Lag Feature Engineering Strategy
class LagFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def __init__(self, lags: List[int] = None, scale_noise: float = 1.6):
        super().__init__(scale_noise)
        self.lags = lags

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for lag in self.lags:
            df_copy[f'sales_lag_{lag}'] = (
                    df_copy.groupby(['store', 'item'])['sales']
                    .transform(lambda x: x.shift(lag)) + add_random_noise(len(df_copy), self.scale_noise)
            )
        return df_copy


# Rolling Mean Feature Engineering Strategy
class RollingMeanFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def __init__(self, windows: List[int] = None, scale_noise: float = 1.6):
        super().__init__(scale_noise)
        self.windows = windows

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for window in self.windows:
            df_copy[f'sales_roll_mean_{window}'] = (
                    df_copy.groupby(['store', 'item'])['sales']
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean())
                    + add_random_noise(len(df_copy), self.scale_noise)
            )
        return df_copy


# Exponentially Weighted Mean Feature Engineering Strategy
class ExponentiallyWeightedMeanEngineeringStrategy(FeatureEngineeringStrategy):
    def __init__(self, alphas: List[float] = None, lags: List[int] = None):
        super().__init__()
        self.alphas = alphas or [0.95, 0.9, 0.8, 0.7, 0.5]
        self.lags = lags

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for alpha in self.alphas:
            for lag in self.lags:
                df_copy[f'sales_ewm_alpha_{str(alpha).replace(".", "")}_lag_{lag}'] = (
                    df_copy.groupby(['store', 'item'])['sales']
                    .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
                )
        return df_copy


# Date Feature Engineering Strategy
class DateFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    def __init__(self, date_column: str = 'date', features: List[str] = None):
        super().__init__()
        self.date_column = date_column
        self.features = features or [
            'month', 'day_of_month', 'day_of_year', 'day_of_week',
            'year', 'is_weekend', 'is_month_start', 'is_month_end'
        ]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.date_column not in df.columns:
            raise ValueError(f"'{self.date_column}' column not found in DataFrame.")

        df_copy = df.copy()
        df_copy[self.date_column] = pd.to_datetime(df_copy[self.date_column], errors='coerce')

        feature_methods = {
            'month': lambda x: x.dt.month,
            'day_of_month': lambda x: x.dt.day,
            'day_of_year': lambda x: x.dt.dayofyear,
            'day_of_week': lambda x: x.dt.dayofweek,
            'year': lambda x: x.dt.year,
            'is_weekend': lambda x: (x.dt.weekday >= 5).astype(int),
            'is_month_start': lambda x: x.dt.is_month_start.astype(int),
            'is_month_end': lambda x: x.dt.is_month_end.astype(int)
        }

        for feature in self.features:
            if feature in feature_methods:
                df_copy[feature] = feature_methods[feature](df_copy[self.date_column])
            else:
                raise ValueError(f"Feature '{feature}' is not supported.")
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
