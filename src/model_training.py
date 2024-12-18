from typing import Tuple, Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.data_splitting import DataSplitter, DateDataSplittingStrategy
from src.model_building import ModelBuilder, LinearRegressionStrategy
from src.model_evaluation import ModelEvaluator, RegressionPipelineEvaluationStrategy


class ModelTrainer:
    def __init__(self,
                 model_builder: ModelBuilder,
                 model_evaluator: ModelEvaluator,
                 data_splitter: DataSplitter):
        self._model_builder = model_builder
        self._model_evaluator = model_evaluator
        self._data_splitter = data_splitter

    def train_and_evaluate(self, data: pd.DataFrame) -> Tuple[GridSearchCV, Dict[str, any]]:
        # Split the data
        X_train, X_test, y_train, y_test = self._data_splitter.split(data)
        # Train the model
        grid_search = self._model_builder.build_model(X_train, y_train)
        # Evaluating the model
        metrics = self._model_evaluator.evaluate(grid_search, X_test, y_test)
        return grid_search, metrics


if __name__ == '__main__':
    import os
    import numpy as np
    from anaysis.src.feature_engineering import FeatureEngineer, DateFeatureEngineeringStrategy, \
        LagFeatureEngineeringStrategy, RollingMeanFeatureEngineeringStrategy, \
        ExponentiallyWeightedMeanEngineeringStrategy

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data/train.csv')
    data = pd.read_csv(data_path)

    feature_engineer = FeatureEngineer(DateFeatureEngineeringStrategy())
    data = feature_engineer.engineer_features(data)

    lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
    lag_feature_engineering_strategy = LagFeatureEngineeringStrategy(lags)
    feature_engineer.set_strategy(lag_feature_engineering_strategy)
    data = feature_engineer.engineer_features(data)

    windows = [365, 546]
    rolling_mean_feature_engineering_strategy = RollingMeanFeatureEngineeringStrategy(windows)
    feature_engineer.set_strategy(rolling_mean_feature_engineering_strategy)
    data = feature_engineer.engineer_features(data)

    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    exponentially_weighted_mean_engineering_strategy = ExponentiallyWeightedMeanEngineeringStrategy(alphas, lags)
    feature_engineer.set_strategy(exponentially_weighted_mean_engineering_strategy)
    data = feature_engineer.engineer_features(data)

    data = data.drop(columns=['is_month_start', 'is_month_end', 'day_of_month'])

    data = pd.get_dummies(data, columns=['store', 'item', 'day_of_week', 'month'])

    data['sales'] = np.log1p(data['sales'].values)

    data = data.fillna(data.mean())

    data_spliter = DataSplitter(DateDataSplittingStrategy())
    evaluation_strategy = ModelEvaluator(RegressionPipelineEvaluationStrategy())
    building_strategy = ModelBuilder(LinearRegressionStrategy())

    trainer = ModelTrainer(building_strategy, evaluation_strategy, data_spliter)

    grid_search, metrics = trainer.train_and_evaluate(data)
