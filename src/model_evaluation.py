import logging
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from src.model_building import smape_scorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_pipeline(self, grid_search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Abstract method to evaluate a pipeline.

        :param:
            pipeline (Pipeline): The trained pipeline to be evaluated.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/targets.
        :return:
            Dict[str, float]: The evaluation metrics.
        """
        pass


class RegressionPipelineEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_pipeline(self, grid_search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a regression pipeline using R-squared and RSME
        :param:
            pipeline (Pipeline): The trained pipeline to be evaluated.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/targets.
        :return:
            Dict[str, float]: The evaluation metrics.
        """
        y_pred = grid_search.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        smape = smape_scorer(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "Mean Squared Error": round(mse, 4),
            "Symmetric Mean Absolute Percentage Error": round(smape, 4),
            "R-squared": round(r2, 4),
            "Best Params": grid_search.best_params_,
        }

        logger.info(f"Model Evaluation Metrics: {metrics}")
        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initialize the evaluator with the given model evaluation strategy.

        :param:
            strategy (ModelEvaluationStrategy): The model evaluation strategy to be used.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets the model evaluation strategy.
        :param:
            strategy (ModelEvaluationStrategy): Sets the new model evaluation strategy to be used.
        """
        self._strategy = strategy

    def evaluate(self, grid_search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the pipeline using the current model evaluation strategy.

        :param:
            pipeline (Pipeline): The trained pipeline to be evaluated.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/targets.
        :return:
            Dict[str, float]: The evaluation metrics.
        """
        return self._strategy.evaluate_pipeline(grid_search, X_test, y_test)