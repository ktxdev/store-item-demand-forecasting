import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def smape_scorer(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: SMAPE value as a percentage.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_value = np.mean(numerator / denominator) * 100
    return smape_value


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        """
        Abstract method for building and training a model.

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            RegressorMixin: The trained model instance
        """
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        """
        Builds and trains a linear regression model with hyperparameter tuning

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            Pipeline: A pipeline with a trained regression model instance
        """

        logger.info("Initializing regression model")

        param_grid = {
            'fit_intercept': [True, False],
            'n_jobs': [1, -1]
        }

        grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid,
                                   scoring=make_scorer(smape_scorer, greater_is_better=False), n_jobs=-1, cv=5)

        logger.info("Training Linear Regression model")
        grid_search.fit(X_train, y_train)

        return grid_search


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the Model builder with a specified model building strategy.

        :param:
            strategy (ModelBuildingStrategy): The strategy to use for building the model
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        :param:
            strategy (ModelBuildingStrategy): The new strategy to use for building the model
        """
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        """
        Executes the model building and training using the current strategy.

        :param:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/targets
        :return:
            Pipeline: A pipeline with a trained regression model instance
        """
        logger.info("Building and training model using the set strategy")
        return self._strategy.build_and_train_model(X_train, y_train)
