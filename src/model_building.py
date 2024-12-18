import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

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
                                   scoring=make_scorer(smape_scorer, greater_is_better=False), n_jobs=-1, cv=5,
                                   verbose=3)

        logger.info("Training Linear Regression model")
        grid_search.fit(X_train, y_train)

        return grid_search


class RandomForestRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        logger.info("Initializing random forest model")

        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid,
                                   scoring=make_scorer(smape_scorer, greater_is_better=False), n_jobs=6, cv=5,
                                   verbose=3)
        logger.info("Training Random Forest Regression model")
        grid_search.fit(X_train, y_train)

        return grid_search


class SupportVectorRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        logger.info("Initializing Gradient Boosting Regression model")

        # Define the parameter grid
        param_grid = {
            'C': [900, 1000, 1200],
            'epsilon': [0.1, 1, 2, 3],
            'kernel': ['linear', 'rbf']
        }

        grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid,
                                   scoring=make_scorer(smape_scorer, greater_is_better=False), n_jobs=6, cv=5,
                                   verbose=3)
        logger.info("Training Gradient Boosting Regression model")
        grid_search.fit(X_train, y_train)

        return grid_search


class GradientBoostingRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        logger.info("Initializing Gradient Boosting Regression model")

        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }

        grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=param_grid,
                                   scoring=make_scorer(smape_scorer, greater_is_better=False), n_jobs=6, cv=5,
                                   verbose=3)
        logger.info("Training Gradient Boosting Regression model")
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
