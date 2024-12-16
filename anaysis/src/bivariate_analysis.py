from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Abstract method to perform bivariate analysis on the dataframe.
        :param:
            data (pd.DataFrame): Dataframe containing the data to be analyzed.
            feature1 (str): Name of the 1st feature to be analyzed.
            feature2 (str): Name of the 2nd feature to be analyzed.
        :return:
            None: Plots a bar plot for the given features
        """
        pass

    def _setup_plot(self, feature1: str, feature2: str) -> None:
        plt.figure(figsize=(15, 6))
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f"{feature1} mean for {feature2}")
        plt.xticks(rotation=90)


class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, data: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Performs bivariate analysis on the dataframe for categorical vs numerical features.
        :param:
            data (pd.DataFrame): Dataframe containing the data to be analyzed.
            feature1 (str): The categorical feature to be analyzed.
            feature2 (str): The numerical feature to be analyzed.
        :return:
        """
        self._setup_plot(feature1, feature2)
        # Summarize data frame by the categorical feature
        summary_df = data.groupby(feature1)[feature2].mean().reset_index()
        feature2 = f"Mean {feature2}"
        summary_df.columns = [feature1, feature2]
        # Plot the barplot
        sns.barplot(x=summary_df[feature1].astype(str), y=summary_df[feature2])
        plt.show()


class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes a Bivariate Analyzer object with a specified strategy.
        :param:
            strategy (BivariateAnalysisStrategy): The strategy to be used for Bivariate analysis.
        """

        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets the strategy to be used for Bivariate analysis.
        :param:
            strategy (BivariateAnalysisStrategy): The new strategy to be used for Bivariate analysis.
        """
        self._strategy = strategy

    def analyze(self, data: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Performs bivariate analysis on the dataframe for the given features with the specified strategy.
        :param:
            data (pd.DataFrame): Dataframe containing the data to be analyzed.
            feature1 (str): The categorical feature to be analyzed.
            feature2 (str): The numerical feature to be analyzed.
        """
        self._strategy.analyze(data, feature1, feature2)
