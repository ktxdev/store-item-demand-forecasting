import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from abc import ABC, abstractmethod


class SeasonalTrendAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Abstract method to analyze the seasonal trend
        :param:
            df (pd.DataFrame): The dataframe to analyze seasonal trend
            feature (str): The date feature to analyze seasonal trend
        :return:
            None: Plots a bar graph of the seasonal trend
        """
        pass


class DateFeaturesSeasonalTrendAnalysis(SeasonalTrendAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        monthly_avg_sales = df.groupby(['store', 'item', feature])["sales"].mean().reset_index()

        stores = monthly_avg_sales['store'].unique()
        num_stores = len(stores)
        fig, axes = plt.subplots(num_stores, 1, figsize=(20, 8 * num_stores), sharey=True)

        for i, store in enumerate(stores):
            store_data = monthly_avg_sales[monthly_avg_sales["store"] == store]
            sns.barplot(
                x=feature,
                y="sales",
                data=store_data,
                ax=axes[i] if num_stores > 1 else axes
            )
            axes[i % num_stores].set_title(f"Sales AVG for Store {store} per {feature}")
            axes[i % num_stores].set_ylabel("Average Sales")
            axes[i % num_stores].set_xlabel(feature)

        plt.tight_layout()
        plt.show()


class SeasonalTrendAnalyzer:
    def __init__(self, strategy: SeasonalTrendAnalysisStrategy):
        """
        Initializes a SeasonalTrendAnalyzer object with the given strategy.
        :param:
            strategy (SeasonalTrendAnalysisStrategy): The strategy to use for seasonal trend analysis
        """
        self._strategy = strategy

    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Uses the strategy to analyze the seasonal trend
        :param:
            df (pd.DataFrame): The dataframe to analyze seasonal trend
            feature (str): The date feature to analyze seasonal trend :
        :return:
            None: Plots a bar graph of the seasonal trend
        """
        self._strategy.analyze(df, feature)
