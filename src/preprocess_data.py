import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/project.log"),
        logging.StreamHandler()
    ]
)

class DataCleaner:
    """A class for cleaning and preparing data for analysis and visualization."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize with a pandas DataFrame."""
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    def drop_multiple_columns(self, columns: list[str]) -> pd.DataFrame:
        """Drop multiple columns from the DataFrame.

        Args:
            columns: A list of column names to drop.

        Returns:
            A pandas DataFrame with specified columns dropped.
        """
        self.logger.info(f"Dropping columns: {columns}")
        try:
            return self.data.drop(columns, axis=1)
        except KeyError as e:
            self.logger.error(f"Columns not found in dataset: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unable to drop columns: {e}")
            raise

    def plot_histogram(self, column: str) -> plt.Axes:
        """Plot a histogram for a single column.

        Args:
            column: The name of the column to plot.

        Returns:
            A matplotlib Axes object with the histogram.
        """
        self.logger.info(f"Plotting histogram for column: {column}")
        ax = self.data[column].value_counts().plot(kind='hist')
        plt.title(f"{column} Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        return ax

    def plot_bar(self, column: str) -> plt.Axes:
        """Plot a bar chart for a column.

        Args:
            column: The name of the column to plot.

        Returns:
            A matplotlib Axes object with the bar chart.
        """
        self.logger.info(f"Plotting bar chart for column: {column}")
        ax = self.data[column].value_counts().plot(
            kind='barh', title=f"{column} Frequency", xlabel="Count", ylabel=f"{column}s"
        )
        return ax

    def fill_na_with_mean(self, column: str) -> pd.Series:
        """Fill missing values in a column with the column's mean.

        Args:
            column: The name of the column to fill.

        Returns:
            A pandas Series with missing values filled.
        """
        self.logger.info(f"Filling NA values in {column} with mean")
        try:
            return self.data[column].fillna(self.data[column].mean())
        except Exception as e:
            self.logger.error(f"Error filling NA values in {column}: {e}")
            raise

    def fill_na_with_value(self, column: str, value: any) -> pd.Series:
        """Fill missing values in a column with a specific value.

        Args:
            column: The name of the column to fill.
            value: The value to fill missing entries with.

        Returns:
            A pandas Series with missing values filled.
        """
        self.logger.info(f"Filling NA values in {column} with {value}")
        try:
            return self.data[column].fillna(value)
        except KeyError as e:
            self.logger.error(f"Column {column} not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error filling NA values in {column}: {e}")
            raise