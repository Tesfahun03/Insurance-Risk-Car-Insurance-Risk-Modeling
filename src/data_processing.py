import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/project.log"),
        logging.StreamHandler()
    ]
)


class DataProcessor:
    """A class for processing data including encoding and standardization."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize with a pandas DataFrame."""
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    def show_correlation_matrix(self, target: str) -> pd.Series:
        """Calculate and return the correlation of numeric features with the target.

        Args:
            target: The name of the target column to predict.

        Returns:
            A pandas Series with correlation values for each numeric feature.
        """
        self.logger.info(
            f"Calculating correlation matrix with target: {target}")
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        if target not in numeric_data.columns:
            self.logger.error(
                f"Target '{target}' not found in numeric columns")
            raise ValueError(f"Target '{target}' must be numeric")
        correlation_matrix = numeric_data.corr()
        return correlation_matrix[target]

    def encode_data(self) -> pd.DataFrame:
        """Encode categorical columns into numeric values for regression.

        Returns:
            A pandas DataFrame with encoded categorical columns.
        """
        self.logger.info("Encoding categorical columns")
        categorical_cols = self.data.select_dtypes(
            include=['object', 'bool']).columns
        df_encoded = self.data.copy()
        for col in categorical_cols:
            self.logger.debug(f"Encoding column: {col}")
            label_encoder = LabelEncoder()
            df_encoded[col] = label_encoder.fit_transform(
                df_encoded[col].values)
        return df_encoded

    def standardize_data(self) -> pd.DataFrame:
        """Standardize numeric columns in the dataset.

        Returns:
            A pandas DataFrame with standardized numeric columns.
        """
        self.logger.info("Standardizing numeric columns")
        numeric_cols = self.data.select_dtypes(
            include=['float64', 'int64']).columns
        df_standard = self.data.copy()
        scaler = StandardScaler()
        df_standard[numeric_cols] = scaler.fit_transform(
            df_standard[numeric_cols])
        return df_standard
