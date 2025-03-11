import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/project.log"),
        logging.StreamHandler()
    ]
)


class InsuranceAnalyzer:
    """A class for analyzing insurance data across various dimensions."""

    def export_results(self, result: pd.DataFrame, filename: str) -> None:
        """Export analysis results to a CSV file.

        Args:
            result: DataFrame containing analysis results.
            filename: Name of the file to save (e.g., 'claims_by_bank.csv').
        """
        self.logger.info(f"Exporting results to {filename}")
        result.to_csv(filename, index=False)
        self.logger.info(f"Results successfully exported to {filename}")

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize with a pandas DataFrame."""
        self.data = data
        self.logger = logging.getLogger(self.__class__.__name__)

    def claims_across_bank_and_account(self) -> pd.DataFrame:
        """Generate descriptive analysis for average claims across bank and account type.

        Returns:
            A pandas DataFrame with average claims and counts.
        """
        self.logger.info("Analyzing claims across bank and account type")
        return self.data.groupby(['Bank', 'AccountType']).agg(
            avg_claim=('TotalClaims', 'mean'),
            count=('TotalClaims', 'size')
        ).reset_index()

    def claims_across_cover_type(self) -> pd.DataFrame:
        """Generate descriptive analysis for average claims across cover type.

        Returns:
            A pandas DataFrame with average claims by cover type.
        """
        self.logger.info("Analyzing claims across cover type")
        return self.data.groupby('CoverType')['TotalClaims'].mean().reset_index()

    def claims_across_vehicle(self) -> pd.DataFrame:
        """Generate descriptive analysis for claims and premiums across vehicle type and province.

        Returns:
            A pandas DataFrame with average claims, premiums, and counts.
        """
        self.logger.info("Analyzing claims across vehicle type and province")
        return self.data.groupby(['VehicleType', 'Province']).agg(
            avg_claim=('TotalClaims', 'mean'),
            avg_premium=('TotalPremium', 'mean'),
            count=('TotalClaims', 'size')
        ).reset_index()

    def claims_by_gender_province(self) -> pd.DataFrame:
        """Generate descriptive analysis for claims and premiums across gender and province.

        Returns:
            A pandas DataFrame with average claims, premiums, and counts.
        """
        self.logger.info("Analyzing claims by gender and province")
        return self.data.groupby(['Province', 'Gender-2']).agg(
            avg_total_claim=('TotalClaims', 'mean'),
            avg_premium=('TotalPremium', 'mean'),
            count=('TotalClaims', 'size')
        ).reset_index()
