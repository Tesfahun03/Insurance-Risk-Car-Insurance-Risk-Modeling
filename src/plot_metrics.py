import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/project.log"),
        logging.StreamHandler()
    ]
)


class MetricsVisualizer:
    """A class for plotting model accuracy metrics."""

    def __init__(self, models: List[str], mae_scores: List[float], mse_scores: List[float], r2_scores: List[float]) -> None:
        """Initialize with model names and their metrics."""
        self.models = models
        self.mae_scores = mae_scores
        self.mse_scores = mse_scores
        self.r2_scores = r2_scores
        self.logger = logging.getLogger(self.__class__.__name__)

    def plot_metrics(self) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
        """Plot accuracy metrics (MAE, MSE, R2) for each model.

        Returns:
            A tuple of matplotlib Axes objects for MAE, MSE, and R2 plots.
        """
        self.logger.info("Plotting model metrics")

        # MAE plot
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(self.models, self.mae_scores, color='green')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Comparison of MAE Scores')
        plt.xticks(rotation=45)

        # MSE plot
        fig, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(self.models, self.mse_scores, color='yellow')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Comparison of MSE Scores')
        plt.xticks(rotation=45)

        # R2 plot
        fig, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(self.models, self.r2_scores, color='red')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('R2 Scores')
        ax3.set_title('Comparison of R2 Scores')
        plt.xticks(rotation=45)

        plt.tight_layout()
        return ax1, ax2, ax3
