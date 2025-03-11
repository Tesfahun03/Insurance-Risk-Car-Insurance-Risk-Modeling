import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from typing import Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/project.log"),
        logging.StreamHandler()
    ]
)


class DataSplitter:
    """A class for splitting data into training and testing sets."""

    def __init__(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Initialize with features and target."""
        self.x = x
        self.y = y
        self.logger = logging.getLogger(self.__class__.__name__)

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into 80% training and 20% testing sets.

        Returns:
            A tuple of (x_train, x_test, y_train, y_test).
        """
        self.logger.info("Splitting data into train and test sets")
        return train_test_split(self.x, self.y, test_size=0.2, random_state=42)


class ModelTrainer:
    """A class for training multiple machine learning models."""

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Initialize with training data."""
        self.x_train = x_train
        self.y_train = y_train
        self.logger = logging.getLogger(self.__class__.__name__)

    def linear_regression(self) -> LinearRegression:
        """Train a Linear Regression model.

        Returns:
            A fitted LinearRegression model.
        """
        self.logger.info("Training Linear Regression model")
        model = LinearRegression()
        model.fit(self.x_train, self.y_train)
        return model

    def decision_tree_regressor(self) -> DecisionTreeRegressor:
        """Train a Decision Tree Regressor model.

        Returns:
            A fitted DecisionTreeRegressor model.
        """
        self.logger.info("Training Decision Tree Regressor model")
        model = DecisionTreeRegressor(random_state=42)
        model.fit(self.x_train, self.y_train)
        return model

    def random_forest(self) -> RandomForestRegressor:
        """Train a Random Forest Regressor model.

        Returns:
            A fitted RandomForestRegressor model.
        """
        self.logger.info("Training Random Forest Regressor model")
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        model.fit(self.x_train, self.y_train)
        return model

    def xgboost(self) -> XGBRegressor:
        """Train an XGBoost Regressor model.

        Returns:
            A fitted XGBRegressor model.
        """
        self.logger.info("Training XGBoost Regressor model")
        model = XGBRegressor(random_state=42)
        model.fit(self.x_train, self.y_train)
        return model

    def get_feature_importance(self, model) -> pd.Series:
        """Extract feature importance from a trained model.

        Args:
            model: A fitted tree-based model (e.g., RandomForestRegressor, XGBRegressor).

        Returns:
            A pandas Series with feature names and their importance scores.
        """
        self.logger.info(
            f"Extracting feature importance from {model.__class__.__name__}")
        importance = pd.Series(model.feature_importances_,
                               index=self.x_train.columns)
        return importance.sort_values(ascending=False)

    def tune_random_forest(self, param_grid: dict = None) -> RandomForestRegressor:
        """Tune Random Forest hyperparameters using GridSearchCV.

        Args:
            param_grid: Dictionary of parameters to tune (default provided if None).

        Returns:
            A fitted RandomForestRegressor with the best parameters.
        """
        self.logger.info("Tuning Random Forest model")
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.x_train, self.y_train)
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_


class ModelEvaluator:
    """A class for evaluating machine learning model performance."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_model(self, model, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float, np.ndarray]:
        """Evaluate a model using accuracy metrics.

        Args:
            model: A trained regression model.
            x_test: Testing features.
            y_test: Testing target.

        Returns:
            A tuple of (mae, mse, r2, y_pred).
        """
        self.logger.info(f"Evaluating model: {model.__class__.__name__}")
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.logger.info(f"MAE: {mae}, MSE: {mse}, R2: {r2}")
        return mae, mse, r2, y_pred
