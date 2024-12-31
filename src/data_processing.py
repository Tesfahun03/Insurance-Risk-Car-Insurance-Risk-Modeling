import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessing:
    def __init__(self, data):
        self.data = data

    def show_correlation_matrix(self,  target):
        """show the correaltion of each numeric feature with the target

        Args:
            data (_pd.dataFrame_): _a pandas dataframe with features and target_
            target (_pd.series_): _target column of the dataframe to pe predicted_

        Returns:
            _correlation_matrix_: _a matrix that shows how strong is a feature correlate to the target_
        """
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_data.corr()
        correlation_matrix_with_target = correlation_matrix[target]

        return correlation_matrix_with_target

    def encode_data(self):
        """_encodes catagorical coloumns into sum randomly assigned numbers for regression purpose_

        Returns:
            _DataFrame_: _encoded dataframe_
        """
        columns_label = self.data.select_dtypes(
            include=['object', 'bool']).columns
        df_lbl = self.data.copy()
        for col in columns_label:
            label = LabelEncoder()
            label.fit(list(self.data[col].values))
            df_lbl[col] = label.transform(df_lbl[col].values)

        return df_lbl

    def standardize_data(self, dataframe):
        """_standrardize the dataset columns_

        Args:
            dataframe (_Pd.DataFrame_): _pandas dataframe_

        Returns:
            _Pd.Dataframe_: _standardize dataframe_
        """
        column_scaler = dataframe.select_dtypes(
            include=['object', 'float64', 'int64']).columns
        df_standard = dataframe.copy()
        standard = StandardScaler()
        for col in column_scaler:
            df_standard[column_scaler] = standard.fit_transform(
                df_standard[column_scaler])
        return df_standard
