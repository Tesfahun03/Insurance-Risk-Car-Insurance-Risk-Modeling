import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CleanData:
    """Used in cleaning the data( droping columns, rows, filling NA values, droping NA values)
       preparing the data for visualization and caluculation(converting datatypes of a specific column...)

       arg:
            data -> pandas DataFrame
    """

    def __init__(self, data):
        self.data = data

    def drop_multi_column(self, columns):
        """droping multiple columns

        Args:
            columns (Pandas Series): a list of pandas series(columns)

        Returns:
            DataFrame: Pandas dataframe
        """
        try:
            return self.data.drop(columns, axis=1)

        except KeyError as e:
            print(f'The columns mantioned does not exist in the dataset {e}')

        except Exception as e:
            print(f'Unable to drop columns: {e}')


# df['Bank'].value_counts().plot(kind = 'barh')

    def plot_hist(self, column):
        """plot histogram for single series

        Args:
            column (pandas series): 

        Returns:
            matplotlib plot: 
        """
        return self.data[column].value_counts().plot(kind='hist')

    def plot_bar(self, column):
        """plot a bar chart for a column

        Args:
            column (Series): pandas series

        Returns:
            plot:
        """
        return self.data[column].value_counts().plot(kind='barh', title=f'{column} frquency ', xlabel='Count', ylabel=f'{column}s')

    def fill_na_mean(self, column):
        """method to fill the missing value of a series with mean of that column

        Args:
            column (pd.series): 
        """
        try:
            return self.data[column].fillna(
                self.data[column].mean())

        except Exception as e:
            print(f'an error occured during filling values {e}')

    def fillna(self, column, value):
        """method to fill missing value of a column with specific value

        Args:
            column (pd series): pandas dataframe or series
            value (any): value to fill the missing values
        Returns:
            DataFrame: Pandas dataframe
        """
        try:
            return self.data[column].fillna(value)
        except KeyError as e:
            print(f'{column} name does not exist in df: {e}')
        except Exception as e:
            print(f'Error occured {e}')
