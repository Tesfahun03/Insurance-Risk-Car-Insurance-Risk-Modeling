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
            return self.data.drop(columns, axis=1, inplace=True)

        except KeyError as e:
            print(f'The columns mantioned does not exist in the dataset {e}')

        except Exception as e:
            print(f'Unable to drop columns: {e}')


# df['Bank'].value_counts().plot(kind = 'barh')

    def plot_bar(self, column):
        """plot a bar chart for a column

        Args:
            column (Series): pandas series

        Returns:
            plot:
        """
        return self.data[column].value_counts().plot(kind='barh', title=f'{column} frquency ', xlabel='Count', ylabel=f'{column}s')
