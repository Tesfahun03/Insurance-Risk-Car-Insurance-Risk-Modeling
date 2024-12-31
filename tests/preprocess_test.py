from src.preprocess_data import CleanData
import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))


class TestCleanData(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, np.nan],
            'B': [5, 6, np.nan, 8, 9],
            'C': ['cat', 'dog', 'cat', 'mouse', 'dog'],
            'D': [np.nan, np.nan, np.nan, 10, 11]
        })
        self.cleaner = CleanData(self.data)

    def test_drop_multi_column(self):
        """Test dropping multiple columns."""
        result = self.cleaner.drop_multi_column(['A', 'B'])
        self.assertNotIn('A', result.columns)
        self.assertNotIn('B', result.columns)

    def test_plot_hist(self):
        """Test plotting a histogram for a column."""
        try:
            plot = self.cleaner.plot_hist('A')
            self.assertIsNotNone(plot)
        except Exception as e:
            self.fail(f"plot_hist method failed with exception: {e}")

    def test_plot_bar(self):
        """Test plotting a bar chart for a column."""
        try:
            plot = self.cleaner.plot_bar('C')
            self.assertIsNotNone(plot)
        except Exception as e:
            self.fail(f"plot_bar method failed with exception: {e}")

    def test_fill_na_mean(self):
        """Test filling missing values with the mean of the column."""
        result = self.cleaner.fill_na_mean('A')
        self.assertAlmostEqual(result[4], self.data['A'].mean(skipna=True))

    def test_fillna(self):
        """Test filling missing values with a specific value."""
        result = self.cleaner.fillna('D', 0)
        self.assertEqual(result.isna().sum(), 0)
        self.assertEqual(result.iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
