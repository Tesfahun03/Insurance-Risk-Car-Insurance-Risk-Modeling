import pytest
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess_data import DataCleaner


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for cleaning."""
    data = pd.DataFrame({
        'A': [1, 2, None],
        'B': ['x', 'y', 'z'],
        'C': [10, 20, 30]
    })
    return data


def test_drop_multiple_columns(sample_data):
    cleaner = DataCleaner(sample_data)
    result = cleaner.drop_multiple_columns(['B'])
    assert isinstance(result, pd.DataFrame)
    assert 'B' not in result.columns


def test_plot_histogram(sample_data):
    cleaner = DataCleaner(sample_data)
    ax = cleaner.plot_histogram('C')
    assert ax.get_title() == 'C Histogram'
    plt.close()


def test_plot_bar(sample_data):
    cleaner = DataCleaner(sample_data)
    ax = cleaner.plot_bar('B')
    assert ax.get_title() == 'B Frequency'
    plt.close()


def test_fill_na_with_mean(sample_data):
    cleaner = DataCleaner(sample_data)
    result = cleaner.fill_na_with_mean('A')
    assert isinstance(result, pd.Series)
    assert result.isna().sum() == 0
    assert result[2] == pytest.approx(1.5)  # Mean of [1, 2]


def test_fill_na_with_value(sample_data):
    cleaner = DataCleaner(sample_data)
    result = cleaner.fill_na_with_value('A', 0)
    assert isinstance(result, pd.Series)
    assert result[2] == 0
