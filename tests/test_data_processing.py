import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z'],
        'C': [True, False, True]
    })
    return data

def test_show_correlation_matrix(sample_data):
    processor = DataProcessor(sample_data)
    corr = processor.show_correlation_matrix('A')
    assert isinstance(corr, pd.Series)
    assert 'A' in corr.index
    assert corr['A'] == 1.0  # Correlation with itself is 1

def test_encode_data(sample_data):
    processor = DataProcessor(sample_data)
    encoded = processor.encode_data()
    assert isinstance(encoded, pd.DataFrame)
    assert encoded['B'].dtype == 'int64'  # Categorical column encoded
    assert encoded['C'].dtype == 'int64'  # Boolean column encoded

def test_standardize_data(sample_data):
    processor = DataProcessor(sample_data)
    standardized = processor.standardize_data()
    assert isinstance(standardized, pd.DataFrame)
    assert np.allclose(standardized['A'].mean(), 0, atol=1e-8)  # Mean ~ 0 after standardization
    assert np.allclose(standardized['A'].std(), 1, atol=1e-8)   # Std ~ 1 after standardization