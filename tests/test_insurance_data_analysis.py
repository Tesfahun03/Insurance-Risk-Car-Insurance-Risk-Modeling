import pytest
import pandas as pd
from src.insurance_data_analysis import InsuranceAnalyzer


@pytest.fixture
def sample_insurance_data():
    """Fixture to provide sample insurance data."""
    data = pd.DataFrame({
        'Bank': ['B1', 'B2', 'B1'],
        'AccountType': ['A1', 'A2', 'A1'],
        'CoverType': ['C1', 'C2', 'C1'],
        'VehicleType': ['V1', 'V2', 'V1'],
        'Province': ['P1', 'P2', 'P1'],
        'Gender-2': ['M', 'F', 'M'],
        'TotalClaims': [100, 200, 150],
        'TotalPremium': [50, 60, 55]
    })
    return data


def test_claims_across_bank_and_account(sample_insurance_data):
    analyzer = InsuranceAnalyzer(sample_insurance_data)
    result = analyzer.claims_across_bank_and_account()
    assert isinstance(result, pd.DataFrame)
    assert 'avg_claim' in result.columns
    assert len(result) == 2  # Two unique Bank-AccountType combos


def test_claims_across_cover_type(sample_insurance_data):
    analyzer = InsuranceAnalyzer(sample_insurance_data)
    result = analyzer.claims_across_cover_type()
    assert isinstance(result, pd.DataFrame)
    assert 'TotalClaims' in result.columns
    assert len(result) == 2  # Two unique CoverTypes


def test_claims_across_vehicle(sample_insurance_data):
    analyzer = InsuranceAnalyzer(sample_insurance_data)
    result = analyzer.claims_across_vehicle()
    assert isinstance(result, pd.DataFrame)
    assert 'avg_claim' in result.columns
    assert 'avg_premium' in result.columns


def test_claims_by_gender_province(sample_insurance_data):
    analyzer = InsuranceAnalyzer(sample_insurance_data)
    result = analyzer.claims_by_gender_province()
    assert isinstance(result, pd.DataFrame)
    assert 'avg_total_claim' in result.columns
    assert 'avg_premium' in result.columns
