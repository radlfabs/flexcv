import pandas as pd
import pytest
from flexcv.utilities import get_fixed_effects_formula

def test_get_fixed_effects_formula_single_column():
    # Test with a single-column DataFrame
    df = pd.DataFrame({'column1': [1, 2, 3]})
    result = get_fixed_effects_formula('target', df)
    assert result == 'target ~ column1'

def test_get_fixed_effects_formula_two_columns():
    # Test with a two-column DataFrame
    df = pd.DataFrame({'column1': [1, 2, 3], 'column2': [4, 5, 6]})
    result = get_fixed_effects_formula('target', df)
    assert result == 'target ~ column1 + column2'

def test_get_fixed_effects_formula_multiple_columns():
    # Test with a multi-column DataFrame
    df = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': [4, 5, 6],
        'column3': [7, 8, 9],
        'column4': [10, 11, 12]
    })
    result = get_fixed_effects_formula('target', df)
    assert result == 'target ~ column1 + column2 + column3 + column4'

def test_get_fixed_effects_formula_no_columns():
    # Test with an empty DataFrame
    df = pd.DataFrame()
    with pytest.raises(IndexError):
        get_fixed_effects_formula('target', df)