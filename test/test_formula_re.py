import pandas as pd
import pytest
from flexcv.utilities import get_re_formula


def test_get_re_formula_none():
    # Test with None
    result = get_re_formula(None)
    assert result == ""


def test_get_re_formula_dataframe():
    # Test with a DataFrame
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
    result = get_re_formula(df)
    assert result == "~ column1 + column2"


def test_get_re_formula_series():
    # Test with a Series
    series = pd.Series([1, 2, 3], name="column1")
    result = get_re_formula(series)
    assert result == "~ column1"


def test_get_re_formula_invalid_type():
    # Test with an invalid type
    with pytest.raises(TypeError):
        get_re_formula("invalid type")
