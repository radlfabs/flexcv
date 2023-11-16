import pytest
from flexcv.utilities import handle_duplicate_kwargs


def test_handle_duplicate_kwargs_no_duplicates():
    # Test handle_duplicate_kwargs function with no duplicates
    kwargs = {"param1": 1, "param2": 2, "param3": 3}
    expected = {"param1": 1, "param2": 2, "param3": 3}
    result = handle_duplicate_kwargs(kwargs)
    assert result == expected


def test_handle_duplicate_kwargs_with_duplicates():
    # Test handle_duplicate_kwargs function with duplicates
    kwargs = {"param1": 1, "param2": 2, "param3": 3, "param1": 4}
    expected = {"param1": 4, "param2": 2, "param3": 3}
    result = handle_duplicate_kwargs(kwargs)
    assert result == expected


def test_handle_duplicate_kwargs_empty():
    # Test handle_duplicate_kwargs function with empty input
    kwargs = {}
    expected = {}
    result = handle_duplicate_kwargs(kwargs)
    assert result == expected


def test_remove_duplicate_kwargs_multiple_inputs_no_duplicates():
    # Test handle_duplicate_kwargs function with multiple inputs and no duplicates
    kwargs1 = {"param1": 1, "param2": 2, "param3": 3}
    kwargs2 = {"param4": 4, "param5": 5, "param6": 6}
    expected = {
        "param1": 1,
        "param2": 2,
        "param3": 3,
        "param4": 4,
        "param5": 5,
        "param6": 6,
    }
    result = handle_duplicate_kwargs(kwargs1, kwargs2)
    assert result == expected


def test_remove_duplicate_kwargs_multiple_duplicate_inputs_no_conflicts():
    # Test handle_duplicate_kwargs function with multiple inputs
    kwargs1 = {"param1": 1, "param2": 2, "param3": 3}
    kwargs2 = {"param1": 1, "param5": 5, "param6": 6}
    expected = {"param1": 1, "param2": 2, "param3": 3, "param5": 5, "param6": 6}
    result = handle_duplicate_kwargs(kwargs1, kwargs2)
    assert result == expected


def test_remove_duplicate_kwargs_multiple_duplicate_inputs_with_conflicts():
    # Test if handle_duplicate_kwargs function raises ValueError with multiple inputs and duplicates
    kwargs1 = {"param1": 1, "param2": 2, "param3": 3}
    kwargs2 = {"param1": 4, "param2": 5, "param3": 6, "param1": 7}
    expected = {"param1": 7, "param2": 5, "param3": 6}
    with pytest.raises(ValueError):
        handle_duplicate_kwargs(kwargs1, kwargs2)
