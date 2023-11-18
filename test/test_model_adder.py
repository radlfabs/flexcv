import pytest

from flexcv.utilities import add_model_to_keys


def test_add_model_to_keys_empty():
    # Test with an empty dictionary
    result = add_model_to_keys({})
    assert result == {}


def test_add_model_to_keys_single():
    # Test with a single-item dictionary
    result = add_model_to_keys({"param": "value"})
    assert result == {"model__param": "value"}


def test_add_model_to_keys_multiple():
    # Test with a multi-item dictionary
    result = add_model_to_keys({"param1": "value1", "param2": "value2"})
    assert result == {"model__param1": "value1", "model__param2": "value2"}


def test_add_model_to_keys_nested():
    # Test with a nested dictionary
    result = add_model_to_keys({"param": {"nested_param": "nested_value"}})
    assert result == {"model__param": {"nested_param": "nested_value"}}
