import pytest

from flexcv.utilities import rm_model_from_keys


def test_rm_model_from_keys_empty():
    # Test with an empty dictionary
    result = rm_model_from_keys({})
    assert result == {}


def test_rm_model_from_keys_single():
    # Test with a single-item dictionary
    result = rm_model_from_keys({"model__param": "value"})
    assert result == {"param": "value"}


def test_rm_model_from_keys_multiple():
    # Test with a multi-item dictionary
    result = rm_model_from_keys({"model__param1": "value1", "model__param2": "value2"})
    assert result == {"param1": "value1", "param2": "value2"}


def test_rm_model_from_keys_no_model_prefix():
    # Test with a dictionary that doesn't have 'model__' prefix
    result = rm_model_from_keys({"param": "value"})
    assert result == {"param": "value"}


def test_rm_model_from_keys_mixed():
    # Test with a dictionary that has some keys with 'model__' prefix and some without
    result = rm_model_from_keys({"model__param1": "value1", "param2": "value2"})
    assert result == {"param1": "value1", "param2": "value2"}
