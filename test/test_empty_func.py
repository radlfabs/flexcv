import pytest
from flexcv.utilities import empty_func

def test_empty_func_no_args():
    # Test with no arguments
    args, kwargs = empty_func()
    assert args == ()
    assert kwargs == {}

def test_empty_func_with_args():
    # Test with positional arguments
    args, kwargs = empty_func(1, 2, 3)
    assert args == (1, 2, 3)
    assert kwargs == {}

def test_empty_func_with_kwargs():
    # Test with keyword arguments
    args, kwargs = empty_func(a=1, b=2, c=3)
    assert args == ()
    assert kwargs == {'a': 1, 'b': 2, 'c': 3}

def test_empty_func_with_args_and_kwargs():
    # Test with both positional and keyword arguments
    args, kwargs = empty_func(1, 2, 3, a=1, b=2, c=3)
    assert args == (1, 2, 3)
    assert kwargs == {'a': 1, 'b': 2, 'c': 3}