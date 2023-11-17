import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold, KFold

from flexcv.split import CrossValMethod, make_cross_val_split


def test_make_cross_val_split_kfold():
    # Test make_cross_val_split function with KFold
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    groups = pd.Series(np.random.choice([1, 2, 3, 4, 5], size=100))
    cv = make_cross_val_split(
        method=CrossValMethod.KFOLD, n_splits=5, random_state=42, groups=groups
    )
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0


def test_make_cross_val_split_group():
    # Test make_cross_val_split function with GroupKFold
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    groups = np.random.choice([1, 2, 3, 4, 5], size=100)
    cv = make_cross_val_split(method=CrossValMethod.GROUP, n_splits=5, groups=groups)
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0


def test_make_cross_val_split_stratgroup():
    # Test make_cross_val_split function with StratGroupKFold
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    groups = np.random.choice([1, 2, 3, 4, 5], size=100)
    cv = make_cross_val_split(
        method=CrossValMethod.STRATGROUP, n_splits=5, groups=groups, random_state=42
    )
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0


def test_make_cross_val_split_contistrat():
    # Test make_cross_val_split function with CustomStratGroupKFold
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    groups = np.random.choice([1, 2, 3, 4, 5], size=100)
    cv = make_cross_val_split(
        method=CrossValMethod.CONTISTRAT, n_splits=5, groups=groups, random_state=42
    )
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0


def test_make_cross_val_split_contigroup():
    # Test make_cross_val_split function with CustomStratifiedKFold
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    groups = np.random.choice([1, 2, 3, 4, 5], size=100)
    cv = make_cross_val_split(
        method=CrossValMethod.CONTISTRATGROUP,
        n_splits=5,
        groups=groups,
        random_state=42,
    )
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0


def test_make_cross_val_split_contigroup():
    # Test make_cross_val_split function with CustomStratifiedKFold
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    groups = np.random.choice([1, 2, 3, 4, 5], size=100)
    cv = make_cross_val_split(
        method=CrossValMethod.CONTISTRATGROUP,
        n_splits=5,
        groups=groups,
        random_state=42,
    )
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0


def test_make_cross_val_split_invalid_method():
    # Test make_cross_val_split function with an invalid method
    try:
        make_cross_val_split(method="invalid", n_splits=5, random_state=42)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from flexcv.split import CrossValMethod, make_cross_val_split


def test_make_cross_val_split_group_is_callable():
    # Test GroupKFold
    groups = pd.Series(np.random.choice([1, 2, 3], size=100))
    split_func = make_cross_val_split(groups=groups, method=CrossValMethod.GROUP)
    assert callable(split_func)


def test_make_cross_val_split_invalid_method_raises():
    # Test invalid method
    with pytest.raises(TypeError):
        make_cross_val_split(groups=None, method="InvalidMethod")


def test_make_cross_val_split_with_iterator():
    # Test with iterator
    iterator = iter([np.array([1, 2, 3]), np.array([4, 5, 6])])
    split_func = make_cross_val_split(groups=None, method=iterator)
    assert split_func == iterator


def test_make_cross_val_split_with_cross_validator():
    # Test with cross validator
    kfold = KFold(n_splits=5)
    split_func = make_cross_val_split(groups=None, method=kfold)
    assert split_func == kfold.split


def test_make_cross_val_split_with_groups_consumer():
    # Test with groups consumer
    gkf = GroupKFold(n_splits=5)
    groups = pd.Series(np.random.choice([1, 2, 3], size=100))
    split_func = make_cross_val_split(groups=groups, method=gkf)
    assert callable(split_func)
