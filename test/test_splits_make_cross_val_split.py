from flexcv.split import make_cross_val_split, CrossValMethod
from sklearn.model_selection import KFold, GroupKFold
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

def test_make_cross_val_split_kfold():
    # Test make_cross_val_split function with KFold
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    groups = pd.Series(np.random.choice([1, 2, 3, 4, 5], size=100))
    cv = make_cross_val_split(method=CrossValMethod.KFOLD, n_splits=5, random_state=42, groups=groups)
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0

def test_make_cross_val_split_group():
    # Test make_cross_val_split function with GroupKFold
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)
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
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    groups = np.random.choice([1, 2, 3], size=100)
    cv = make_cross_val_split(method=CrossValMethod.STRATGROUP, n_splits=5, groups=groups, random_state=42)
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0

def test_make_cross_val_split_customstratgroup():
    # Test make_cross_val_split function with CustomStratGroupKFold
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    groups = np.random.choice([1, 2, 3, 4, 5], size=100)
    cv = make_cross_val_split(method=CrossValMethod.CUSTOMSTRATGROUP, n_splits=5, groups=groups, random_state=42)
    splits = list(cv(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0

def test_make_cross_val_split_customstrat():
    # Test make_cross_val_split function with CustomStratifiedKFold
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    groups = np.random.choice([1, 2, 3], size=100)
    cv = make_cross_val_split(method=CrossValMethod.CUSTOMSTRAT, n_splits=5, groups=groups, random_state=42)
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