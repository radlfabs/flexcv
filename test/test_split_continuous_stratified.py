from flexcv.stratification import ContinuousStratifiedKFold
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

def test_continuous_stratified_kfold_init():
    # Test initialization
    cv = ContinuousStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    assert isinstance(cv, ContinuousStratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle == True
    assert cv.random_state == 42

def test_continuous_stratified_kfold_split():
    # Test split method
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    cv = ContinuousStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0

def test_continuous_stratified_kfold_split_with_series():
    # Test split method with y as a Series
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    y = pd.Series(y)
    cv = ContinuousStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0

def test_continuous_stratified_kfold_get_n_splits():
    # Test get_n_splits method
    cv = ContinuousStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    assert cv.get_n_splits() == 5