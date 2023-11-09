from flexcv.split import CustomStratifiedKFold
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

def test_custom_stratified_kfold_init():
    # Test initialization
    cv = CustomStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    assert isinstance(cv, CustomStratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle == True
    assert cv.random_state == 42

def test_custom_stratified_kfold_split():
    # Test split method
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    groups = np.random.choice([1, 2, 3], size=100)
    cv = CustomStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X, y, groups))
    # Check that the correct number of splits are returned
    assert len(splits) == 5
    # Check that the training and test sets are disjoint
    for train_index, test_index in splits:
        assert len(set(train_index) & set(test_index)) == 0

def test_custom_stratified_kfold_get_n_splits():
    # Test get_n_splits method
    cv = CustomStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = np.random.rand(100, 20)
    assert cv.get_n_splits(X) == 5