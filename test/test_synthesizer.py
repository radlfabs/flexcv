import pandas as pd
import numpy as np
from flexcv.synthesizer import generate_regression

def test_generate_regression():
    # Test generate_regression function
    m_features = 5
    n_samples = 100
    n_groups = 3
    n_slopes = 2
    random_seed = 42
    noise_level = 0.1
    fixed_random_ratio = 0.01

    X, y, group, random_slopes = generate_regression(m_features, n_samples, n_groups, n_slopes, random_seed, noise_level, fixed_random_ratio)

    # Check that the returned X, y, group, and random_slopes are of the correct types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(group, pd.Series)
    assert isinstance(random_slopes, pd.DataFrame)

    # Check that the returned X, y, group, and random_slopes have the correct shapes
    assert X.shape == (n_samples, m_features)
    assert y.shape == (n_samples,)
    assert group.shape == (n_samples,)
    assert random_slopes.shape == (n_samples, n_slopes)

    # Check that the returned group contains the correct number of unique values
    assert group.nunique() == n_groups

    # Check that the returned random_slopes contains the correct columns
    assert all(column in X.columns for column in random_slopes.columns)