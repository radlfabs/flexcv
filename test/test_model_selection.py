import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from flexcv.model_selection import ObjectiveScorer
from flexcv.model_selection import custom_scorer
from flexcv.model_selection import objective, ObjectiveScorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from flexcv.model_selection import parallel_objective, ObjectiveScorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from flexcv.model_selection import objective_cv, ObjectiveScorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from unittest.mock import MagicMock
import numpy as np
import pandas as pd


def test_objective_scorer_init_valid_scorer():
    # Test initialization with valid scorer
    def valid_scorer(y_valid, y_pred, y_train_in, y_pred_train):
        return np.mean(y_valid - y_pred)

    scorer = ObjectiveScorer(valid_scorer)
    assert isinstance(scorer, ObjectiveScorer)
    assert scorer.scorer == valid_scorer


def test_objective_scorer_init_invalid_scorer():
    # Test initialization with invalid scorer
    def invalid_scorer(y_valid, y_pred, y_train_in):
        return np.mean(y_valid - y_pred)

    with pytest.raises(ValueError):
        ObjectiveScorer(invalid_scorer)


def test_objective_scorer_call():
    # Test __call__ method
    def valid_scorer(y_valid, y_pred, y_train_in, y_pred_train):
        return np.mean(y_valid - y_pred)

    scorer = ObjectiveScorer(valid_scorer)
    y_valid = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    y_train_in = np.array([1, 2, 3])
    y_pred_train = np.array([1, 2, 3])
    result = scorer(y_valid, y_pred, y_train_in, y_pred_train)
    assert result == 0


def test_custom_scorer_same_values():
    # Test custom_scorer function with same values
    y_valid = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    y_train_in = np.array([1, 2, 3])
    y_pred_train = np.array([1, 2, 3])
    result = custom_scorer(y_valid, y_pred, y_train_in, y_pred_train)
    assert result == 0


def test_custom_scorer_different_values():
    # Test custom_scorer function with different values
    y_valid = np.array([1, 2, 3])
    y_pred = np.array([2, 3, 4])
    y_train_in = np.array([1, 2, 3])
    y_pred_train = np.array([3, 4, 5])
    result = custom_scorer(y_valid, y_pred, y_train_in, y_pred_train)
    expected_result = (
        1 * mean_squared_error(y_valid, y_pred)
        + 0.5
        * abs(
            mean_squared_error(y_train_in, y_pred_train)
            - mean_squared_error(y_valid, y_pred)
        )
        + 2
        * max(
            0,
            (
                mean_squared_error(y_train_in, y_pred_train)
                - mean_squared_error(y_valid, y_pred)
                - 0.05
            ),
        )
        + 1
        * max(
            0,
            -(
                mean_squared_error(y_train_in, y_pred_train)
                - mean_squared_error(y_valid, y_pred)
            ),
        )
    )
    assert result == expected_result


def test_objective():
    # Test objective function
    X_train_in = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y_train_in = pd.Series([1, 2, 3, 4, 5])
    X_valid = pd.DataFrame({"feature": [6, 7, 8]})
    y_valid = pd.Series([6, 7, 8])
    pipe = Pipeline([("regressor", LinearRegression())])
    params = {"regressor__fit_intercept": False}

    def valid_scorer(y_valid, y_pred, y_train_in, y_pred_train):
        return np.mean(y_valid - y_pred)

    objective_scorer = ObjectiveScorer(valid_scorer)

    score_valid, score_train, score_of = objective(
        X_train_in, y_train_in, X_valid, y_valid, pipe, params, objective_scorer
    )

    y_pred = pipe.predict(X_valid)
    y_pred_train = pipe.predict(X_train_in)
    expected_score_valid = -mean_squared_error(y_valid, y_pred)
    expected_score_train = -mean_squared_error(y_train_in, y_pred_train)
    expected_score_of = -valid_scorer(y_valid, y_pred, y_train_in, y_pred_train)

    assert np.isclose(score_valid, expected_score_valid)
    assert np.isclose(score_train, expected_score_train)
    assert np.isclose(score_of, expected_score_of)


def test_parallel_objective():
    # Test parallel_objective function
    train_idx = np.array([0, 1, 2])
    valid_idx = np.array([3, 4])
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    pipe = Pipeline([("regressor", LinearRegression())])
    params_ = {"regressor__fit_intercept": False}

    def valid_scorer(y_valid, y_pred, y_train_in, y_pred_train):
        return np.mean(y_valid - y_pred)

    objective_scorer = ObjectiveScorer(valid_scorer)

    score_valid, score_train, score_of = parallel_objective(
        train_idx, valid_idx, X, y, pipe, params_, objective_scorer
    )

    y_pred = pipe.predict(X.iloc[valid_idx])
    y_pred_train = pipe.predict(X.iloc[train_idx])
    expected_score_valid = -mean_squared_error(y.iloc[valid_idx], y_pred)
    expected_score_train = -mean_squared_error(y.iloc[train_idx], y_pred_train)
    expected_score_of = -valid_scorer(
        y.iloc[valid_idx], y_pred, y.iloc[train_idx], y_pred_train
    )

    assert np.isclose(score_valid, expected_score_valid)
    assert np.isclose(score_train, expected_score_train)
    assert np.isclose(score_of, expected_score_of)


def test_objective_cv_sequential():
    # Test objective_cv function with n_jobs=1 (sequential)
    def valid_scorer(y_valid, y_pred, y_train_in, y_pred_train):
        return np.mean(y_valid - y_pred)

    trial = MagicMock()
    trial._suggest.return_value = False
    cross_val_split = KFold(n_splits=2).split
    pipe = Pipeline([("regressor", LinearRegression())])
    params = {
        "regressor__fit_intercept": {
            "name": "regressor__fit_intercept",
            "type": "categorical",
            "choices": [True, False],
        }
    }
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    run = MagicMock()
    n_jobs = 1
    objective_scorer = ObjectiveScorer(valid_scorer)

    result = objective_cv(
        trial, cross_val_split, pipe, params, X, y, run, n_jobs, objective_scorer
    )

    assert np.isclose(result, 0)


def valid_scorer(y_valid, y_pred, y_train_in, y_pred_train):
    return np.mean(y_valid - y_pred)


def test_objective_cv_parallel():
    # Test objective_cv function with n_jobs=-1 (parallel)

    trial = MagicMock()
    trial._suggest.return_value = False
    cross_val_split = KFold(n_splits=2).split
    pipe = Pipeline([("regressor", LinearRegression())])
    params = {
        "regressor__fit_intercept": {
            "name": "regressor__fit_intercept",
            "type": "categorical",
            "choices": [True, False],
        }
    }
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    run = MagicMock()
    n_jobs = -1
    objective_scorer = ObjectiveScorer(valid_scorer)

    result = objective_cv(
        trial, cross_val_split, pipe, params, X, y, run, n_jobs, objective_scorer
    )

    assert np.isclose(result, 0)


def test_objective_cv_same_results_sequential_parallel():
    # Test objective_cv function with n_jobs=1 and n_jobs=-1

    trial = MagicMock()
    trial._suggest.return_value = False
    cross_val_split = KFold(n_splits=2).split
    pipe = Pipeline([("regressor", LinearRegression())])
    params = {
        "regressor__fit_intercept": {
            "name": "regressor__fit_intercept",
            "type": "categorical",
            "choices": [True, False],
        }
    }
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([1, 2, 3, 4, 5])
    run = MagicMock()
    objective_scorer = ObjectiveScorer(valid_scorer)

    result_sequential = objective_cv(
        trial, cross_val_split, pipe, params, X, y, run, 1, objective_scorer
    )
    result_parallel = objective_cv(
        trial, cross_val_split, pipe, params, X, y, run, -1, objective_scorer
    )

    assert np.isclose(result_sequential, result_parallel)
