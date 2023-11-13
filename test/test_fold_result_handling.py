import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from flexcv.run import Run 
from flexcv.fold_results_handling import SingleModelFoldResult

def test_single_model_fold_result_init():
    # Test initialization
    k = 1
    model_name = "RandomForestRegressor"
    best_model = RandomForestRegressor()
    best_params = {"n_estimators": 100}
    y_pred = pd.Series(np.random.rand(10))
    y_test = pd.Series(np.random.rand(10))
    X_test = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series(np.random.rand(10))
    y_pred_train = pd.Series(np.random.rand(10))
    X_train = pd.DataFrame(np.random.rand(10, 5))
    fit_result = None
    single_model_fold_result = SingleModelFoldResult(
        k,
        model_name,
        best_model,
        best_params,
        y_pred,
        y_test,
        X_test,
        y_train,
        y_pred_train,
        X_train,
        fit_result,
    )
    assert single_model_fold_result.k == k
    assert single_model_fold_result.model_name == model_name
    assert single_model_fold_result.best_model == best_model
    assert single_model_fold_result.best_params == best_params
    pd.testing.assert_series_equal(single_model_fold_result.y_pred, y_pred)
    pd.testing.assert_series_equal(single_model_fold_result.y_test, y_test)
    pd.testing.assert_frame_equal(single_model_fold_result.X_test, X_test)
    pd.testing.assert_series_equal(single_model_fold_result.y_train, y_train)
    pd.testing.assert_series_equal(single_model_fold_result.y_pred_train, y_pred_train)
    pd.testing.assert_frame_equal(single_model_fold_result.X_train, X_train)
    assert single_model_fold_result.fit_result == fit_result

def test_single_model_fold_result_make_results_invalid_run():
    # Test make_results method with invalid run
    k = 1
    model_name = "RandomForestRegressor"
    best_model = RandomForestRegressor()
    best_params = {"n_estimators": 100}
    y_pred = pd.Series(np.random.rand(10))
    y_test = pd.Series(np.random.rand(10))
    X_test = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series(np.random.rand(10))
    y_pred_train = pd.Series(np.random.rand(10))
    X_train = pd.DataFrame(np.random.rand(10, 5))
    fit_result = None
    single_model_fold_result = SingleModelFoldResult(
        k,
        model_name,
        best_model,
        best_params,
        y_pred,
        y_test,
        X_test,
        y_train,
        y_pred_train,
        X_train,
        fit_result,
    )
    run = "invalid type"
    results_all_folds = {}
    study = None
    metrics = {"mse": mean_squared_error}
    with pytest.raises(TypeError):
        single_model_fold_result.make_results(run, results_all_folds, study, metrics)

def test_single_model_fold_result_make_results_eval_metrics():
    # Test make_results method with focus on evaluation metrics
    k = 1
    model_name = "RandomForestRegressor"
    best_model = RandomForestRegressor()
    best_params = {"n_estimators": 100}
    y_pred = pd.Series(np.random.rand(10))
    y_test = pd.Series(np.random.rand(10))
    X_test = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series(np.random.rand(10))
    y_pred_train = pd.Series(np.random.rand(10))
    X_train = pd.DataFrame(np.random.rand(10, 5))
    fit_result = None
    single_model_fold_result = SingleModelFoldResult(
        k,
        model_name,
        best_model,
        best_params,
        y_pred,
        y_test,
        X_test,
        y_train,
        y_pred_train,
        X_train,
        fit_result,
    )
    run = Run()
    results_all_folds = {}
    study = None
    metrics = {"mse": mean_squared_error}
    results = single_model_fold_result.make_results(run, results_all_folds, study, metrics)
    assert isinstance(results, dict)
    assert model_name in results
    assert isinstance(results[model_name], dict)
    assert "metrics" in results[model_name]
    assert isinstance(results[model_name]["metrics"], list)
    assert len(results[model_name]["metrics"]) == 1
    assert isinstance(results[model_name]["metrics"][0], dict)
    assert "mse" in results[model_name]["metrics"][0]
    assert "mse_train" in results[model_name]["metrics"][0]
    assert np.isclose(results[model_name]["metrics"][0]["mse"], mean_squared_error(y_test, y_pred))
    assert np.isclose(results[model_name]["metrics"][0]["mse_train"], mean_squared_error(y_train, y_pred_train))
