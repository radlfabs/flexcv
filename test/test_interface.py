import pandas as pd
import pytest
from unittest.mock import patch
from unittest.mock import MagicMock

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
    
from flexcv.interface import CrossValidationResults, CrossValidation
from flexcv.split import CrossValMethod
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.run import NeptuneRun
from flexcv.run import Run as DummyRun


def test_cross_validation_init():
    # Test initialization
    cv = CrossValidation()
    assert isinstance(cv, CrossValidation)
    assert cv.config["n_splits_out"] == 5
    assert cv.config["n_splits_in"] == 5
    assert cv.config["split_out"] == CrossValMethod.KFOLD
    assert cv.config["split_in"] == CrossValMethod.KFOLD
    assert cv.config["scale_out"] == True
    assert cv.config["scale_in"] == True
    assert cv.config["random_seed"] == 42
    assert cv.config["diagnostics"] == False
    assert cv.config["break_cross_val"] == False
    assert isinstance(cv.config["run"], DummyRun)

def test_cross_validation_set_data():
    # Test set_data method
    cv = CrossValidation()
    X = pd.DataFrame({'column1': [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    cv.set_data(X, y, target_name='target', dataset_name='dataset')
    assert cv.config["X"].equals(X)
    assert cv.config["y"].equals(y)
    assert cv.config["target_name"] == 'target'
    assert cv.config["dataset_name"] == 'dataset'

def test_cross_validation_set_data_invalid_X():
    # Test set_data method with invalid X
    cv = CrossValidation()
    X = "invalid type"
    y = pd.Series([1, 2, 3])
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name='target', dataset_name='dataset')

def test_cross_validation_set_data_invalid_y():
    # Test set_data method with invalid y
    cv = CrossValidation()
    X = pd.DataFrame({'column1': [1, 2, 3]})
    y = "invalid type"
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name='target', dataset_name='dataset')

def test_cross_validation_set_data_invalid_groups():
    # Test set_data method with invalid groups
    cv = CrossValidation()
    X = pd.DataFrame({'column1': [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    groups = "invalid type"
    with pytest.raises(TypeError):
        cv.set_data(X, y, groups=groups, target_name='target', dataset_name='dataset')

def test_cross_validation_set_data_invalid_slopes():
    # Test set_data method with invalid slopes
    cv = CrossValidation()
    X = pd.DataFrame({'column1': [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    slopes = "invalid type"
    with pytest.raises(TypeError):
        cv.set_data(X, y, slopes=slopes, target_name='target', dataset_name='dataset')

def test_cross_validation_set_data_invalid_target_name():
    # Test set_data method with invalid target_name
    cv = CrossValidation()
    X = pd.DataFrame({'column1': [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    target_name = 123
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name=target_name, dataset_name='dataset')

def test_cross_validation_set_data_invalid_dataset_name():
    # Test set_data method with invalid dataset_name
    cv = CrossValidation()
    X = pd.DataFrame({'column1': [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    dataset_name = 123
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name='target', dataset_name=dataset_name)

def test_cross_validation_set_splits():
    # Test set_splits method
    cv = CrossValidation()
    cv.set_splits(
        split_out=CrossValMethod.GROUP, 
        split_in=CrossValMethod.GROUP, 
        n_splits_out=3, 
        n_splits_in=3, 
        scale_out=False, 
        scale_in=False
        )
    assert cv.config["n_splits_out"] == 3
    assert cv.config["n_splits_in"] == 3
    assert cv.config["split_out"] == CrossValMethod.GROUP
    assert cv.config["split_in"] == CrossValMethod.GROUP
    assert cv.config["scale_out"] == False
    assert cv.config["scale_in"] == False
    
def test_cross_validation_set_splits_invalid_split_out():
    # Test set_splits method with invalid split_out
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(split_out="InvalidMethod")

def test_cross_validation_set_splits_invalid_split_in():
    # Test set_splits method with invalid split_in
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(split_in="InvalidMethod")

def test_cross_validation_set_splits_invalid_n_splits_out():
    # Test set_splits method with invalid n_splits_out
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(n_splits_out="invalid type")

def test_cross_validation_set_splits_invalid_n_splits_in():
    # Test set_splits method with invalid n_splits_in
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(n_splits_in="invalid type")

def test_cross_validation_set_splits_invalid_scale_out():
    # Test set_splits method with invalid scale_out
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(scale_out="invalid type")

def test_cross_validation_set_splits_invalid_scale_in():
    # Test set_splits method with invalid scale_in
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(scale_in="invalid type")

def test_cross_validation_set_splits_invalid_break_cross_val():
    # Test set_splits method with invalid break_cross_val
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(break_cross_val="invalid type")

def test_cross_validation_set_splits_invalid_metrics():
    # Test set_splits method with invalid metrics
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(metrics="invalid type")

def test_set_splits_with_cross_val_method():
    cv = CrossValidation()
    cv.set_splits(split_out=CrossValMethod.KFOLD, split_in=CrossValMethod.KFOLD)
    assert cv.config["split_out"] == CrossValMethod.KFOLD
    assert cv.config["split_in"] == CrossValMethod.KFOLD

def test_set_splits_with_string():
    cv = CrossValidation()
    cv.set_splits(split_out="KFold", split_in="KFold")
    assert cv.config["split_out"] == CrossValMethod.KFOLD
    assert cv.config["split_in"] == CrossValMethod.KFOLD

def test_set_splits_with_sklearn_cross_validator():
    cv = CrossValidation()
    kfold = KFold(n_splits=5)
    cv.set_splits(split_out=kfold, split_in=kfold)
    assert cv.config["split_out"] == kfold
    assert cv.config["split_in"] == kfold

def test_set_splits_with_iterator():
    cv = CrossValidation()
    iterator = iter([np.array([1, 2, 3]), np.array([4, 5, 6])])
    cv.set_splits(split_out=iterator, split_in=iterator)
    assert cv.config["split_out"] == iterator
    assert cv.config["split_in"] == iterator

def test_set_splits_with_invalid_string():
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(split_out="InvalidMethod", split_in="InvalidMethod")

def test_set_splits_with_invalid_type():
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_splits(split_out=123, split_in=123)
        
def test_cross_validation_set_models_valid():
    # Test set_models method with valid mapping
    cv = CrossValidation()
    mapping = ModelMappingDict({
        "RandomForestRegressor": ModelConfigDict({
            "model": RandomForestRegressor,
            "parameters": {"n_estimators": 100}
        })
    })
    cv.set_models(mapping)
    assert cv.config["mapping"] == mapping

def test_cross_validation_set_models_invalid():
    # Test set_models method with invalid mapping
    cv = CrossValidation()
    mapping = "invalid type"
    with pytest.raises(TypeError):
        cv.set_models(mapping)
        
def test_cross_validation_set_inner_cv_valid():
    # Test set_inner_cv method with valid arguments
    cv = CrossValidation()
    cv.set_inner_cv(n_trials=100, objective_scorer=None)
    assert cv.config["n_trials"] == 100
    assert cv.config["objective_scorer"] is None

def test_cross_validation_set_inner_cv_invalid_n_trials():
    # Test set_inner_cv method with invalid n_trials
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_inner_cv(n_trials="invalid type")

def test_cross_validation_set_inner_cv_invalid_objective_scorer():
    # Test set_inner_cv method with invalid objective_scorer
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_inner_cv(objective_scorer="invalid type")
        
def test_cross_validation_set_mixed_effects_valid():
    # Test set_mixed_effects method with valid arguments
    cv = CrossValidation()
    cv.set_mixed_effects(
        model_mixed_effects=True, 
        em_max_iterations=100, 
        em_stopping_threshold=0.1, 
        em_stopping_window=10, 
        predict_known_groups_lmm=True
        )
    assert cv.config["model_effects"] == "mixed"
    assert cv.config["em_max_iterations"] == 100
    assert cv.config["em_stopping_threshold"] == 0.1
    assert cv.config["em_stopping_window"] == 10
    assert cv.config["predict_known_groups_lmm"] == True

def test_cross_validation_set_mixed_effects_invalid_model_mixed_effects():
    # Test set_mixed_effects method with invalid model_mixed_effects
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_mixed_effects(model_mixed_effects="invalid type")

def test_cross_validation_set_mixed_effects_invalid_em_max_iterations():
    # Test set_mixed_effects method with invalid em_max_iterations
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_mixed_effects(em_max_iterations="invalid type")

def test_cross_validation_set_mixed_effects_invalid_em_stopping_threshold():
    # Test set_mixed_effects method with invalid em_stopping_threshold
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_mixed_effects(em_stopping_threshold="invalid type")

def test_cross_validation_set_mixed_effects_invalid_em_stopping_window():
    # Test set_mixed_effects method with invalid em_stopping_window
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_mixed_effects(em_stopping_window="invalid type")

def test_cross_validation_set_mixed_effects_invalid_predict_known_groups_lmm():
    # Test set_mixed_effects method with invalid predict_known_groups_lmm
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_mixed_effects(predict_known_groups_lmm="invalid type")
        
def test_cross_validation_set_run_valid():
    # Test set_run method with valid arguments
    cv = CrossValidation()
    run = DummyRun()
    cv.set_run(run=run, diagnostics=True, random_seed=123)
    assert cv.config["run"] == run
    assert cv.config["diagnostics"] == True
    assert cv.config["random_seed"] == 123

def test_cross_validation_set_run_invalid_run():
    # Test set_run method with invalid run
    cv = CrossValidation()
    run = "invalid type"
    with pytest.raises(TypeError):
        cv.set_run(run=run)

def test_cross_validation_set_run_invalid_diagnostics():
    # Test set_run method with invalid diagnostics
    cv = CrossValidation()
    diagnostics = "invalid type"
    with pytest.raises(TypeError):
        cv.set_run(diagnostics=diagnostics)

def test_cross_validation_set_run_invalid_random_seed():
    # Test set_run method with invalid random_seed
    cv = CrossValidation()
    random_seed = "invalid type"
    with pytest.raises(TypeError):
        cv.set_run(random_seed=random_seed)

def test_cross_validation_log_no_run():
    # Test _log method with no run
    cv = CrossValidation()
    cv.config["dataset_name"] = "dataset"
    cv.config["target_name"] = "target"
    cv.config["model_effects"] = "mixed"
    cv.config["X"] = pd.DataFrame({'column1': [1, 2, 3]})
    cv.config["y"] = pd.Series([1, 2, 3])
    cv.config["groups"] = None
    cv.config["slopes"] = None
    cv.config["split_out"] = CrossValMethod.KFOLD
    cv.config["split_in"] = CrossValMethod.KFOLD
    cv.config["n_splits_out"] = 5
    cv.config["n_splits_in"] = 5
    cv.config["scale_in"] = True
    cv.config["scale_out"] = True
    cv.config["break_cross_val"] = False
    cv.config["metrics"] = ["r2", "mse", "mae"]
    cv.config["mapping"] = ModelMappingDict({
        "RandomForestRegressor": ModelConfigDict({
            "model": RandomForestRegressor,
            "parameters": {"n_estimators": 100}
        })
    })
    cv.config["n_trials"] = 100
    cv.config["objective_scorer"] = None
    cv.config["em_max_iterations"] = 100
    cv.config["em_stopping_threshold"] = 0.1
    cv.config["em_stopping_window"] = 10
    cv.config["predict_known_groups_lmm"] = True
    cv.config["diagnostics"] = False
    cv.config["random_seed"] = 42
    cv.results_ = MagicMock()
    cv.results_.summary = pd.DataFrame({'column1': [1, 2, 3]})
    cv._prepare_before_perform()
    cv._log()
    assert cv._was_logged_ == True

def test_cross_validation_log_with_run():
    # Test _log method with run
    cv = CrossValidation()
    cv.config["dataset_name"] = "dataset"
    cv.config["target_name"] = "target"
    cv.config["model_effects"] = "mixed"
    cv.config["X"] = pd.DataFrame({'column1': [1, 2, 3]})
    cv.config["y"] = pd.Series([1, 2, 3])
    cv.config["groups"] = None
    cv.config["slopes"] = None
    run = DummyRun()
    cv.config["run"] = run
    cv._log()
    assert cv._was_logged_ == True

def test_cross_validation_log_no_results():
    # Test _log method when results_ does not exist
    cv = CrossValidation()
    cv.config["dataset_name"] = "dataset"
    cv.config["target_name"] = "target"
    cv.config["model_effects"] = "mixed"
    cv.config["X"] = pd.DataFrame({'column1': [1, 2, 3]})
    cv.config["y"] = pd.Series([1, 2, 3])
    cv.config["groups"] = None
    cv.config["slopes"] = None
    cv._was_logged_ = False
    with patch('flexcv.interface.logger.warning') as mock_warning:
        cv._log()
        mock_warning.assert_called_once_with("You have not called perform() yet. No results to log. Call perform() to log the results.")

def test_cross_validation_get_results_was_performed():
    # Test get_results method when was_performed_ is True
    cv = CrossValidation()
    cv.was_performed_ = True
    random_results = {
        "RandomForestRegressor": {
            "r2": [0.1, 0.2, 0.3],
            "mse": [0.1, 0.2, 0.3],
            "mae": [0.1, 0.2, 0.3],
        }
    }
    cv.results_ = CrossValidationResults(random_results)
    results = cv.get_results()
    assert isinstance(results, CrossValidationResults)

def test_cross_validation_get_results_not_performed():
    # Test get_results method when was_performed_ is False
    cv = CrossValidation()
    cv.was_performed_ = False
    with pytest.raises(RuntimeError):
        cv.get_results()
        
def test_cross_validation_results_no_dict_passed():
    # Test results property when was_performed_ is True
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.results_ = CrossValidationResults()

def test_cross_validation_results_not_performed():
    # Test results property when was_performed_ is False
    cv = CrossValidation()
    with pytest.raises(RuntimeError):
        cv.results

def test_prepare_before_perform():
    # Test _prepare_before_perform method
    cv = CrossValidation()
    cv.config["split_out"] = "kfold"
    cv.config["split_in"] = "group"
    cv.config["mapping"] = ModelMappingDict({
        "RandomForestRegressor": ModelConfigDict({
            "model": RandomForestRegressor,
            "parameters": {"n_estimators": 100}
        })
    })
    cv.config["n_trials"] = 100
    cv._prepare_before_perform()
    assert cv.config["split_out"] == CrossValMethod.KFOLD
    assert cv.config["split_in"] == CrossValMethod.GROUP
    assert isinstance(cv.config["run"], DummyRun)
    assert cv.config["mapping"]["RandomForestRegressor"]["n_trials"] == 100

def test_prepare_before_perform_with_run():
    # Test _prepare_before_perform method with a run already set
    cv = CrossValidation()
    cv.config["run"] = NeptuneRun()
    cv._prepare_before_perform()
    assert isinstance(cv.config["run"], NeptuneRun)

def test_prepare_before_perform_with_n_trials():
    # Test _prepare_before_perform method with n_trials already set in mapping
    cv = CrossValidation()
    cv.config["mapping"] = ModelMappingDict({
        "RandomForestRegressor": ModelConfigDict({
            "model": RandomForestRegressor,
            "parameters": {"n_estimators": 100},
            "n_trials": 50
        })
    })
    cv.config["n_trials"] = 100
    cv._prepare_before_perform()
    assert cv.config["mapping"]["RandomForestRegressor"]["n_trials"] == 50
    
