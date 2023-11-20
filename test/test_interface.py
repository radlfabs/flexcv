from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
from flexcv._data import DATA_TUPLE_3_25
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback

import flexcv
from flexcv import model_postprocessing
from flexcv.interface import CrossValidation, CrossValidationResults
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearModel
from flexcv.run import NeptuneRun
from flexcv.run import Run as DummyRun
from flexcv.split import CrossValMethod



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
    X = pd.DataFrame({"column1": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    cv.set_data(X, y, target_name="target", dataset_name="dataset")
    assert cv.config["X"].equals(X)
    assert cv.config["y"].equals(y)
    assert cv.config["target_name"] == "target"
    assert cv.config["dataset_name"] == "dataset"


def test_cross_validation_set_data_invalid_X():
    # Test set_data method with invalid X
    cv = CrossValidation()
    X = "invalid type"
    y = pd.Series([1, 2, 3])
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name="target", dataset_name="dataset")


def test_cross_validation_set_data_invalid_y():
    # Test set_data method with invalid y
    cv = CrossValidation()
    X = pd.DataFrame({"column1": [1, 2, 3]})
    y = "invalid type"
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name="target", dataset_name="dataset")


def test_cross_validation_set_data_invalid_groups():
    # Test set_data method with invalid groups
    cv = CrossValidation()
    X = pd.DataFrame({"column1": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    groups = "invalid type"
    with pytest.raises(TypeError):
        cv.set_data(X, y, groups=groups, target_name="target", dataset_name="dataset")


def test_cross_validation_set_data_invalid_slopes():
    # Test set_data method with invalid slopes
    cv = CrossValidation()
    X = pd.DataFrame({"column1": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    slopes = "invalid type"
    with pytest.raises(TypeError):
        cv.set_data(X, y, slopes=slopes, target_name="target", dataset_name="dataset")


def test_cross_validation_set_data_invalid_target_name():
    # Test set_data method with invalid target_name
    cv = CrossValidation()
    X = pd.DataFrame({"column1": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    target_name = 123
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name=target_name, dataset_name="dataset")


def test_cross_validation_set_data_invalid_dataset_name():
    # Test set_data method with invalid dataset_name
    cv = CrossValidation()
    X = pd.DataFrame({"column1": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    dataset_name = 123
    with pytest.raises(TypeError):
        cv.set_data(X, y, target_name="target", dataset_name=dataset_name)


def test_cross_validation_set_splits():
    # Test set_splits method
    cv = CrossValidation()
    cv.set_splits(
        split_out=CrossValMethod.GROUP,
        split_in=CrossValMethod.GROUP,
        n_splits_out=3,
        n_splits_in=3,
        scale_out=False,
        scale_in=False,
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
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": ModelConfigDict(
                {"model": RandomForestRegressor, "parameters": {"n_estimators": 100}}
            )
        }
    )
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


def test_set_lmer_valid():
    # Test set_lmer method with valid arguments
    cv = CrossValidation()
    cv.set_lmer(predict_known_groups_lmm=True)
    assert cv.config["predict_known_groups_lmm"] == True


def test_set_lmer_invalid_predict_known_groups_lmm():
    # Test set_lmer method with invalid predict_known_groups_lmm
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_lmer(predict_known_groups_lmm="invalid type")


def test_cross_validation_set_run_valid():
    # Test set_run method with valid arguments
    cv = CrossValidation()
    run = DummyRun()
    cv.set_run(run=run, diagnostics=True, random_seed=123)
    assert cv.config["run"] == run
    assert cv.config["diagnostics"] == True
    assert cv.config["random_seed"] == 123


def test_set_models_with_mapping():
    # Test set_models method with valid mapping
    cv = CrossValidation()
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": ModelConfigDict(
                {"model": RandomForestRegressor, "parameters": {"n_estimators": 100}}
            )
        }
    )
    cv.set_models(mapping=mapping)
    assert cv.config["mapping"] == mapping


def test_set_models_with_path():
    # Test set_models method with valid path
    cv = CrossValidation()
    path = "path/to/yaml/file"
    with patch("flexcv.interface.read_mapping_from_yaml_file") as mock_read:
        mock_mapping = ModelMappingDict(
            {
                "RandomForestRegressor": ModelConfigDict(
                    {
                        "model": RandomForestRegressor,
                        "parameters": {"n_estimators": 100},
                    }
                )
            }
        )
        mock_read.return_value = mock_mapping
        cv.set_models(yaml_path=path)
        mock_read.assert_called_once_with(path)
        assert cv.config["mapping"] == mock_mapping


def test_set_models_with_none():
    # Test set_models method with None for both mapping and path
    cv = CrossValidation()
    with pytest.raises(ValueError):
        cv.set_models()


def test_set_models_with_invalid_mapping():
    # Test set_models method with invalid mapping
    cv = CrossValidation()
    mapping = "invalid type"
    with pytest.raises(TypeError):
        cv.set_models(mapping=mapping)


def test_set_models_with_yaml_code():
    # Test set_models method with YAML code
    cv = CrossValidation()
    yaml_code = """
    LinearModel:
        model: flexcv.models.LinearModel
        post_processor: flexcv.model_postprocessing.LinearModelPostProcessor
        params:
            max_depth: !Int
                low: 5
                high: 100
                log: true
            min_impurity_decrease: !Float
                low: 0.00000001
                high: 0.02
            features: !Cat 
                choices: [a, b, c]
    """
    with patch("flexcv.interface.read_mapping_from_yaml_string") as mock_read:
        mock_mapping = ModelMappingDict(
            {
                "LinearModel": ModelConfigDict(
                    {
                        "model": flexcv.models.LinearModel,
                        "post_processor": flexcv.model_postprocessing.LinearModelPostProcessor,
                        "params": {
                            "max_depth": optuna.distributions.IntDistribution(
                                low=5, high=100, log=True
                            ),
                            "min_impurity_decrease": optuna.distributions.FloatDistribution(
                                low=0.00000001, high=0.02
                            ),
                            "features": optuna.distributions.CategoricalDistribution(
                                choices=["a", "b", "c"]
                            ),
                        },
                    }
                )
            }
        )
        mock_read.return_value = mock_mapping
        cv.set_models(yaml_string=yaml_code)
        mock_read.assert_called_once_with(yaml_code)
        assert cv.config["mapping"] == mock_mapping


def test_set_models_with_none():
    # Test set_models method with None for all arguments
    cv = CrossValidation()
    with pytest.raises(ValueError):
        cv.set_models()


def test_set_models_with_multiple_arguments():
    # Test set_models method with multiple arguments
    cv = CrossValidation()
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": ModelConfigDict(
                {"model": RandomForestRegressor, "parameters": {"n_estimators": 100}}
            )
        }
    )
    yaml_code = """
    RandomForest:
        model: sklearn.ensemble.RandomForestRegressor
        post_processor: flexcv.model_postprocessing.MixedEffectsPostProcessor
        requires_inner_cv: True
        params:
            max_depth: !Int
                low: 1
                high: 10
    """
    with pytest.raises(ValueError):
        cv.set_models(mapping=mapping, yaml_string=yaml_code)



def test_set_models_with_invalid_yaml_code():
    # Test set_models method with invalid yaml_code
    cv = CrossValidation()
    yaml_code = 123
    with pytest.raises(TypeError):
        cv.set_models(yaml_code=yaml_code)


def test_set_merf_valid():
    # Test set_merf method with valid arguments
    cv = CrossValidation()
    cv.set_merf(
        add_merf_global=True,
        em_max_iterations=100,
        em_stopping_threshold=0.1,
        em_stopping_window=10,
    )
    assert cv.config["add_merf_global"] == True
    assert cv.config["em_max_iterations"] == 100
    assert cv.config["em_stopping_threshold"] == 0.1
    assert cv.config["em_stopping_window"] == 10


def test_set_merf_add_merf_defaults():
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": ModelConfigDict(
                {"model": RandomForestRegressor, "parameters": {"n_estimators": 100}}
            )
        }
    )
    cv = CrossValidation()
    cv.set_models(mapping)
    cv._prepare_before_perform()
    assert cv.config["mapping"]["RandomForestRegressor"]["add_merf"] == False


def test_set_merf_add_merf_override():
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": ModelConfigDict(
                {
                    "model": RandomForestRegressor,
                    "parameters": {"n_estimators": 100},
                    "add_merf": True,
                }
            )
        }
    )
    cv = CrossValidation()
    cv.set_models(mapping)
    cv.set_merf(add_merf_global=False)
    cv._prepare_before_perform()
    assert cv.config["mapping"]["RandomForestRegressor"]["add_merf"] == True


def test_set_merf_invalid_add_merf_global():
    # Test set_merf method with invalid add_merf_global
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_merf(add_merf_global="invalid type")


def test_set_merf_invalid_em_max_iterations():
    # Test set_merf method with invalid em_max_iterations
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_merf(em_max_iterations="invalid type")


def test_set_merf_invalid_em_stopping_threshold():
    # Test set_merf method with invalid em_stopping_threshold
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_merf(em_stopping_threshold="invalid type")


def test_set_merf_invalid_em_stopping_window():
    # Test set_merf method with invalid em_stopping_window
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.set_merf(em_stopping_window="invalid type")


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
    cv.config["X"] = pd.DataFrame({"column1": [1, 2, 3]})
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
    cv.config["mapping"] = ModelMappingDict(
        {
            "RandomForestRegressor": ModelConfigDict(
                {"model": RandomForestRegressor, "parameters": {"n_estimators": 100}}
            )
        }
    )
    cv.config["n_trials"] = 100
    cv.config["objective_scorer"] = None
    cv.config["em_max_iterations"] = 100
    cv.config["em_stopping_threshold"] = 0.1
    cv.config["em_stopping_window"] = 10
    cv.config["predict_known_groups_lmm"] = True
    cv.config["diagnostics"] = False
    cv.config["random_seed"] = 42
    cv.results_ = MagicMock()
    cv.results_.summary = pd.DataFrame({"column1": [1, 2, 3]})
    cv._prepare_before_perform()
    cv._log_config()
    assert cv._config_logged_ == True


def test_cross_validation_log_with_run():
    # Test _log method with run
    cv = CrossValidation()
    cv.config["dataset_name"] = "dataset"
    cv.config["target_name"] = "target"
    cv.config["model_effects"] = "mixed"
    cv.config["X"] = pd.DataFrame({"column1": [1, 2, 3]})
    cv.config["y"] = pd.Series([1, 2, 3])
    cv.config["groups"] = None
    cv.config["slopes"] = None
    run = DummyRun()
    cv.config["run"] = run
    cv._log_config()
    assert cv._config_logged_ == True


def test_cross_validation_log_no_results():
    # Test _log method when results_ does not exist
    cv = CrossValidation()
    cv.config["dataset_name"] = "dataset"
    cv.config["target_name"] = "target"
    cv.config["model_effects"] = "mixed"
    cv.config["X"] = pd.DataFrame({"column1": [1, 2, 3]})
    cv.config["y"] = pd.Series([1, 2, 3])
    cv.config["groups"] = None
    cv.config["slopes"] = None
    cv._was_logged_ = False
    with patch("flexcv.interface.logger.warning") as mock_warning:
        cv._log_config()
        cv._log_results()
        mock_warning.assert_called_once_with(
            "You have not called perform() yet. No results to log. Call perform() to log the results."
        )


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
    X, y = pd.DataFrame({"column1": [1, 2, 3]}), pd.Series([1, 2, 3])
    cv = CrossValidation()
    cv.set_data(X, y, target_name="target", dataset_name="dataset")
    cv.set_models(
        ModelMappingDict(
            {
                "RandomForestRegressor": ModelConfigDict(
                    {
                        "model": RandomForestRegressor,
                        "parameters": {"n_estimators": 100},
                    }
                )
            }
        )
    )
    cv._prepare_before_perform()
    assert cv.config["split_out"] == CrossValMethod.KFOLD
    assert cv.config["split_in"] == CrossValMethod.KFOLD
    assert isinstance(cv.config["run"], DummyRun)
    assert cv.config["add_merf_global"] == False
    assert (
        cv.config["mapping"]["RandomForestRegressor"]["n_trials"]
        == cv.config["n_trials"]
    )
    assert (
        cv.config["mapping"]["RandomForestRegressor"]["add_merf"]
        == cv.config["add_merf_global"]
    )
    assert cv.config["mapping"]["RandomForestRegressor"]["consumes_clusters"] == False
    assert cv.config["mapping"]["RandomForestRegressor"]["requires_formula"] == False
    assert (
        cv.config["mapping"]["RandomForestRegressor"]["model_kwargs"]["n_jobs"]
        == cv.config["mapping"]["RandomForestRegressor"]["n_jobs_model"]
    )
    assert (
        cv.config["mapping"]["RandomForestRegressor"]["model_kwargs"]["random_state"]
        == cv.config["random_seed"]
    )
    assert cv.config["mapping"]["RandomForestRegressor"]["fit_kwargs"] == {}


def test_prepare_before_perform_n_trials_added_to_mapping():
    # Test _prepare_before_perform method when n_trials is added to mapping
    X, y = pd.DataFrame({"column1": [1, 2, 3]}), pd.Series([1, 2, 3])
    cv = CrossValidation()
    cv.config["n_trials"] = 10
    cv.set_data(X, y, target_name="target", dataset_name="dataset")
    cv.set_models(
        ModelMappingDict(
            {
                "RandomForestRegressor": ModelConfigDict(
                    {
                        "model": RandomForestRegressor,
                        "parameters": {"n_estimators": 100},
                    }
                )
            }
        )
    )
    cv._prepare_before_perform()
    assert cv.config["mapping"]["RandomForestRegressor"]["n_trials"] == 10


def test_prepare_before_perform_mapping_overrides_n_trials():
    # Test _prepare_before_perform method when mapping overrides n_trials
    X, y = pd.DataFrame({"column1": [1, 2, 3]}), pd.Series([1, 2, 3])
    cv = CrossValidation()
    cv.config["n_trials"] = 20
    cv.set_data(X, y, target_name="target", dataset_name="dataset")
    cv.set_models(
        ModelMappingDict(
            {
                "RandomForestRegressor": ModelConfigDict(
                    {
                        "model": RandomForestRegressor,
                        "parameters": {"n_estimators": 100},
                        "n_trials": 10,
                    }
                )
            }
        )
    )
    cv._prepare_before_perform()
    assert cv.config["mapping"]["RandomForestRegressor"]["n_trials"] == 10


def test_prepare_before_perform_invalid_split_out():
    # Test _prepare_before_perform method with invalid split_out
    cv = CrossValidation()
    cv.config["split_out"] = "InvalidMethod"
    with pytest.raises(TypeError):
        cv._prepare_before_perform()


def test_prepare_before_perform_invalid_split_in():
    # Test _prepare_before_perform method with invalid split_in
    cv = CrossValidation()
    cv.config["split_in"] = "InvalidMethod"
    with pytest.raises(TypeError):
        cv._prepare_before_perform()


def test_prepare_before_perform_invalid_model():
    # Test _prepare_before_perform method with invalid model
    X, y = pd.DataFrame({"column1": [1, 2, 3]}), pd.Series([1, 2, 3])
    cv = CrossValidation()
    cv.set_data(X, y, target_name="target", dataset_name="dataset")
    cv.set_models(
        ModelMappingDict(
            {
                "InvalidModel": ModelConfigDict(
                    {"model": "InvalidModel", "params": {"n_estimators": 100}}
                )
            }
        )
    )
    with pytest.raises(TypeError):
        cv._prepare_before_perform()


def test_add_model_valid():
    # Test add_model method with valid arguments
    cv = CrossValidation()
    model_class = RandomForestRegressor
    model_name = "RandomForestRegressor"
    post_processor = model_postprocessing.ModelPostProcessor
    params = {"n_estimators": 100}
    callbacks = [MagicMock()]
    cv.add_model(model_class, False, model_name, post_processor, params, callbacks)
    assert model_name in cv.config["mapping"]
    assert isinstance(cv.config["mapping"][model_name], ModelConfigDict)
    assert cv.config["mapping"][model_name]["model"] == model_class
    assert cv.config["mapping"][model_name]["post_processor"] == post_processor
    assert (
        cv.config["mapping"][model_name]["params"]["n_estimators"]
        == params["n_estimators"]
    )


def test_add_model_no_model_name():
    # Test add_model method with no model_name
    cv = CrossValidation()
    model_class = RandomForestRegressor
    post_processor = model_postprocessing.ModelPostProcessor
    params = {"n_estimators": 100}
    cv.add_model(model_class, False, "", post_processor, params)
    model_name = model_class.__name__
    assert model_name in cv.config["mapping"]
    assert isinstance(cv.config["mapping"][model_name], ModelConfigDict)
    assert cv.config["mapping"][model_name]["model"] == model_class
    assert cv.config["mapping"][model_name]["post_processor"] == post_processor
    assert (
        cv.config["mapping"][model_name]["params"]["n_estimators"]
        == params["n_estimators"]
    )


def test_add_model_invalid_model_name():
    # Test add_model method with invalid model_name
    cv = CrossValidation()
    model_class = RandomForestRegressor
    model_name = 123
    with pytest.raises(TypeError):
        cv.add_model(model_class, model_name)


def test_add_model_invalid_model_class():
    # Test add_model method with invalid model_class
    cv = CrossValidation()
    model_class = "invalid type"
    with pytest.raises(TypeError):
        cv.add_model(model_class)


def test_add_model_invalid_post_processor():
    # Test add_model method with invalid post_processor
    cv = CrossValidation()
    model_class = RandomForestRegressor
    post_processor = "invalid type"
    with pytest.raises(TypeError):
        cv.add_model(model_class, post_processor=post_processor)


def test_add_model_invalid_skip_inner_cv():
    # Test add_model method with invalid skip_inner_cv
    cv = CrossValidation()
    model_class = RandomForestRegressor
    skip_inner_cv = "invalid type"
    with pytest.raises(TypeError):
        cv.add_model(model_class, skip_inner_cv)


def test_add_model_invalid_params():
    # Test add_model method with invalid params
    cv = CrossValidation()
    model_class = RandomForestRegressor
    params = "invalid type"
    with pytest.raises(TypeError):
        cv.add_model(model_class, True, params=params)


def test_add_model_requires_inner_cv_without_params_warns():
    # Test add_model method with skip_inner_cv=True and params provided
    cv = CrossValidation()
    model_class = RandomForestRegressor
    with pytest.warns(UserWarning):
        cv.add_model(model_class, True)


def test_add_model_requires_inner_cv_with_params_adds():
    # Test add_model method with skip_inner_cv=True and params provided
    cv = CrossValidation()
    model_class = RandomForestRegressor
    params = {"n_estimators": 100}
    cv.add_model(model_class, params=params)
    assert cv.config["mapping"][model_class.__name__]["requires_inner_cv"] == True


def test_add_mode_valid_callback():
    # Test add_model method with valid callbacks
    cv = CrossValidation()
    model_class = RandomForestRegressor
    callbacks = [MagicMock()]
    cv.add_model(model_class, callbacks=callbacks)
    assert cv.config["mapping"][model_class.__name__]["callbacks"] == {"callbacks": callbacks}


def test_add_model_invalid_callbacks():
    # Test add_model method with invalid callbacks
    cv = CrossValidation()
    with pytest.raises(TypeError):
        cv.add_model(LinearModel, model_name="LinearModel", callbacks=123)


def test_add_model_multiple_model_calls():
    # Test add_model with multiple calls
    cv = CrossValidation()
    model_class_rf = RandomForestRegressor
    model_class_lm = LinearModel
    model_class_xgb = XGBRegressor
    params_rf = {"n_estimators": [10, 100]}
    params_xgb = {"n_estimators": [100, 1000]}
    cv.add_model(model_class_rf, params=params_rf)
    cv.add_model(model_class_lm)
    cv.add_model(model_class_xgb, params=params_xgb)

    assert len(cv.config["mapping"]) == 3
    assert model_class_rf.__name__ in cv.config["mapping"]
    assert model_class_lm.__name__ in cv.config["mapping"]
    assert model_class_xgb.__name__ in cv.config["mapping"]

    assert isinstance(cv.config["mapping"][model_class_rf.__name__], ModelConfigDict)
    assert isinstance(cv.config["mapping"][model_class_lm.__name__], ModelConfigDict)
    assert isinstance(cv.config["mapping"][model_class_xgb.__name__], ModelConfigDict)
    assert cv.config["mapping"][model_class_rf.__name__]["model"] == model_class_rf
    assert (
        cv.config["mapping"][model_class_rf.__name__]["params"]["n_estimators"]
        == params_rf["n_estimators"]
    )

    assert cv.config["mapping"][model_class_lm.__name__]["model"] == model_class_lm
    with pytest.raises(KeyError):
        cv.config["mapping"][model_class_lm.__name__]["params"]["n_estimators"]

    assert cv.config["mapping"][model_class_xgb.__name__]["model"] == model_class_xgb
    assert (
        cv.config["mapping"][model_class_xgb.__name__]["params"]["n_estimators"]
        == params_xgb["n_estimators"]
    )


def test_perform_fit_callbacks():
    X, y, _, _ = DATA_TUPLE_3_25
    cv = CrossValidation()
    # make magic mock an instance of type TrainingCallback
    xgb_callback = MagicMock(spec=TrainingCallback)
    cv.add_model(XGBRegressor, callbacks=[xgb_callback])
    cv.set_data(X, y)
    cv.set_splits(n_splits_out=3, break_cross_val=True)
    # test if the callback fails in XGBoost with AssertionError with specific desc "before_training should return the model"
    with pytest.raises(AssertionError, match="before_training should return the model"):
        cv.perform()
