import pytest
from sklearn.base import BaseEstimator
from optuna.distributions import IntDistribution

from flexcv.model_mapping import ModelConfigDict, map_backwards
from flexcv.utilities import empty_func


def test_model_config_dict_init():
    # Test initialization
    config = ModelConfigDict()
    assert isinstance(config, ModelConfigDict)
    assert config["requires_inner_cv"] == False
    assert config["requires_formula"] == False
    assert config["allows_seed"] == True
    assert config["allows_n_jobs"] == True
    assert config["n_trials"] == 100
    assert config["n_jobs_model"] == 1
    assert config["n_jobs_cv"] == 1
    assert config["params"] == {}
    assert config["post_processor"] == empty_func

def test_model_config_dict_custom_config():
    # Test initialization with custom configuration
    custom_config = {
        "requires_inner_cv": True,
        "n_trials": 50,
        "n_jobs_model": 2,
        "n_jobs_cv": 2,
        "model": BaseEstimator,
        "params": {"param": IntDistribution(1, 10)},
        "post_processor": lambda x: x,
        "mixed_model": BaseEstimator,
        "mixed_post_processor": lambda x: x,
        "mixed_name": "MixedModel"
    }
    defaults = ModelConfigDict({})
    custom_config = custom_config | defaults
    assert isinstance(custom_config, dict)
    config = ModelConfigDict(custom_config)
    assert set(config.keys()) == set(custom_config.keys())
    for key in custom_config:
        assert config[key] == custom_config[key]

def test_model_config_dict_has_key():
    # Test _has_key method
    config = ModelConfigDict()
    assert config._has_key("requires_inner_cv") == True
    assert config._has_key("nonexistent_key") == False

def test_model_config_dict_check_key_set_default():
    # Test _check_key_set_default method
    config = ModelConfigDict()
    config._check_key_set_default("new_key", "new_value")
    assert config["new_key"] == "new_value"
    config._check_key_set_default("requires_inner_cv", True)
    assert config["requires_inner_cv"] == False  # should not change the existing value
    
def test_map_backwards_valid_mapping():
    # Test map_backwards function with valid mapping
    mapping = {
        "Model1": {"mixed_name": "MixedModel1"},
        "Model2": {"mixed_name": "MixedModel2"},
        "Model3": {"mixed_name": "MixedModel3"}
    }
    expected_result = {
        "MixedModel1": "Model1",
        "MixedModel2": "Model2",
        "MixedModel3": "Model3"
    }
    result = map_backwards(mapping)
    assert result == expected_result

def test_map_backwards_empty_mapping():
    # Test map_backwards function with empty mapping
    mapping = {}
    expected_result = {}
    result = map_backwards(mapping)
    assert result == expected_result

def test_map_backwards_no_mixed_name():
    # Test map_backwards function with mapping without mixed_name
    mapping = {
        "Model1": {"mixed_name": "MixedModel1"},
        "Model2": {}
    }
    with pytest.raises(KeyError):
        map_backwards(mapping)