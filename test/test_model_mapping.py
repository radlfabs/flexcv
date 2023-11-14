import pytest
from sklearn.base import BaseEstimator

from flexcv.model_mapping import ModelConfigDict, ModelMappingDict


def test_model_config_dict_init_empty():
    # Test initialization with no arguments
    config = ModelConfigDict()
    assert config["requires_inner_cv"] == False
    assert config["n_jobs_model"] == -1
    assert config["n_jobs_cv"] == -1
    assert config["params"] == {}


def test_model_config_dict_init_with_mapping():
    # Test initialization with a mapping
    mapping = {
        "requires_inner_cv": True,
        "n_trials": 50,
        "n_jobs_model": 2,
        "n_jobs_cv": 2,
        "params": {},
    }
    config = ModelConfigDict(mapping)
    assert config["requires_inner_cv"] == True
    assert config["n_trials"] == 50
    assert config["n_jobs_model"] == 2
    assert config["n_jobs_cv"] == 2
    assert config["params"] == {}


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


def test_model_mapping_dict_init_empty():
    # Test initialization with no arguments
    mapping = ModelMappingDict()
    assert len(mapping) == 0


def test_model_mapping_dict_init_with_mapping():
    # Test initialization with a mapping
    model_config = ModelConfigDict(
        {
            "requires_inner_cv": True,
            "n_trials": 50,
            "n_jobs_model": 2,
            "n_jobs_cv": 2,
            "params": {},
        }
    )
    mapping = ModelMappingDict({"Model1": model_config})
    assert len(mapping) == 1
    assert "Model1" in mapping
    assert isinstance(mapping["Model1"], ModelConfigDict)
    assert mapping["Model1"]["requires_inner_cv"] == True
    assert mapping["Model1"]["n_trials"] == 50
    assert mapping["Model1"]["n_jobs_model"] == 2
    assert mapping["Model1"]["n_jobs_cv"] == 2
    assert mapping["Model1"]["params"] == {}


def test_model_mapping_dict_set_item():
    # Test __setitem__ method
    mapping = ModelMappingDict()
    model_config = ModelConfigDict(
        {
            "requires_inner_cv": True,
            "n_trials": 50,
            "n_jobs_model": 2,
            "n_jobs_cv": 2,
            "params": {},
        }
    )
    mapping["Model1"] = model_config
    assert len(mapping) == 1
    assert "Model1" in mapping
    assert isinstance(mapping["Model1"], ModelConfigDict)
    assert mapping["Model1"]["requires_inner_cv"] == True
    assert mapping["Model1"]["n_trials"] == 50
    assert mapping["Model1"]["n_jobs_model"] == 2
    assert mapping["Model1"]["n_jobs_cv"] == 2
    assert mapping["Model1"]["params"] == {}
