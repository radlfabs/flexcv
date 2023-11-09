from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from flexcv.repeated import init_repeated_runs
from flexcv.synthesizer import generate_regression
from flexcv.models import LinearModel
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.repeated import RepeatedCV
from flexcv.repeated import try_mean
from flexcv.repeated import aggregate_, try_mean


def test_try_mean_numeric_values():
    # Test try_mean function with numeric values
    x = np.array([1, 2, 3, 4, 5])
    result = try_mean(x)
    assert result == np.mean(x)

def test_try_mean_nan_values():
    # Test try_mean function with "NaN" values
    x = np.array(["NaN", "NaN", "NaN"])
    result = try_mean(x)
    assert result == -99

def test_try_mean_mixed_values():
    # Test try_mean function with mixed "NaN" and numeric values
    x = np.array([1, 2, "NaN"])
    result = try_mean(x)
    assert result == -999

def test_try_mean_empty_array():
    # Test try_mean function with an empty array
    x = np.array([])
    result = try_mean(x)
    assert np.isnan(result)

def run_repeated(seeds):
    # make sample data
    X, y, _, _ = generate_regression(10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42)

    # create a model mapping
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                    "requires_formula": True,
                }
            ),
        }
    )

    rcv = (
        RepeatedCV()
        .set_data(X, y, dataset_name="ExampleData")
        .set_models(model_map)
        .set_n_repeats(3)
        .set_seeds(seeds)
        .perform()
        .get_results()
    )

    return rcv.summary

def test_repeated_equal_seeds():
    """Tests that the standard deviation of the R2 over runs with equal seeds is zero."""
    rtn_value = run_repeated([42, 42])
    assert rtn_value.loc["r2_std", "LinearModel"] == 0.0


def test_repeated_different_seeds():
    """Test that the standard deviation of the R2 over runs with different seeds is not zero."""
    rtn_value = run_repeated([42, 43])
    assert rtn_value.loc["r2_std", "LinearModel"] != 0.0

# TODO add more repeated tests for the aggregation stuff?