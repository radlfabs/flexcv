import numpy as np
from data import DATA_TUPLE_3_25

from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearModel
from flexcv.repeated import RepeatedCV, try_mean


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
    X, y, _, _ = DATA_TUPLE_3_25

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
        .set_splits(n_splits_in=3, n_splits_out=3, break_cross_val=True)
        .set_n_repeats(2)
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
