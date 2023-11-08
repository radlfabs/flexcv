from flexcv.synthesizer import generate_regression
from flexcv.models import LinearModel
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.repeated import RepeatedCV


def run_repeated(seeds):
    # make sample data
    X, y, _, _ = generate_regression(10, 100, n_slopes=1, noise_level=9.1e-2)

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
    assert rtn_value.loc["r2_std", "LinearModel"] > 0.0
