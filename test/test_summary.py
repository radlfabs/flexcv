import numpy as np

from flexcv.synthesizer import generate_regression
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.interface import CrossValidation
from flexcv.run import Run
from flexcv.models import LinearModel


def regression_with_summary():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "requires_inner_cv": False,
                    "requires_formula": True,
                    "n_jobs_model": 1,
                    "model": LinearModel,
                    "params": {},
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_models(model_map)
        .set_run(Run())
        .perform()
        .get_results()
    )
    return results.summary


def test_summary():
    """Test if the summary of the results is correct."""
    check_value = regression_with_summary()
    mean_r2_lm = check_value.loc[("mean", "r2")].to_numpy()
    eps = np.finfo(float).eps
    ref_value = 0.4265339487499462
    assert np.isclose(mean_r2_lm[0], ref_value)
    assert np.isclose(mean_r2_lm[0], ref_value)
