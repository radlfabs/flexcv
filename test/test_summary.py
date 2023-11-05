import numpy as np

from flexcv.synthesizer import generate_regression
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.interface import CrossValidation
from flexcv.run import Run
from flexcv.models import LinearModel
from flexcv.utilities import empty_func


def regression_with_summary():
    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "inner_cv": False,
                    "n_jobs_model": {"n_jobs": 1},
                    "model": LinearModel,
                    "params": {},
                    "post_processor": empty_func,
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
    assert (mean_r2_lm / 0.36535132545331933) > (1 - eps)
