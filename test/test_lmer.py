import numpy as np
import pandas as pd

from flexcv.data_generation import generate_regression
from flexcv.funcs import empty_func
from flexcv.interface_functional import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearMixedEffectsModel, LinearModel
from flexcv.run import Run


def simple_regression():

    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "inner_cv": False,
                    "n_trials": 100,
                    "n_jobs_model": {"n_jobs": 1},
                    "n_jobs_cv": 1,
                    "model": LinearModel,
                    "params": {},
                    "post_processor": empty_func,
                    "mixed_model": LinearMixedEffectsModel,
                    "mixed_post_processor": empty_func,
                    "mixed_name": "MixedLM",
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_splits(n_splits_out=3)
        .set_models(model_map)
        .set_mixed_effects(True)
        .set_run(Run())
        .perform()
        .get_results()
    )
    
    n_values = len(results["MixedLM"]["metrics"])
    r2_values = [results["MixedLM"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_lmer_regression_k3():
    check_value = simple_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.31799553543275216) > (1 - eps)
