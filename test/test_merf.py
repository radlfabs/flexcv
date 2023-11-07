import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.run import Run
import flexcv.model_postprocessing as mp
from flexcv.merf import MERF

def merf_mixed_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2
    )

    model_map = ModelMappingDict(
        {
            "RandomForest": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "requires_formula": False,
                    "allows_seed": True,
                    "allows_n_jobs": True,
                    "n_jobs_model": -1,
                    "n_jobs_cv": -1,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(5, 100),
                        "n_estimators": optuna.distributions.CategoricalDistribution(
                            [10]
                        ),
                    },
                    "mixed_model": MERF,
                    "post_processor": mp.rf_post,
                    "mixed_post_processor": mp.expectation_maximation_post,
                    "mixed_name": "MERF",
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_models(model_map)
        .set_inner_cv(3)
        .set_splits(n_splits_out=3)
        .set_run(Run(), random_seed=42)
        .set_mixed_effects(True, 25)
        .perform()
        .get_results()
    )
    summary = results.summary
    n_values = len(results["MERF"]["metrics"])
    r2_values = [results["MERF"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_merf_mixed():
    """Test if the mean r2 value of the random forest regression is correct."""
    check_value = merf_mixed_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.07334709720199191) > (1 - eps)


test_merf_mixed()