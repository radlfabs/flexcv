import numpy as np
import optuna

from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.run import Run
import flexcv.model_postprocessing as mp
from flexcv.merf import MERF
from flexcv.models import EarthRegressor


def merf_earth_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2
    )

    model_map = ModelMappingDict(
        {
            "Earth": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "allows_n_jobs": False,
                    "allows_seed": True,
                    "model": EarthRegressor,
                    "params": {  # 'degree', 'endspan', 'fast_beta', 'fast_k', 'minspan', 'newvar_penalty', 'nk', 'nprune', 'pmethod', 'random_state', 'thresh'
                        "degree": optuna.distributions.IntDistribution(1, 5),
                    },
                    "post_processor": mp.mars_post,
                    "mixed_model": MERF,
                    "mixed_post_processor": mp.expectation_maximation_post,
                    "mixed_name": "EarthEM",
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
    n_values = len(results["EarthEM"]["metrics"])
    r2_values = [results["EarthEM"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_merf_earth():
    """Test if the mean r2 value of the random forest regression is correct."""
    check_value = merf_earth_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.07334709720199191) > (1 - eps)
