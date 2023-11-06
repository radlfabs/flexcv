import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.run import Run
import flexcv.model_postprocessing as mp


def random_forest_regression():
    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise_level=9.1e-2)

    model_map = ModelMappingDict(
        {
            "RandomForest": ModelConfigDict(
                {
                    "inner_cv": True,
                    "n_jobs_model": {"n_jobs": -1},
                    "n_jobs_cv": -1,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(5, 100),
                    },
                    "post_processor": mp.rf_post,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y)
        .set_models(model_map)
        .set_inner_cv(3)
        .set_splits(n_splits_out=3)
        .set_run(Run())
        .perform()
        .get_results()
    )
    n_values = len(results["RandomForest"]["metrics"])
    r2_values = [results["RandomForest"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_randomforest_regression_fixed():
    """Test if the mean r2 value of the random forest regression is correct."""
    check_value = random_forest_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.10485552017344309) > (1 - eps)
