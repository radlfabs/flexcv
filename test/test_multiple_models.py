import numpy as np
from sklearn.ensemble import RandomForestRegressor
import optuna

from flexcv import model_postprocessing as mp
from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.interface import ModelConfigDict
from flexcv.interface import ModelMappingDict
from flexcv.run import Run
from flexcv.models import LinearModel


def two_models_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2
    )
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                    "requires_formula": True,
                }
            ),
            "RandomForest": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "requires_formula": False,
                    "n_jobs_model": 1,
                    "n_jobs_cv": -1,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(5, 100),
                        "n_estimators": optuna.distributions.CategoricalDistribution(
                            [10]
                        ),
                    },
                    "post_processor": mp.rf_post,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_models(model_map)
        .set_inner_cv(3)
        .set_run(Run())
        .perform()
        .get_results()
    )

    n_values = len(results["LinearModel"]["metrics"])
    lm_r2_values = [results["LinearModel"]["metrics"][k]["r2"] for k in range(n_values)]
    rf_r2_values = [results["RandomForest"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(lm_r2_values), np.mean(rf_r2_values)


def test_two_models():
    """Checks that the two models (Linear and Random Forest) are performing as expected."""
    check_value_lm, check_value_rf = two_models_regression()
    eps = np.finfo(float).eps
    ref_value_lm = 0.4265339487499462
    assert (check_value_lm / ref_value_lm) > (1 - eps)
    assert (check_value_lm / ref_value_lm) < (1 + eps)
    ref_value_rf = 0.011843873652794202
    assert (check_value_rf / ref_value_rf) > (1 - eps)
    assert (check_value_rf / ref_value_rf) < (1 + eps)
