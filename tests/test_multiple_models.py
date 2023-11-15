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
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
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
                    "n_jobs_model": 1,
                    "n_jobs_cv": 1,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(5, 100),
                        "n_estimators": optuna.distributions.CategoricalDistribution(
                            [10]
                        ),
                    },
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

    return (
        np.mean([results["LinearModel"]["metrics"][k]["r2"] for k in range(3)]),
        np.mean([results["RandomForest"]["metrics"][k]["r2"] for k in range(3)]),
    )


def test_two_models():
    """Checks that the two models (Linear and Random Forest) are performing as expected."""
    check_value_lm, check_value_rf = two_models_regression()
    eps = np.finfo(float).eps
    ref_value_lm = 0.39345711499831665
    assert np.isclose([check_value_lm], [ref_value_lm])
    ref_value_rf = -0.11238529902680768
    assert np.isclose([check_value_rf], [ref_value_rf])
