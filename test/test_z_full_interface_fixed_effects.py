import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import flexcv.model_postprocessing as mp
from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.interface import ModelConfigDict
from flexcv.interface import ModelMappingDict
from flexcv.run import Run
from flexcv.models import LinearModel
from flexcv.model_postprocessing import (
    LinearModelPostProcessor,
    LMERModelPostProcessor,
    RandomForestModelPostProcessor,
)


def simple_regression():
    X, y, _, _ = generate_regression(
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
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y).set_models(model_map).set_run(Run()).perform().get_results()
    )

    return np.mean(results["LinearModel"]["folds_by_metrics"]["r2"])


def set_splits_input_kfold_with_linear_model():
    X, y, _, _ = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                    "requires_formula": True,
                    "post_processor": LinearModelPostProcessor,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y)
        .set_models(model_map)
        .set_splits(kfold)
        .set_run(Run())
        .perform()
        .get_results()
    )

    return np.mean(results["LinearModel"]["folds_by_metrics"]["r2"])


def test_set_splits_input_kfold_with_linear_model():
    assert np.isclose(
        [set_splits_input_kfold_with_linear_model()], [0.4265339487499462]
    )


def random_forest_regression():
    X, y, _, _ = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )

    model_map = ModelMappingDict(
        {
            "RandomForest": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "requires_formula": False,
                    "n_jobs_model": -1,
                    "n_jobs_cv": -1,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(5, 100),
                        "n_estimators": optuna.distributions.CategoricalDistribution(
                            [10]
                        ),
                    },
                    "n_trials": 3,
                    "post_processor": RandomForestModelPostProcessor,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y)
        .set_models(model_map)
        .set_splits(n_splits_out=3)
        .perform()
        .get_results()
    )
    return np.mean(results["RandomForest"]["folds_by_metrics"]["r2"])


def test_randomforest_regression_fixed():
    """Test if the mean r2 value of the random forest regression is exactly the same over time."""
    assert np.isclose([random_forest_regression()], [0.07334709720199191])
