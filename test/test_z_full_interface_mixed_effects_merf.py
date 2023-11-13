import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.run import Run
import flexcv.model_postprocessing as mp
from flexcv.merf import MERF
from flexcv.models import EarthRegressor


def merf_mixed_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )

    model_map = ModelMappingDict(
        {
            "RandomForest": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "n_jobs_model": -1,
                    "n_jobs_cv": -1,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(5, 100),
                        "n_estimators": optuna.distributions.CategoricalDistribution(
                            [10]
                        ),
                    },
                    "post_processor": mp.RandomForestModelPostProcessor,
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
        .set_merf(add_merf_global=True, em_max_iterations=5)
        .perform()
        .get_results()
    )

    return np.mean(results["MERF(RandomForest)"]["folds_by_metrics"]["r2"])


def test_merf_rf():
    """Test if the mean r2 value of the random forest regression is is exactly the same over time."""
    assert np.isclose([merf_mixed_regression()], [0.0832993979557632])


def merf_mixed_xgboost():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )

    model_map = ModelMappingDict(
        {
        "XGBoost": ModelConfigDict(
                    {
                        "requires_inner_cv": True,
                        "n_jobs_model": -1,
                        "n_jobs_cv": -1,
                        "model": XGBRegressor,
                        "params": {
                            "max_depth": optuna.distributions.IntDistribution(2, 700),
                        },
                        "post_processor": mp.XGBoostModelPostProcessor,
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
        .set_merf(add_merf_global=True, em_max_iterations=5)
        .perform()
        .get_results()
    )

    return np.mean(results["MERF(XGBoost)"]["folds_by_metrics"]["r2"])


def test_merf_xgboost():
    """Test if the mean r2 value of the random forest regression is is exactly the same over time."""
    assert np.isclose(
        [merf_mixed_xgboost()],
        [-0.12340731619124896]
        )


def merf_earth_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
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
                    "post_processor": mp.EarthModelPostProcessor,
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
        .set_merf(add_merf_global=True, em_max_iterations=5)
        .perform()
        .get_results()
    )

    return np.mean(results["MERF(Earth)"]["folds_by_metrics"]["r2"])


def test_merf_earth():
    """Test if the mean r2 value of the random forest regression is exactly the same over time."""
    assert np.isclose([merf_earth_regression()], [0.03308048396566532])


def merf_svr_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )

    model_map = ModelMappingDict(
        {
            "SVR": ModelConfigDict(
            {
                "requires_inner_cv": True,
                "allows_n_jobs": False,
                "allows_seed": False,
                "model": SVR,
                "params": {
                    "C": optuna.distributions.FloatDistribution(0.001, 50, log=True),
                },
                "post_processor": mp.SVRModelPostProcessor,
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
        .set_merf(add_merf_global=True, em_max_iterations=5)
        .perform()
        .get_results()
    )

    return np.mean(results["MERF(SVR)"]["folds_by_metrics"]["r2"])


def test_merf_svr_mixed():
    """Test if the mean r2 value of the random forest regression is exactly the same over time."""
    assert np.isclose([merf_svr_regression()], [0.2880957829764944])
