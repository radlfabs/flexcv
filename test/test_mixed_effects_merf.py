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


def test_merf_rf():
    """Test if the mean r2 value of the random forest regression is correct."""
    check_value = merf_mixed_regression()
    eps = np.finfo(float).eps
    ref_value = 0.11496675402377587
    assert (check_value / ref_value) > (1 - eps)
    assert (check_value / ref_value) < (1 + eps)


def merf_mixed_xgboost():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2
    )

    model_map = ModelMappingDict(
        {
        "XGBoost": ModelConfigDict(
                    {
                        "requires_inner_cv": True,
                        "allows_seed": True,
                        "n_jobs_model": -1,
                        "n_jobs_cv": -1,
                        "model": XGBRegressor,
                        "params": {
                            "max_depth": optuna.distributions.IntDistribution(2, 700),
                        },
                        "post_processor": mp.xgboost_post,
                        "mixed_model": MERF,
                        "mixed_post_processor": mp.expectation_maximation_post,
                        "mixed_name": "XGBEM",
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
    n_values = len(results["XGBEM"]["metrics"])
    r2_values = [results["XGBEM"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_merf_xgboost():
    """Test if the mean r2 value of the random forest regression is correct."""
    check_value = merf_mixed_xgboost()
    eps = np.finfo(float).eps
    ref_value = -0.045098611366661046
    assert (check_value / ref_value) > (1 - eps)
    assert (check_value / ref_value) < (1 + eps)


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
    ref_value = -0.17212212996671483
    assert (check_value / ref_value) > (1 - eps)
    assert (check_value / ref_value) < (1 + eps)


def merf_svr_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2
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
                "post_processor": mp.svr_post,
                "mixed_model": MERF,
                "mixed_post_processor": mp.expectation_maximation_post,
                "mixed_name": "SVREM",
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
    n_values = len(results["SVREM"]["metrics"])
    r2_values = [results["SVREM"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_merf_svr_mixed():
    """Test if the mean r2 value of the random forest regression is correct."""
    check_value = merf_svr_regression()
    eps = np.finfo(float).eps
    ref_value = 0.18189191135516083
    assert (check_value / ref_value) > (1 - eps)
    assert (check_value / ref_value) < (1 + eps)
