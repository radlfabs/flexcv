import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
import flexcv.model_postprocessing as mp

from data import DATA_TUPLE_3_100


def merf_mixed_regression():
    X, y, group, random_slopes = DATA_TUPLE_3_100

    model_map = ModelMappingDict(
        {
            "RandomForest": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "n_jobs_model": -1,
                    "n_jobs_cv": -1,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(2, 10, step=2),
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
        .set_splits(n_splits_out=3, n_splits_in=3, break_cross_val=True)
        .set_merf(add_merf_global=True, em_max_iterations=3)
        .perform()
        .get_results()
    )

    return np.mean(results["MERF(RandomForest)"]["folds_by_metrics"]["r2"])


def test_merf_rf():
    """Test if the mean r2 value of the random forest regression is is exactly the same over time."""
    assert np.isclose([merf_mixed_regression()], [-0.007246874039440909])


def merf_mixed_xgboost():
    X, y, group, random_slopes = DATA_TUPLE_3_100

    model_map = ModelMappingDict(
        {
            "XGBoost": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "n_jobs_model": -1,
                    "n_jobs_cv": -1,
                    "model": XGBRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(2, 20, step=5),
                        "n_estimators": optuna.distributions.CategoricalDistribution([10]),
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
        .set_splits(n_splits_out=3, n_splits_in=3, break_cross_val=True)
        .set_merf(add_merf_global=True, em_max_iterations=3)
        .perform()
        .get_results()
    )

    return np.mean(results["MERF(XGBoost)"]["folds_by_metrics"]["r2"])


def test_merf_xgboost():
    """Test if the mean r2 value of the random forest regression is is exactly the same over time."""
    assert np.isclose([merf_mixed_xgboost()], [0.17332563312329563])


def merf_svr_regression():
    X, y, group, random_slopes = DATA_TUPLE_3_100

    model_map = ModelMappingDict(
        {
            "SVR": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "allows_n_jobs": False,
                    "allows_seed": False,
                    "model": SVR,
                    "params": {
                        "C": optuna.distributions.FloatDistribution(
                            0.1, 10,
                        ),
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
        .set_splits(n_splits_out=3, n_splits_in=3, break_cross_val=True)
        .set_merf(add_merf_global=True, em_max_iterations=3)
        .perform()
        .get_results()
    )

    return np.mean(results["MERF(SVR)"]["folds_by_metrics"]["r2"])


def test_merf_svr_mixed():
    """Test if the mean r2 value of the random forest regression is exactly the same over time."""
    assert np.isclose([merf_svr_regression()], [-0.11442446839255993])
