import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

from flexcv.data_generation import generate_regression
from flexcv.interface import DataConfigurator
from flexcv.interface import CrossValConfigurator
from flexcv.interface import RunConfigurator
from flexcv.interface import CrossValidation
from flexcv.interface import ModelConfigDict
from flexcv.interface import ModelMappingDict
from flexcv.interface import OptimizationConfigurator
from flexcv.run import Run
from flexcv.models import LinearModel
from flexcv.models import LinearMixedEffectsModel
import flexcv.model_postprocessing as mp


def random_forest_regression():

    dummy_run = Run()
    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)

    def empty_func(*args, **kwargs):
        pass

    model_map = ModelMappingDict({
        "RandomForest": ModelConfigDict({
            # https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
            "inner_cv": True,
            "n_trials": 400,
            "n_jobs_model": {"n_jobs": -1},
            "n_jobs_cv": -1,
            "model": RandomForestRegressor,
            "params": {
                #    https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/ 
                "max_depth": optuna.distributions.IntDistribution(5, 100), 
                # "min_samples_split": optuna.distributions.IntDistribution(2, 1000, log=True),
                # "min_samples_leaf": optuna.distributions.IntDistribution(2, 5000, log=True), 
                # "max_samples": optuna.distributions.FloatDistribution(.0021, 0.9), 
                # "max_features": optuna.distributions.IntDistribution(1, 10),
                # "max_leaf_nodes": optuna.distributions.IntDistribution(10, 40000), 
                # "min_impurity_decrease": optuna.distributions.FloatDistribution(1e-8, 0.02, log=True), # >>>> can be (1e-8, .01, log=True)
                # "min_weight_fraction_leaf": optuna.distributions.FloatDistribution(0, .5), # must be a float in the range [0.0, 0.5]
                # "ccp_alpha": optuna.distributions.FloatDistribution(1e-8, 0.01), 
                # "n_estimators": optuna.distributions.IntDistribution(2, 7000),
            },
            "post_processor": mp.rf_post,
            # "level_4_model": MERF,
            # "level_4_post_processor": mp.expectation_maximation_post,
            # "level_4_name": "MERF",
        }),
    })
        
    data_config = DataConfigurator(
        dataset_name="random_example",
        model_level="fixed_only",
        target_name=y.name,
        X=X,
        y=y,
        group=group,
        slopes=random_slopes,
    )

    cv_config = CrossValConfigurator(
        n_splits=3,
    )

    run_config = RunConfigurator(
        run=dummy_run
    )

    optim_config = OptimizationConfigurator(
        n_trials=3,
    )

    cv = CrossValidation(
        data_config=data_config,
        cross_val_config=cv_config,
        run_config=run_config,
        model_mapping=model_map,
        optim_config=optim_config,
    )

    results = cv.perform()
    n_values = len(results["RandomForest"]["metrics"])
    r2_values = [results["RandomForest"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_randomforest_regression_fixed():
    """Test if the mean r2 value of the random forest regression is correct."""
    check_value = random_forest_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.10485552017344309) > (1 - eps)
