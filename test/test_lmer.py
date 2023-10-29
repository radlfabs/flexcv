import numpy as np
import pandas as pd

from flexcv.data_generation import generate_regression
from flexcv.funcs import empty_func
from flexcv.interface import (CrossValConfigurator, CrossValidation,
                              DataConfigurator, ModelConfigDict,
                              ModelMappingDict, RunConfigurator)
from flexcv.models import LinearMixedEffectsModel, LinearModel
from flexcv.run import Run


def simple_regression():

    dummy_run = Run()
    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "inner_cv": False,
                    "n_trials": 100,
                    "n_jobs_model": {"n_jobs": 1},
                    "n_jobs_cv": 1,
                    "model": LinearModel,
                    "params": {},
                    "post_processor": empty_func,
                    "mixed_model": LinearMixedEffectsModel,
                    "mixed_post_processor": empty_func,
                    "mixed_name": "MixedLM",
                }
            ),
        }
    )

    data_config = DataConfigurator(
        dataset_name="random_example",
        model_level="mixed",
        target_name=y.name,
        X=X,
        y=y,
        group=group,
        slopes=random_slopes,
    )

    cv_config = CrossValConfigurator(
        n_splits=3,
    )

    run_config = RunConfigurator(run=dummy_run)

    cv = CrossValidation(
        data_config=data_config,
        cross_val_config=cv_config,
        run_config=run_config,
        model_mapping=model_map,
    )

    results = cv.perform()
    n_values = len(results["MixedLM"]["metrics"])
    r2_values = [results["MixedLM"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_lmer_regression_k3():
    check_value = simple_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.31799553543275216) > (1 - eps)
