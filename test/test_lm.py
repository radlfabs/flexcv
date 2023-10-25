import pandas as pd
import numpy as np

from flexcv.data_generation import generate_regression
from flexcv.interface import DataConfigurator
from flexcv.interface import CrossValConfigurator
from flexcv.interface import RunConfigurator
from flexcv.interface import CrossValidation
from flexcv.interface import ModelConfigDict
from flexcv.interface import ModelMappingDict
from flexcv.run import Run
from flexcv.models import LinearModel
from flexcv.models import LinearMixedEffectsModel


def simple_regression():

    dummy_run = Run()
    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)

    def empty_func(*args, **kwargs):
        pass

    model_map = ModelMappingDict({
        "LinearModel": ModelConfigDict({
            "inner_cv": False,
            "n_jobs_model": {"n_jobs": 1},
            "model": LinearModel,
            "params": {},
            "post_processor": empty_func,
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

    cv = CrossValidation(
        data_config=data_config,
        cross_val_config=cv_config,
        run_config=run_config,
        model_mapping=model_map,
    )

    results = cv.perform()
    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_lm_fixed_regression_k3():
    check_value = simple_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.36535132545331933) > (1 - eps)
