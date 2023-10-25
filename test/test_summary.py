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


def regression_with_summary():
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
    return results.summary

def test_summary():
    """Test if the summary of the results is correct."""
    check_value = regression_with_summary()
    mean_r2_lm = check_value.loc[("mean", "r2")].to_numpy()
    eps = np.finfo(float).eps
    assert (mean_r2_lm / 0.36535132545331933) > (1 - eps)
