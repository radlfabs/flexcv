import pandas as pd
import numpy as np
import neptune
import optuna
from sklearn.ensemble import RandomForestRegressor

from flexcv.interface import CrossValidation
from flexcv.synthesizer import generate_regression
from flexcv.model_mapping import ModelMappingDict, ModelConfigDict
from flexcv.model_postprocessing import RandomForestModelPostProcessor

from data import DATA_TUPLE_10_100, DATA_TUPLE_3_25, DATA_TUPLE_3_100

def neptrune_rf_regression(test_run_name):
    X, y, group, random_slopes = DATA_TUPLE_3_25

    model_map = ModelMappingDict(
        {
            "RandomForest": ModelConfigDict(
                {
                    "requires_inner_cv": True,
                    "model": RandomForestRegressor,
                    "params": {
                        "max_depth": optuna.distributions.IntDistribution(5, 100),
                        "n_estimators": optuna.distributions.CategoricalDistribution(
                            [10]
                        ),
                    },
                    "post_processor": RandomForestModelPostProcessor,
                    "add_merf": True,
                }
            ),
        }
    )
    nep_run = neptune.init_run(
        custom_run_id=test_run_name, project="radlfabs/flexcv-testing"
    )
    cv = CrossValidation()
    _ = (
        cv.set_data(X, y, group, random_slopes)
        .set_models(model_map)
        .set_inner_cv(3)
        .set_splits(n_splits_out=3, n_splits_in=3, break_cross_val=True)
        .set_merf(add_merf_global=True, em_max_iterations=3)
        .set_run(run=nep_run)
        .perform()
        .get_results()
    )
    # get neptune id
    nep_run_id = nep_run["sys/id"].fetch()
    nep_run.stop()
    return nep_run_id


def test_neptune_logging():
    test_run_name = "SeriouslyTestRun"
    run_id = neptrune_rf_regression(test_run_name)
    run = neptune.init_run(
        with_id=run_id,
        project="radlfabs/flexcv-testing",
        mode="read-only",
    )
    run.stop()
