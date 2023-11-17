import numpy as np
import optuna
from data import DATA_TUPLE_3_25
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from flexcv.interface import CrossValidation, ModelConfigDict, ModelMappingDict
from flexcv.model_postprocessing import (
    LinearModelPostProcessor,
    RandomForestModelPostProcessor,
)
from flexcv.models import LinearModel
from flexcv.run import Run


def set_splits_input_kfold_with_linear_model():
    X, y, _, _ = DATA_TUPLE_3_25

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
        .set_splits(kfold, n_splits_out=3)
        .set_run(Run())
        .perform()
        .get_results()
    )

    return np.mean(results["LinearModel"]["folds_by_metrics"]["r2"])


def test_set_splits_input_kfold_with_linear_model():
    assert np.isclose(
        [set_splits_input_kfold_with_linear_model()], [-0.7596695802234839]
    )
