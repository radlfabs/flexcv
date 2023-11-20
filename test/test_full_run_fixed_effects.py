import numpy as np
from flexcv._data import DATA_TUPLE_3_25

from flexcv.interface import CrossValidation
from flexcv.models import LinearModel


def set_splits_input_kfold_with_linear_model():
    X, y, _, _ = DATA_TUPLE_3_25

    cv = CrossValidation()
    results = (
        cv.set_data(X, y)
        .add_model(LinearModel)
        .set_splits("KFold", n_splits_out=3)
        .perform()
        .get_results()
    )

    return np.mean(results["LinearModel"]["folds_by_metrics"]["r2"])


def test_set_splits_input_kfold_with_linear_model():
    set_splits_input_kfold_with_linear_model()
