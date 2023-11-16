import numpy as np

from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.interface import CrossValidation
from flexcv.models import LinearModel

from data import DATA_TUPLE_3_25

def regression_with_summary():
    X, y, group, random_slopes = DATA_TUPLE_3_25

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "requires_inner_cv": False,
                    "model": LinearModel,
                    "params": {},
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_models(model_map)
        .set_splits(n_splits_out=3, break_cross_val=True)
        .perform()
        .get_results()
    )
    return results.summary


def test_summary():
    """Test if the summary of the results is correct."""
    check_value = regression_with_summary()
    mean_r2_lm = check_value.loc[("mean", "r2")].to_numpy()
    eps = np.finfo(float).eps
    ref_value = -0.09028321687241014
    assert np.isclose(mean_r2_lm[0], ref_value)
