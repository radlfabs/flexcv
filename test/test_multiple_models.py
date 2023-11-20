import pandas as pd
from flexcv._data import DATA_TUPLE_3_25

from flexcv.interface import CrossValidation, ModelConfigDict, ModelMappingDict
from flexcv.models import LinearMixedEffectsModel, LinearModel
from flexcv.run import Run


def two_models_regression():
    X, y, group, random_slopes = DATA_TUPLE_3_25

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                    "requires_formula": True,
                }
            ),
            "LMER": ModelConfigDict(
                {
                    "model": LinearMixedEffectsModel,
                    "requires_formula": True,
                }
            ),
        }
    )

    cv = CrossValidation()
    result = (
        cv.set_data(X, y, groups=group, slopes=random_slopes)
        .set_splits(n_splits_in=3, n_splits_out=3, break_cross_val=True)
        .set_models(model_map)
        .perform()
        .get_results()
    )
    return result


def test_two_models():
    """Checks that the two models (Linear and Random Forest) are performing as expected."""
    result = two_models_regression()
    assert "LinearModel" in result
    assert "LMER" in result
    assert isinstance(result.summary, pd.DataFrame)
