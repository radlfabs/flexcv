import numpy as np
from flexcv._data import DATA_TUPLE_3_25

from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.model_postprocessing import LMERModelPostProcessor
from flexcv.models import LinearMixedEffectsModel


def lmer_regression():
    X, y, group, random_slopes = DATA_TUPLE_3_25

    model_map = ModelMappingDict(
        {
            "LMER": ModelConfigDict(
                {
                    "requires_inner_cv": False,
                    "model": LinearMixedEffectsModel,
                    "post_processor": LMERModelPostProcessor,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_splits(n_splits_out=3, n_splits_in=3, break_cross_val=True)
        .set_models(model_map)
        .set_lmer(predict_known_groups_lmm=True)
        .perform()
        .get_results()
    )

    return np.mean(results["LMER"]["folds_by_metrics"]["r2"])


def test_linear_mixed_effects():
    lmer_regression()
