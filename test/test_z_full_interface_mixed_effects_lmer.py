import numpy as np

from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearMixedEffectsModel
from flexcv.run import Run
from flexcv.model_postprocessing import LMERModelPostProcessor


def lmer_regression():
    X, y, group, random_slopes = generate_regression(
        10,
        100,
        n_slopes=1,
        noise_level=9.1e-2,
        random_seed=42,
    )

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
        .set_splits(n_splits_out=3)
        .set_models(model_map)
        .set_lmer(predict_known_groups_lmm=True)
        .set_run(run=Run())
        .perform()
        .get_results()
    )

    return np.mean(results["LMER"]["folds_by_metrics"]["r2"])


def test_linear_mixed_effects():
    assert np.isclose([lmer_regression()], [0.3331408486407139])
