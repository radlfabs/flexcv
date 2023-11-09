import numpy as np

from flexcv.synthesizer import generate_regression
from flexcv.utilities import empty_func
from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearMixedEffectsModel, LinearModel
from flexcv.run import Run


def lmer_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42,
    )

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "requires_inner_cv": False,
                    "requires_formula": True,
                    "model": LinearModel,
                    "mixed_model": LinearMixedEffectsModel,
                    "mixed_name": "MixedLM",
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_splits(n_splits_out=3)
        .set_models(model_map)
        .set_mixed_effects(model_mixed_effects=True)
        .set_run(run=Run())
        .perform()
        .get_results()
    )

    return np.mean(results["MixedLM"]["folds_by_metrics"]["r2"])


def test_linear_mixed_effects():
    assert np.isclose([lmer_regression()], [0.3331408486407139])
