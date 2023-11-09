import numpy as np

from flexcv.synthesizer import generate_regression
from flexcv.utilities import empty_func
from flexcv.interface import CrossValidation
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearMixedEffectsModel, LinearModel
from flexcv.run import Run


def lmer_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2
    )

    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "requires_inner_cv": False,
                    "requires_formula": True,
                    "model": LinearModel,
                    "post_processor": empty_func,
                    "mixed_model": LinearMixedEffectsModel,
                    "mixed_post_processor": empty_func,
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
        .set_mixed_effects(True)
        .set_run(Run())
        .perform()
        .get_results()
    )

    n_values = len(results["MixedLM"]["metrics"])
    r2_values = [results["MixedLM"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_linear_mixed_effects():
    check_value = lmer_regression()
    eps = np.finfo(float).eps
    ref_value = 0.33402132305208826
    assert (check_value / ref_value) > (1 - eps)
    assert (check_value / ref_value) < (1 + eps)
