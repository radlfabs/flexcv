import numpy as np

from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.interface import ModelConfigDict
from flexcv.interface import ModelMappingDict
from flexcv.run import Run
from flexcv.models import LinearModel


def simple_regression():
    X, y, _, _ = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                    "requires_formula": True,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y)
        .set_models(model_map)
        .set_run(Run())
        .perform()
        .get_results()
    )

    return np.mean(results["LinearModel"]["folds_by_metrics"]["r2"])


def test_linear_model():
    assert np.isclose([simple_regression()], [0.4265339487499462]) < np.finfo(float).eps
