import numpy as np

from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.interface import ModelConfigDict
from flexcv.interface import ModelMappingDict
from flexcv.run import Run
from flexcv.models import LinearModel


def simple_regression():
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2
    )
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group, random_slopes)
        .set_models(model_map)
        .set_run(Run())
        .perform()
        .get_results()
    )

    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["r2"] for k in range(n_values)]
    return np.mean(r2_values)


def test_lm_fixed_regression_k3():
    check_value = simple_regression()
    eps = np.finfo(float).eps
    assert (check_value / 0.36535132545331933) > (1 - eps)
