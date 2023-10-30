import numpy as np

from flexcv.data_generation import generate_regression
from flexcv.cv_class import CrossValidation
from flexcv.cv_class import ModelConfigDict
from flexcv.cv_class import ModelMappingDict
from flexcv.run import Run
from flexcv.models import LinearModel


def simple_regression():
    X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)
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
