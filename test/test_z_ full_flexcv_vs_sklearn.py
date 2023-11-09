import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

import flexcv.model_postprocessing as mp
from flexcv.synthesizer import generate_regression
from flexcv.interface import CrossValidation
from flexcv.interface import ModelConfigDict
from flexcv.interface import ModelMappingDict
from flexcv.run import Run


##### Test kfold #####

def flexcv_lm_kfold(X, y):
    from sklearn.linear_model import LinearRegression
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearRegression,
                    # "requires_formula": True,
                    "allows_seed": False,
                    "requires_tuning": False,
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

    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["mse"] for k in range(n_values)]
    return np.mean(r2_values)


def sklearn_lm_kfold(X, y):
    """Compute sklearn cross validation score for linear model on random data."""
    # import cross_validation from sklearn
    from sklearn.model_selection import cross_val_score as sklearn_score
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sklearn_cv_results = sklearn_score(
        LinearRegression(),
        X,
        y,
        cv=kf,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    return sklearn_cv_results.mean()


def test_kfold_flexcv_roughly_equals_sklearn():
    """Compare MSE of pipelines of sklearn and flexcv:
    - sklearn: LinearRegression() -> KFold() -> cross_val_score()
    - flexcv: LinearRegression() -> CrossValidation() with "kFold"-> perform() -> get_results()
    assert MAE on the MSE of the two pipelines is less than machine epsilon
    """
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )
    flexcv_val = flexcv_lm_kfold(X, y)
    sklearn_val = - sklearn_lm_kfold(X, y)
    assert np.isclose(np.array([flexcv_val]), np.array([sklearn_val]))


##### Test group kfold #####


def flexcv_lm_groupkfold(X, y, group):
    """Computes flexcv cross validation score for linear model and groupkfold on random data."""
    from sklearn.linear_model import LinearRegression
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearRegression,
                    # "requires_formula": True,
                    "allows_seed": False,
                    "requires_tuning": False,
                }
            ),
        }
    )

    cv = CrossValidation()
    results = (
        cv.set_data(X, y, group)
        .set_models(model_map)
        .set_splits(split_out="GroupKFold")
        .set_run(Run())
        .perform()
        .get_results()
    )

    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["mse"] for k in range(n_values)]
    return np.mean(r2_values)


def sklearn_lm_groupkfold(X, y, group):
    """Compute sklearn cross validation score for linear model and groupkfold on random data."""
    # import cross_validation from sklearn
    from sklearn.model_selection import cross_val_score as sklearn_score
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold
    
    gkf = GroupKFold(n_splits=5)
    sklearn_cv_results = sklearn_score(
        LinearRegression(),
        X,
        y,
        groups=group,
        cv=gkf,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    return sklearn_cv_results.mean()


def test_groupkfold_flexcv_roughly_equals_sklearn():
    """Compare MSE of pipelines of sklearn and flexcv:
    - sklearn: LinearRegression() -> GroupKFold() -> cross_val_score()
    - flexcv: LinearRegression() -> CrossValidation() with "GroupKFold"-> perform() -> get_results()
    assert MAE on the MSE of the two pipelines is less than precision
    """
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )
    flexcv_val = flexcv_lm_groupkfold(X, y, group)
    sklearn_val = - sklearn_lm_groupkfold(X, y, group)
    assert np.isclose(np.array([flexcv_val]), np.array([sklearn_val]))


##### Test sklearn.LinearRegression vs flexcv.models.LinearModel #####

def flexcv_lm_kfold(X, y):
    from flexcv.models import LinearModel
    model_map = ModelMappingDict(
        {
            "LinearModel": ModelConfigDict(
                {
                    "model": LinearModel,
                    "requires_formula": True,
                    "allows_seed": False,
                    "requires_tuning": False,
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

    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["mse"] for k in range(n_values)]
    return np.mean(r2_values)


def sklearn_lm_kfold(X, y):
    """Compute sklearn cross validation score for linear model on random data."""
    # import cross_validation from sklearn
    from sklearn.model_selection import cross_val_score as sklearn_score
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sklearn_cv_results = sklearn_score(
        LinearRegression(),
        X,
        y,
        cv=kf,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    return sklearn_cv_results.mean()


def test_linearmodels_flexcv_roughly_equals_sklearn():
    """Compare MSE of pipelines of sklearn and flexcv:
    - sklearn: LinearRegression() -> KFold() -> cross_val_score()
    - flexcv: LinearRegression() -> CrossValidation() with "kFold"-> perform() -> get_results()
    assert MAE on the MSE of the two pipelines is less than machine epsilon
    """
    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
    )
    flexcv_val = flexcv_lm_kfold(X, y)
    sklearn_val = - sklearn_lm_kfold(X, y)
    assert np.isclose(np.array([flexcv_val]), np.array([sklearn_val]))
