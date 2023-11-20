import numpy as np
import optuna
from data import DATA_TUPLE_3_100
from sklearn.ensemble import RandomForestRegressor

import flexcv.model_postprocessing as mp
from flexcv.interface import CrossValidation, ModelConfigDict, ModelMappingDict
from flexcv.run import Run
from flexcv.synthesizer import generate_regression

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
        .set_splits(n_splits_out=3, break_cross_val=True)
        .set_models(model_map)
        .perform()
        .get_results()
    )

    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["mse"] for k in range(n_values)]
    return r2_values


def sklearn_lm_kfold(X, y):
    """Compute sklearn cross validation score for linear model on random data."""
    # import cross_validation from sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score as sklearn_score

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    sklearn_cv_results = sklearn_score(
        LinearRegression(),
        X,
        y,
        cv=kf,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    return sklearn_cv_results


def test_kfold_flexcv_roughly_equals_sklearn():
    """Compare MSE of pipelines of sklearn and flexcv:
    - sklearn: LinearRegression() -> KFold() -> cross_val_score()
    - flexcv: LinearRegression() -> CrossValidation() with "kFold"-> perform() -> get_results()
    assert MAE on the MSE of the two pipelines is less than machine epsilon
    """
    X, y, _, _ = DATA_TUPLE_3_100

    flexcv_val = flexcv_lm_kfold(X, y)
    sklearn_val = -sklearn_lm_kfold(X, y)
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
        .set_splits(split_out="GroupKFold", n_splits_out=3)
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
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.model_selection import cross_val_score as sklearn_score

    gkf = GroupKFold(n_splits=3)
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
    X, y, group, _ = DATA_TUPLE_3_100

    flexcv_val = flexcv_lm_groupkfold(X, y, group)
    sklearn_val = -sklearn_lm_groupkfold(X, y, group)
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
        .set_splits(n_splits_out=3)
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
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score as sklearn_score

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
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
    X, y, _, _ = DATA_TUPLE_3_100
    flexcv_val = flexcv_lm_kfold(X, y)
    sklearn_val = -sklearn_lm_kfold(X, y)
    assert np.isclose(np.array([flexcv_val]), np.array([sklearn_val]))
