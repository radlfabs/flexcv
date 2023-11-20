import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from flexcv.core import (
    ModelMappingDict,
    cross_validate,
    preprocess_features,
    preprocess_slopes,
)
from flexcv.metrics import MetricsDict, mse_wrapper
from flexcv.run import Run


def test_preprocess_slopes_allows_series():
    Z_train_slope = pd.Series(np.random.rand(5))
    Z_test_slope = pd.Series(np.random.rand(5))
    preprocess_slopes(Z_train_slope.copy(), Z_test_slope.copy(), must_scale=True)


def test_preprocess_slopes_returns_values():
    # Test preprocess_slopes function returns a tuple
    Z_train_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_test_slope = pd.DataFrame(np.random.rand(5, 3))
    val = preprocess_slopes(Z_train_slope.copy(), Z_test_slope.copy(), must_scale=True)
    assert isinstance(val, tuple)
    assert len(val) == 2
    assert isinstance(val[0], np.ndarray)
    assert isinstance(val[1], np.ndarray)
    # test if Intercept column is added
    assert val[0].shape[1] == Z_train_slope.shape[1] + 1
    assert val[1].shape[1] == Z_test_slope.shape[1] + 1


def test_preprocess_slopes_raises():
    Z_train_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_test_slope = pd.DataFrame(np.random.rand(5, 3))
    with pytest.raises(TypeError):
        preprocess_slopes("123", Z_test_slope, must_scale=True)

    with pytest.raises(TypeError):
        preprocess_slopes(Z_train_slope, "123", must_scale=True)

    with pytest.raises(TypeError):
        preprocess_slopes(Z_train_slope, Z_test_slope, must_scale="123")

    # wrong shapes
    Z_test_slope = pd.DataFrame(np.random.rand(5, 4))
    with pytest.raises(ValueError):
        preprocess_slopes(Z_train_slope, Z_test_slope, must_scale=True)


def test_preprocess_slopes_no_scaling():
    # Test preprocess_slopes function with no scaling
    Z_train_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_test_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_train, Z_test = preprocess_slopes(
        Z_train_slope.copy(), Z_test_slope.copy(), must_scale=False
    )
    assert np.array_equal(Z_train[:, 1:], Z_train_slope.to_numpy())
    assert np.array_equal(Z_test[:, 1:], Z_test_slope.to_numpy())
    assert np.all(Z_train[:, 0] == 1)
    assert np.all(Z_test[:, 0] == 1)


def test_preprocess_slopes_with_scaling():
    # Test preprocess_slopes function with scaling
    Z_train_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_test_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_train, Z_test = preprocess_slopes(
        Z_train_slope.copy(), Z_test_slope.copy(), must_scale=True
    )
    scaler = StandardScaler()
    Z_train_slope_scaled = scaler.fit_transform(Z_train_slope)
    Z_test_slope_scaled = scaler.transform(Z_test_slope)
    assert np.allclose(Z_train[:, 1:], Z_train_slope_scaled)
    assert np.allclose(Z_test[:, 1:], Z_test_slope_scaled)
    assert np.all(Z_train[:, 0] == 1)
    assert np.all(Z_test[:, 0] == 1)


def test_preprocess_slopes_invalid_Z_train_slope():
    # Test preprocess_slopes function with invalid Z_train_slope
    Z_train_slope = "invalid type"
    Z_test_slope = pd.DataFrame(np.random.rand(5, 3))
    with pytest.raises(TypeError):
        preprocess_slopes(Z_train_slope, Z_test_slope, must_scale=False)


def test_preprocess_slopes_invalid_Z_test_slope():
    # Test preprocess_slopes function with invalid Z_test_slope
    Z_train_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_test_slope = "invalid type"
    with pytest.raises(TypeError):
        preprocess_slopes(Z_train_slope, Z_test_slope, must_scale=False)


def test_preprocess_slopes_invalid_must_scale():
    # Test preprocess_slopes function with invalid must_scale
    Z_train_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_test_slope = pd.DataFrame(np.random.rand(5, 3))
    with pytest.raises(TypeError):
        preprocess_slopes(Z_train_slope, Z_test_slope, must_scale="invalid type")


def test_preprocess_slopes_diff_dims():
    # Test preprocess_slopes function with different dimensions for Z_train_slope and Z_test_slope
    Z_train_slope = pd.DataFrame(np.random.rand(5, 3))
    Z_test_slope = pd.DataFrame(np.random.rand(10, 4))  # different number of columns
    with pytest.raises(ValueError):
        preprocess_slopes(Z_train_slope, Z_test_slope, must_scale=False)


def test_preprocess_features():
    # Test preprocess_features function
    X_train = pd.DataFrame(np.random.rand(5, 3), columns=list("abc"))
    X_test = pd.DataFrame(np.random.rand(5, 3), columns=list("abc"))
    X_train_scaled, X_test_scaled = preprocess_features(X_train.copy(), X_test.copy())
    scaler = StandardScaler()
    X_train_scaled_expected = scaler.fit_transform(X_train)
    X_test_scaled_expected = scaler.transform(X_test)
    assert np.allclose(X_train_scaled, X_train_scaled_expected)
    assert np.allclose(X_test_scaled, X_test_scaled_expected)
    assert list(X_train_scaled.columns) == list(X_train.columns)
    assert list(X_test_scaled.columns) == list(X_test.columns)
    assert list(X_train_scaled.index) == list(X_train.index)
    assert list(X_test_scaled.index) == list(X_test.index)


def test_preprocess_features_invalid_X_train():
    # Test preprocess_features function with invalid X_train
    X_train = "invalid type"
    X_test = pd.DataFrame(np.random.rand(5, 3), columns=list("abc"))
    with pytest.raises(TypeError):
        preprocess_features(X_train, X_test)


def test_preprocess_features_invalid_X_test():
    # Test preprocess_features function with invalid X_test
    X_train = pd.DataFrame(np.random.rand(5, 3), columns=list("abc"))
    X_test = "invalid type"
    with pytest.raises(TypeError):
        preprocess_features(X_train, X_test)


def test_preprocess_features_diff_dims():
    # must fail for different number of cols
    X_train = pd.DataFrame(np.random.rand(5, 3), columns=list("abc"))
    X_test = pd.DataFrame(np.random.rand(5, 4), columns=list("abcd"))
    with pytest.raises(ValueError):
        preprocess_features(X_train, X_test)
    # must allow different number of rows
    X_train = pd.DataFrame(np.random.rand(5, 3), columns=list("abc"))
    X_test = pd.DataFrame(np.random.rand(10, 3), columns=list("abc"))
    preprocess_features(X_train, X_test)


def test_cross_validate():
    # Test cross_validate function
    X = pd.DataFrame(np.random.rand(25, 3), columns=list("abc"))
    y = pd.Series(np.random.rand(25), name="target")
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25), name="slopes")
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"random_state": 42, "n_jobs": -1},
                "fit_kwargs": {},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    results = cross_validate(
        X=X,
        y=y,
        target_name=target_name,
        run=run,
        groups=groups,
        slopes=slopes,
        split_out=split_out,
        split_in=split_in,
        break_cross_val=break_cross_val,
        scale_in=scale_in,
        scale_out=scale_out,
        n_splits_out=n_splits_out,
        n_splits_in=n_splits_in,
        random_seed=random_seed,
        model_effects=model_effects,
        mapping=mapping,
        metrics=MetricsDict(),
        objective_scorer=mse_wrapper,
    )
    assert isinstance(results, dict)
    assert isinstance(results["RandomForestRegressor"], dict)
    assert isinstance(results["RandomForestRegressor"]["parameters"], list)
    assert isinstance(results["RandomForestRegressor"]["model"], list)
    assert isinstance(results["RandomForestRegressor"]["folds_by_metrics"], dict)
    assert isinstance(results["RandomForestRegressor"]["metrics"], list)
    assert isinstance(results["RandomForestRegressor"]["y_pred"], list)
    assert isinstance(results["RandomForestRegressor"]["y_test"], list)
    assert isinstance(results["RandomForestRegressor"]["y_pred_train"], list)
    assert isinstance(results["RandomForestRegressor"]["y_train"], list)

    assert "RandomForestRegressor" in results
    assert "model" in results["RandomForestRegressor"]
    assert "parameters" in results["RandomForestRegressor"]
    assert "folds_by_metrics" in results["RandomForestRegressor"]
    assert "metrics" in results["RandomForestRegressor"]
    assert "y_pred" in results["RandomForestRegressor"]
    assert "y_test" in results["RandomForestRegressor"]
    assert "y_pred_train" in results["RandomForestRegressor"]
    assert "y_train" in results["RandomForestRegressor"]

    assert len(results["RandomForestRegressor"]["model"]) == n_splits_out
    assert len(results["RandomForestRegressor"]["metrics"]) == n_splits_out
    assert len(results["RandomForestRegressor"]["y_pred"]) == n_splits_out
    assert len(results["RandomForestRegressor"]["y_test"]) == n_splits_out
    assert len(results["RandomForestRegressor"]["y_pred_train"]) == n_splits_out
    assert len(results["RandomForestRegressor"]["y_train"]) == n_splits_out
    assert len(results["RandomForestRegressor"]["parameters"]) == n_splits_out

    for i in range(n_splits_out):
        assert isinstance(
            results["RandomForestRegressor"]["model"][i], RandomForestRegressor
        )
        assert isinstance(results["RandomForestRegressor"]["metrics"][i], dict)
        assert isinstance(results["RandomForestRegressor"]["parameters"][i], dict)
        assert isinstance(
            results["RandomForestRegressor"]["parameters"][i]["n_estimators"], int
        )
        assert isinstance(
            results["RandomForestRegressor"]["parameters"][i]["random_state"], int
        )

        assert len(results["RandomForestRegressor"]["parameters"][i]) == 2

        assert len(results["RandomForestRegressor"]["y_pred"][i]) == len(
            results["RandomForestRegressor"]["y_test"][i]
        )
        assert len(results["RandomForestRegressor"]["y_pred_train"][i]) == len(
            results["RandomForestRegressor"]["y_train"][i]
        )


def test_cross_validate_invalid_X():
    # Test cross_validate function with invalid X
    X = "invalid type"
    y = pd.Series(np.random.rand(25))
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25))
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "n_trials": 3,
                "n_jobs_model": -1,
                "n_jobs_cv": -1,
            }
        }
    )
    with pytest.raises(TypeError):
        results = cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_y():
    # Test cross_validate function with invalid y
    X = pd.DataFrame(np.random.rand(25, 3))
    y = "invalid type"
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25))
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "model_kwargs": {"n_jobs": -1, "random_state": 42},
                "requires_inner_cv": True,
                "n_trials": 3,
                "n_jobs_cv": -1,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_target_name():
    # Test cross_validate function with invalid target_name
    X = pd.DataFrame(np.random.rand(25, 3))
    y = pd.Series(np.random.rand(25))
    target_name = 123  # should be a string
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25))
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"n_jobs": -1, "random_state": 42},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_run():
    # Test cross_validate function with invalid run
    X = pd.DataFrame(np.random.rand(25, 3))
    y = pd.Series(np.random.rand(25))
    target_name = "target"
    run = "invalid type"  # should be an instance of Run
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25))
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"n_jobs": -1, "random_state": 42},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_groups():
    # Test cross_validate function with invalid groups
    X = pd.DataFrame(np.random.rand(25, 3))
    y = pd.Series(np.random.rand(25))
    target_name = "target"
    run = Run()
    groups = "invalid type"  # should be a pandas Series
    slopes = pd.Series(np.random.rand(25))
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"n_jobs": -1, "random_state": 42},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_slopes():
    # Test cross_validate function with invalid slopes
    X = pd.DataFrame(np.random.rand(25, 3))
    y = pd.Series(np.random.rand(25))
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = "invalid type"  # should be a pandas Series
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"n_jobs": -1, "random_state": 42},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_split_out():
    # Test cross_validate function with invalid split_out
    X = pd.DataFrame(np.random.rand(25, 3))
    y = pd.Series(np.random.rand(25))
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25))
    split_out = "invalid type"  # should be an instance of KFold
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    n_trials = 3
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_split_in():
    # Test cross_validate function with invalid split_in
    X = pd.DataFrame(np.random.rand(25, 3))
    y = pd.Series(np.random.rand(25))
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25))
    split_out = KFold(n_splits=3)
    split_in = "invalid type"  # should be an instance of KFold
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"n_jobs": -1, "random_state": 42},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_invalid_groups_but_groupkfold():
    # Test cross_validate function with invalid groups but GroupKFold
    X = pd.DataFrame(np.random.rand(25, 3))
    y = pd.Series(np.random.rand(25))
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25))
    split_out = KFold(n_splits=3)
    split_in = GroupKFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = None
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"n_jobs": -1, "random_state": 42},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(TypeError):
        cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )


def test_cross_validate_missing_fit_kwargs():
    # Test cross_validate function
    X = pd.DataFrame(np.random.rand(25, 3), columns=list("abc"))
    y = pd.Series(np.random.rand(25), name="target")
    target_name = "target"
    run = Run()
    groups = pd.Series(np.random.choice(["group1", "group2"], 25))
    slopes = pd.Series(np.random.rand(25), name="slopes")
    split_out = KFold(n_splits=3)
    split_in = KFold(n_splits=3)
    break_cross_val = False
    scale_in = True
    scale_out = True
    n_splits_out = 3
    n_splits_in = 3
    random_seed = 42
    model_effects = "fixed"
    mapping = ModelMappingDict(
        {
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(
                        10, 100, step=10
                    )
                },
                "requires_inner_cv": True,
                "add_merf": False,
                "model_kwargs": {"random_state": 42, "n_jobs": -1},
                "n_trials": 3,
                "n_jobs_cv": -1,
                "consumes_clusters": False,
                "requires_formula": False,
            }
        }
    )
    with pytest.raises(KeyError):
        results = cross_validate(
            X=X,
            y=y,
            target_name=target_name,
            run=run,
            groups=groups,
            slopes=slopes,
            split_out=split_out,
            split_in=split_in,
            break_cross_val=break_cross_val,
            scale_in=scale_in,
            scale_out=scale_out,
            n_splits_out=n_splits_out,
            n_splits_in=n_splits_in,
            random_seed=random_seed,
            model_effects=model_effects,
            mapping=mapping,
            metrics=MetricsDict(),
            objective_scorer=mse_wrapper,
        )
