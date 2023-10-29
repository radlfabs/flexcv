import logging
from dataclasses import dataclass
from pprint import pformat
from typing import Callable, Dict, Type

import pandas as pd
from neptune.metadata_containers.run import Run as NeptuneRun
from neptune.types import File

from cross_validate import cross_validate
from cv_metrics import MetricsDict
from cv_objective import ObjectiveScorer
from cv_results import CrossValidationResults
from cv_split import CrossValMethod
from funcs import add_module_handlers, run_padding
from model_mapping import ModelConfigDict, ModelMappingDict
from run import DummyRun

logger = logging.getLogger(__name__)
add_module_handlers(logger)


@dataclass
class CrossValidation:
    """Functional interface for cross validation.

    This class is a functional interface for cross validation. It allows you to
    configure the cross validation and then run it. It also allows you to log
    the configuration and results to Neptune.

    Example:
        >>> import flexcv
        >>> import neptune
        >>> X = pd.DataFrame({"x": [1, 2, 3, 4, 5], "z": [1, 2, 3, 4, 5]})
        >>> y = pd.Series([1, 2, 3, 4, 5])
        >>> mapping = flexcv.ModelMappingDict(
        ...     {
        ...         "LinearModel": flexcv.ModelConfigDict(
        ...             {
        ...                 "model": "LinearRegression",
        ...                 "kwargs": {"fit_intercept": True},
        ...             }
        ...         ),
        ...     }
        ... )
        >>> run = neptune.init_run()
        >>> cv = CrossValidation()
        >>> results = (
        ...     cv
        ...     .with_data(X, y)
        ...     .with_models(mapping)
        ...     .log(run)
        ...     .perform()
        ...     .get_results()
        ... )

    """

    def __init__(self) -> None:
        self.config = {
            # Data related
            "X": None,
            "y": None,
            "target_name": "",
            "dataset_name": "",
            "groups": None,
            "slopes": None,
            # CV strategy related
            "n_splits": 5,
            "split_out": CrossValMethod.KFOLD,
            "split_in": CrossValMethod.KFOLD,
            "scale_out": True,
            "scale_in": True,
            "metrics": None,
            # models and optimisation
            "mapping": None,
            "effects": "fixed",
            # optimization related
            "n_trials": 100,
            "objective_scorer": None,
            # regarding mixed effects
            "em_max_iterations": 100,
            "em_stopping_threshold": None,
            "em_stopping_window": None,
            "predict_known_groups_lmm": True,
            # run related
            "random_seed": 42,
            "diagnostics": False,
            "break_cross_val": False,
            "run": DummyRun(),
        }

    def __repr__(self) -> str:
        return pformat(self.config)

    def __str__(self) -> str:
        return pformat(self.config)

    def set_dataframes(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        groups: pd.DataFrame | pd.Series = None,
        slopes: pd.DataFrame | pd.Series = None,
        target_name: str = "",
        dataset_name: str = "",
    ):
        # check values
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas DataFrame or Series")
        if (
            not isinstance(groups, pd.DataFrame)
            and not isinstance(groups, pd.Series)
            and groups is not None
        ):
            raise TypeError("group must be a pandas DataFrame or Series")
        if (
            not isinstance(slopes, pd.DataFrame)
            and not isinstance(slopes, pd.Series)
            and slopes is not None
        ):
            raise TypeError("slopes must be a pandas DataFrame or Series")
        if not isinstance(target_name, str):
            raise TypeError("target_name must be a string")
        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be a string")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if groups is not None and X.shape[0] != groups.shape[0]:
            raise ValueError("X and group must have the same number of rows")
        if slopes is not None:
            if X.shape[0] != slopes.shape[0]:
                raise ValueError("X and slopes must have the same number of rows")
            # make sure slopes (which can be multiple columns or a series) is in X
            if isinstance(slopes, pd.Series):
                if slopes.name not in X.columns:
                    raise ValueError(f"{slopes.name} is not in X")
            elif isinstance(slopes, pd.DataFrame):
                for slope in slopes.columns:
                    if slope not in X.columns:
                        raise ValueError(f"{slope} is not in X")

        # assign values
        self.config["X"] = X
        self.config["y"] = y
        self.config["groups"] = groups
        self.config["slopes"] = slopes
        if target_name:
            self.config["target_name"] = target_name
        if dataset_name:
            self.config["dataset_name"] = dataset_name
        return self

    def set_splits(
        self,
        method_outer_split: CrossValMethod | str = CrossValMethod.KFOLD,
        method_inner_split: CrossValMethod | str = CrossValMethod.KFOLD,
        n_splits: int = 5,
        scale_inner_fold: bool = True,
        scale_outer_fold: bool = True,
        break_cross_val: bool = False,
        metrics: MetricsDict = None,
    ):
        ALLOWED_METHODS = CrossValMethod.allowed_methods

        # check values
        if not isinstance(method_outer_split, CrossValMethod) and not isinstance(
            method_outer_split, str
        ):
            raise TypeError("method_outer_split must be a CrossValMethod")
        if not isinstance(method_inner_split, CrossValMethod) and not isinstance(
            method_inner_split, str
        ):
            raise TypeError("method_inner_split must be a CrossValMethod")
        if method_outer_split not in ALLOWED_METHODS and isinstance(
            method_outer_split, str
        ):
            raise ValueError(f"method_outer_split must be one of {ALLOWED_METHODS}")
        if method_inner_split not in ALLOWED_METHODS and isinstance(
            method_inner_split, str
        ):
            raise ValueError(f"method_inner_split must be one of {ALLOWED_METHODS}")

        if not isinstance(n_splits, int):
            raise TypeError("n_splits must be an integer")
        if not isinstance(scale_inner_fold, bool):
            raise TypeError("scale_inner_fold must be a boolean")
        if not isinstance(scale_outer_fold, bool):
            raise TypeError("scale_outer_fold must be a boolean")
        if not isinstance(break_cross_val, bool):
            raise TypeError("break_cross_val must be a boolean")
        if metrics and not isinstance(metrics, MetricsDict):
            raise TypeError("metrics must be a MetricsDict")

        # assign values
        self.config["method_out"] = method_outer_split
        self.config["method_in"] = method_inner_split
        self.config["n_splits"] = n_splits
        self.config["scale_inner_fold"] = scale_inner_fold
        self.config["scale_outer_fold"] = scale_outer_fold
        self.config["break_cross_val"] = break_cross_val
        if metrics:
            self.config["metrics"] = metrics
        return self

    def set_models(
        self,
        mapping: ModelMappingDict,
        model_effects: str = "fixed",
    ):
        # check values
        if not isinstance(mapping, ModelMappingDict):
            raise TypeError("mapping must be a ModelMappingDict")
        if not isinstance(model_effects, str):
            raise TypeError("model_effects must be a string")

        # assign values
        self.config["mapping"] = mapping
        self.config["model_effects"] = model_effects
        return self

    def set_inner_cv(
        self,
        n_trials: int = 100,
        objective_scorer: ObjectiveScorer = None,
    ):
        # check values
        if not isinstance(n_trials, int):
            raise TypeError("n_trials must be an integer")
        if objective_scorer and not isinstance(objective_scorer, ObjectiveScorer):
            raise TypeError("objective_scorer must be an ObjectiveScorer")

        # assign values
        self.config["n_trials"] = n_trials
        self.config["objective_scorer"] = objective_scorer
        return self

    def set_mixed_effects(
        self,
        em_max_iterations: int = 100,
        em_stopping_threshold: float = None,
        em_stopping_window: int = None,
        predict_known_groups_lmm: bool = True,
    ):
        # check values
        if not isinstance(em_max_iterations, int):
            raise TypeError("em_max_iterations must be an integer")
        if em_stopping_threshold and not isinstance(em_stopping_threshold, float):
            raise TypeError("em_stopping_threshold must be a float")
        if em_stopping_window and not isinstance(em_stopping_window, int):
            raise TypeError("em_stopping_window must be an integer")
        if not isinstance(predict_known_groups_lmm, bool):
            raise TypeError("predict_known_groups_lmm must be a boolean")

        # assign values
        self.config["em_max_iterations"] = em_max_iterations
        self.config["em_stopping_threshold"] = em_stopping_threshold
        self.config["em_stopping_window"] = em_stopping_window
        self.config["predict_known_groups_lmm"] = predict_known_groups_lmm
        return self

    def set_run(
        self,
        run: NeptuneRun = None,
        diagnostics: bool = False,
        random_seed: int = 42,
    ):
        # check values
        if run and not isinstance(run, NeptuneRun):
            raise TypeError("run must be a NeptuneRun")
        if not isinstance(diagnostics, bool):
            raise TypeError("diagnostics must be a boolean")
        if not isinstance(random_seed, int):
            raise TypeError(
                "random_seed is not 42 hahaha. No seriously, random_seed must be an integer"
            )

        # assign values
        self.config["run"] = run
        self.config["diagnostics"] = diagnostics
        self.config["random_seed"] = random_seed
        return self

    def log(self, run: NeptuneRun = None):
        """Logs the config to Neptune"""
        if not run:
            if hasattr(self.config, "run"):
                run = self.config["run"]
            else:
                run = DummyRun()
        else:
            self.config["run"] = run

        run["data/dataset_name"].log(self.config["dataset_name"])
        run["data/target_name"].log(self.config["target_name"])
        run["data/effects"].log(self.config["effects"])

        run["data/X"].upload(File.as_html(self.config["X"]))
        run["data/y"].upload(File.as_html(pd.DataFrame(self.config["y"])))
        if self.config["groups"] is not None:
            run["data/groups"].upload(File.as_html(pd.DataFrame(self.config["groups"])))
            run["data/groups_name"].log(self.config["group"].name)
        if self.config["slopes"] is not None:
            run["data/slopes"].upload(File.as_html(pd.DataFrame(self.config["slopes"])))
            run["data/slopes_name"].log(
                pd.DataFrame(self.config["slopes"]).columns.tolist()
            )

        if isinstance(self.config["split_out"], CrossValMethod):
            self.config["split_out"] = self.config["split_out"].value
        elif isinstance(self.config["split_out"], CrossValMethod):
            self.config["split_out"] = self.config["split_out"]

        if isinstance(self.config["split_in"], CrossValMethod):
            self.config["split_in"] = self.config["split_in"].value
        elif isinstance(self.config["split_in"], str):
            self.config["split_in"] = self.config["split_in"]

        run["cross_val/cross_val_method_out"].log(self.config["split_out"].value)
        run["cross_val/cross_val_method_in"].log(self.config["split_in"].value)
        run["cross_val/n_splits_out"].log(self.config["n_splits_out"])
        run["cross_val/n_splits_in"].log(self.config["n_splits_in"])
        run["cross_val/scale_in"].log(self.config["scale_in"])
        run["cross_val/scale_out"].log(self.config["scale_out"])
        run["cross_val/break_cross_val"].log(self.config["break_cross_val"])
        run["cross_val/metrics"].log(pformat(self.config["metrics"]))

        run["models/mapping"].log(pformat(self.config["mapping"]))
        run["optimization/n_trials"].log(self.config["n_trials"])
        run["optimization/objective_scorer"].log(
            pformat(self.config["objective_scorer"])
        )
        run["mixed_effects/em_max_iterations"].log(self.config["em_max_iterations"])
        run["mixed_effects/em_stopping_threshold"].log(
            self.config["em_stopping_threshold"]
        )
        run["mixed_effects/em_stopping_window"].log(self.config["em_stopping_window"])
        run["mixed_effects/predict_known_groups_lmm"].log(
            self.config["predict_known_groups_lmm"]
        )

        run["run/diagnostics"].log(self.config["diagnostics"])
        run["run/random_seed"].log(self.config["random_seed"])

        if self.results_:
            summary_df = self.results_.summary
            run["results/summary"].upload(File.as_html(summary_df))
        self._was_logged = True
        return self

    @run_padding
    def perform(self):
        if not hasattr(self.config, "run"):
            self.config["run"] = DummyRun()
        run = self.config["run"]

        results = cross_validate(**self.config)
        self.results_ = CrossValidationResults(results)
        if self._was_logged:
            run["results/summary"].upload(File.as_html(self.results_.summary))
        return self

    def get_results(self) -> CrossValidationResults:
        return self.results_

    @property
    def results(self) -> CrossValidationResults:
        return self.results_


if __name__ == "__main__":

    import flexcv
    from flexcv.data_generation import generate_regression
    from flexcv.models import LinearModel
    from flexcv.run import Run  # import dummy run object

    # make sample data
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
        cv.set_dataframes(X, y, group, dataset_name="ExampleData")
        .set_splits(
            method_outer_split=flexcv.CrossValMethod.GROUP,
            method_inner_split=flexcv.CrossValMethod.KFOLD,
        )
        .set_models(model_map)
        .set_run(Run())
        .perform()
        .get_results()
    )

    results.summary.to_excel("my_cv_results.xlsx")


if __name__ == "__main__":
    from data_generation import generate_regression
    
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
        cv.set_dataframes(X, y, group, random_slopes)
        .set_models(model_map)
        .set_run(DummyRun())
        .perform()
        .get_results()
    )