"""
This module contains the CrossValidation class. This class is the central interface to interact with `flexcv`.
"""

import logging
from dataclasses import dataclass
from pprint import pformat

import pandas as pd
from neptune.metadata_containers.run import Run as NeptuneRun
from neptune.types import File

from .core import cross_validate
from .metrics import MetricsDict
from .model_selection import ObjectiveScorer
from .results_handling import CrossValidationResults
from .split import CrossValMethod
from .utilities import add_module_handlers, run_padding
from .model_mapping import ModelConfigDict, ModelMappingDict
from .run import Run as DummyRun

logger = logging.getLogger(__name__)
add_module_handlers(logger)


@dataclass
class CrossValidation:
    """This class is the central interface to interact with `flexcv`.
    Use this dataclass to configure your cross validation run with it's `set_***()` methods.
    You can use method chaining to set multiple configurations at once. 
    This allows you to provide extensive configuration with few lines of code. 
    It also helps you to log the configuration and results to Neptune.
    
    Example:
        ```python
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
        ```
    
    Methods:
        set_data: Sets the data for cross validation. Pass your dataframes and series here.
        set_splits: Sets the cross validation strategy for inner and outer folds. You may need to import `flexcv.CrossValMethod`. 
        set_models: Sets the models to be cross validated. Pass hyperparameter distributions for model tuning here. You may need to import `flexcv.ModelMappingDict` and `flexcv.ModelConfigDict`.
        set_inner_cv: Sets the inner cross validation configuration. Pass arguments regarding the hyperparameter optimization process.  
        set_mixed_effects: Sets the mixed effects parameters. Control if mixed effects are modeled and set arguments regarding the Expectation Maximization algorithm.
        set_run: Sets the run parameters. Pass your Neptune run object here.
        perform: Performs cross validation. Just call this method without args to trigger the nested cross validation run.

    Returns:
      (CrossValidation): CrossValidation object.


    """

    def __init__(self) -> None:
        self._was_logged = False
        self.config = {
            # Data related
            "X": None,
            "y": None,
            "target_name": "",
            "dataset_name": "",
            "groups": None,
            "slopes": None,
            # CV strategy related
            "n_splits_out": 5,
            "n_splits_in": 5,
            "split_out": CrossValMethod.KFOLD,
            "split_in": CrossValMethod.KFOLD,
            "scale_out": True,
            "scale_in": True,
            "metrics": None,
            # models and optimisation
            "mapping": None,
            "model_effects": "fixed",
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

    def set_data(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        groups: pd.DataFrame | pd.Series = None,
        slopes: pd.DataFrame | pd.Series = None,
        target_name: str = "",
        dataset_name: str = "",
    ):
        """Set the data for cross validation.

        Args:
          X: pd.DataFrame: The features. Must not contain the target or groups.
          y: pd.DataFrame | pd.Series: The target variable.
          groups: pd.DataFrame | pd.Series:  The grouping/clustering variable. (Default value = None) (Default value = None)
          slopes: pd.DataFrame | pd.Series: The random slopes variable(s) (Default value = None)
          target_name: str: Customize the target's name. This string will be used in logging. (Default value = "")
          dataset_name: str: Customize your datasdet's name. This string will be used in logging. (Default value = "")

        Returns:
            (CrossValidation): self

        """
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
        else:
            self.config["target_name"] = y.name
        if dataset_name:
            self.config["dataset_name"] = dataset_name
        return self

    def set_splits(
        self,
        split_out: CrossValMethod = CrossValMethod.KFOLD,
        split_in: CrossValMethod = CrossValMethod.KFOLD,
        n_splits_out: int = 5,
        n_splits_in: int = 5,
        scale_out: bool = True,
        scale_in: bool = True,
        break_cross_val: bool = False,
        metrics: MetricsDict = None,
    ):
        """Set the cross validation strategy.

        Args:
          split_out: CrossValMethod: Outer split method (Default value = CrossValMethod.KFOLD)
          split_in: CrossValMethod: Inner split method for hyperparameter tuning (Default value = CrossValMethod.KFOLD)
          n_splits_out: int: Number of splits in outer loop (Default value = 5)
          n_splits_in: int: Number of splits in inner loop (Default value = 5)
          scale_out: bool: Whether or not the Features of the outer loop will be scaled to mean 0 and variance 1  (Default value = True)
          scale_in: bool: Whether or not the Features of the inner loop will be scaled to mean 0 and variance 1 (Default value = True)
          break_cross_val: bool: If True, the outer loop we break after first iteration. Use for debugging (Default value = False)
          metrics: MetricsDict: A dict containint evaluation metrics for the outer loop results. See MetricsDict for Details. (Default value = None)

        Returns:
          (CrossValidation): self

        """
        # get values of CrossValMethod enums
        ALLOWED_METHODS = [method.value for method in CrossValMethod]

        # check values
        if not (split_out.value in ALLOWED_METHODS):
            raise TypeError("split_out must be a CrossValMethod ")

        if not (split_in.value in ALLOWED_METHODS):
            raise TypeError("split_in must be a CrossValMethod")

        if not isinstance(n_splits_out, int):
            raise TypeError("n_splits_out must be an integer")
        if not isinstance(n_splits_in, int):
            raise TypeError("n_splits_in must be an integer")

        if not isinstance(scale_in, bool):
            raise TypeError("scale_in must be a boolean")
        if not isinstance(scale_out, bool):
            raise TypeError("scale_out must be a boolean")

        if not isinstance(break_cross_val, bool):
            raise TypeError("break_cross_val must be a boolean")
        if metrics and not isinstance(metrics, MetricsDict):
            raise TypeError("metrics must be a MetricsDict")

        # assign values
        self.config["split_out"] = split_out
        self.config["split_in"] = split_in
        self.config["n_splits_out"] = n_splits_out
        self.config["n_splits_in"] = n_splits_in
        self.config["scale_in"] = scale_in
        self.config["scale_out"] = scale_out
        self.config["break_cross_val"] = break_cross_val
        if metrics:
            self.config["metrics"] = metrics
        return self

    def set_models(
        self,
        mapping: ModelMappingDict,
    ):
        """Set your models and related parameters.

        Args:
          mapping: ModelMappingDict: Dict of model names and model configurations. See ModelMappingDict for more information.

        Returns:
          (CrossValidation): self

        """
        # check values
        if not isinstance(mapping, ModelMappingDict):
            raise TypeError("mapping must be a ModelMappingDict")

        # assign values
        self.config["mapping"] = mapping
        return self

    def set_inner_cv(
        self,
        n_trials: int = 100,
        objective_scorer: ObjectiveScorer = None,
    ):
        """Configure parameters regarding inner cross validation and Optuna optimization.

        Args:
          n_trials: int: Number of trials to sample from the parameter distributions (Default value = 100)
          objective_scorer: ObjectiveScorer: Callable to provide the optimization objective value. Is called during Optuna SearchCV (Default value = None)

        Returns:
          (CrossValidation): self

        """
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
        model_mixed_effects: bool = False,
        em_max_iterations: int = 100,
        em_stopping_threshold: float = None,
        em_stopping_window: int = None,
        predict_known_groups_lmm: bool = True,
    ):
        """Configure mixed effects parameters.

        Args:
          model_mixed_effects: bool: If mixed effects will be modelled. Set the model_mapping attribute accordingly with set_models (Default value = False)
          em_max_iterations: int: For use with EM. Max number of iterations (Default value = 100)
          em_stopping_threshold: float: For use with EM. Threshold of GLL residuals for early stopping (Default value = None)
          em_stopping_window: int: For use with EM. Number of consecutive iterations to be below threshold for early stopping (Default value = None)
          predict_known_groups_lmm: bool: For use with LMER, whether or not known groups should be predicted (Default value = True)

        Returns:
          (CrossValidation): self

        """
        # check values
        if not isinstance(model_mixed_effects, bool):
            raise TypeError("model_effects must be bool")
        if not isinstance(em_max_iterations, int):
            raise TypeError("em_max_iterations must be an integer")
        if em_stopping_threshold and not isinstance(em_stopping_threshold, float):
            raise TypeError("em_stopping_threshold must be a float")
        if em_stopping_window and not isinstance(em_stopping_window, int):
            raise TypeError("em_stopping_window must be an integer")
        if not isinstance(predict_known_groups_lmm, bool):
            raise TypeError("predict_known_groups_lmm must be a boolean")

        # assign values
        self.config["model_effects"] = "mixed" if model_mixed_effects else "fixed"
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
        """

        Args:
          run: NeptuneRun: The run object to use for logging (Default value = None)
          diagnostics: bool: If True, extended diagnostic plots are logged (Default value = False)
          random_seed: int: Seed for random processes (Default value = 42)

        Returns:
            (CrossValidation): self

        """
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

    def _log(self, run: NeptuneRun = None):
        """Logs the config to Neptune. If None, a Dummy is instantiated.

        Args:
          run: NeptuneRun: The run to log to (Default value = None)

        Returns:
          (CrossValidation): self

        """
        if not run:
            if hasattr(self.config, "run"):
                run = self.config["run"]
            else:
                run = DummyRun()
        else:
            self.config["run"] = run

        run["data/dataset_name"].log(self.config["dataset_name"])
        run["data/target_name"].log(self.config["target_name"])
        run["data/model_effects"].log(self.config["model_effects"])

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
        """Perform the cross validation according to the configuration passed by the user.
        Checks if a neptune run object has been set. If the user did not provide a neptune run object, a dummy run is instantiated.
        All logs and plots will be logged to the dummy run and will be lost.
        However, the cross validation results is created and can be returned via the `CrossValidation.results` property.
        
        Args:
            None
            
        Returns:
            (CrossValidation): self
        """
        if not hasattr(self.config, "run"):
            self.config["run"] = DummyRun()
        run = self.config["run"]

        results = cross_validate(**self.config)
        self.results_ = CrossValidationResults(results)
        if self._was_logged:
            self._log()
            run["results/summary"].upload(File.as_html(self.results_.summary))
        return self

    def get_results(self) -> CrossValidationResults:
        """Returns a `CrossValidationResults` object. This results object is a wrapper class around the results dict from the `cross_validate` function."""
        return self.results_

    @property
    def results(self) -> CrossValidationResults:
        """Returns a `CrossValidationResults` object. This results object is a wrapper class around the results dict from the `cross_validate` function."""
        return self.results_


if __name__ == "__main__":
    import numpy as np
    from .models import LinearModel
    from .synthesizer import generate_regression

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
        .set_run(DummyRun())
        .perform()
        .get_results()
    )

    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["r2"] for k in range(n_values)]
    print(f"Mean R2: {np.mean(r2_values)}")
    results.summary.to_csv("results.csv")
