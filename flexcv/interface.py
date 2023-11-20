"""
This module contains the CrossValidation class. This class is the central interface to interact with `flexcv`.
"""
import inspect
import logging
import pathlib
import warnings
from dataclasses import dataclass
from pprint import pformat
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from neptune.metadata_containers.run import Run as NeptuneRun
from neptune.types import File
from sklearn.model_selection import BaseCrossValidator
from xgboost.callback import TrainingCallback

from .core import cross_validate
from .metrics import MetricsDict
from .model_mapping import ModelConfigDict, ModelMappingDict
from .model_postprocessing import ModelPostProcessor
from .model_selection import ObjectiveScorer
from .results_handling import CrossValidationResults
from .run import Run as DummyRun
from .split import CrossValMethod, string_to_crossvalmethod
from .utilities import add_module_handlers, run_padding
from .yaml_parser import read_mapping_from_yaml_file, read_mapping_from_yaml_string

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
        self._was_logged_ = False
        self._was_performed_ = False
        self._config_logged_ = False
        self._result_logged_ = False
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
            "mapping": ModelMappingDict({}),
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
            X (pd.DataFrame): The features. Must not contain the target or groups.
            y (pd.DataFrame | pd.Series): The target variable.
            groups (pd.DataFrame | pd.Series): The grouping/clustering variable. (Default value = None)
            slopes (pd.DataFrame | pd.Series): The random slopes variable(s) (Default value = None)
            target_name (str): Customize the target's name. This string will be used in logging. (Default value = "")
            dataset_name (str): Customize your datasdet's name. This string will be used in logging. (Default value = "")

        Returns:
            (CrossValidation): self

        Example:
            ```python
            >>> X = pd.DataFrame({"x": [1, 2, 3, 4, 5], "z": [1, 2, 3, 4, 5]})
            >>> y = pd.Series([1, 2, 3, 4, 5])
            >>> cv = CrossValidation()
            >>> cv.set_data(X, y)
            ```
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
            self.config["target_name"] = str(y.name)
        if dataset_name:
            self.config["dataset_name"] = dataset_name
        return self

    def set_splits(
        self,
        split_out: str
        | CrossValMethod
        | BaseCrossValidator
        | Iterator = CrossValMethod.KFOLD,
        split_in: str
        | CrossValMethod
        | BaseCrossValidator
        | Iterator = CrossValMethod.KFOLD,
        n_splits_out: int = 5,
        n_splits_in: int = 5,
        scale_out: bool = True,
        scale_in: bool = True,
        break_cross_val: bool = False,
        metrics: MetricsDict = None,
    ):
        """Set the cross validation strategy.
        Set the split method simply by passing the `CrossValMethod` as a string or enum value. Passing as string might be more convenient for you but could lead to typos.
        When passing as string, the string must be a valid value of the `CrossValMethod` enum.
        See the reference for `CrossValMethod` for more details.

        Valid strings for `split_out` and `split_in`:
            - "KFold"
            - "StratifiedKFold"
            - "CustomStratifiedKFold"
            - "GroupKFold"
            - "StratifiedGroupKFold"
            - "CustomStratifiedGroupKFold"

        Args:
            split_out (str | CrossValMethod): Outer split method. (Default value = CrossValMethod.KFOLD)
            split_in (str | CrossValMethod): Inner split method for hyperparameter tuning. (Default value = CrossValMethod.KFOLD)
            n_splits_out (int): Number of splits in outer loop. (Default value = 5)
            n_splits_in (int): Number of splits in inner loop. (Default value = 5)
            scale_out (bool): Whether or not the Features of the outer loop will be scaled to mean 0 and variance 1. (Default value = True)
            scale_in (bool): Whether or not the Features of the inner loop will be scaled to mean 0 and variance 1. (Default value = True)
            break_cross_val (bool): If True, the outer loop we break after first iteration. Use for debugging. (Default value = False)
            metrics (MetricsDict): A dict containing evaluation metrics for the outer loop results. See MetricsDict for details. (Default value = None)

        Returns:
          (CrossValidation): self

        Example:
            Passing the method as instance of CrossValMethod:
            ```python
            >>> from flexcv import CrossValidation, CrossValMethod
            >>> cv = CrossValidation()
            >>> cv.set_splits(split_out=CrossValMethod.KFOLD, split_in=CrossValMethod.KFOLD)
            ```
            Passing the method as a string:
            ```python
            >>> from flexcv import CrossValidation
            >>> cv = CrossValidation()
            >>> cv.set_splits(split_out="KFold", split_in="KFold")
            # Valid strings: "KFold", "StratifiedKFold", "CustomStratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "CustomStratifiedGroupKFold"
            ```


        Split methods:
            The split strategy is controlled by the `split_out` and `split_in` arguments. You can pass the actual `CrossValMethod` enum or a string.

            The `split_out` argument controls the fold assignment in the outer cross validation loop.
            In each outer loop the model is fit on the training fold and model performance is evaluated on unseen data of the test fold.
            The `split_in` argument controls the inner loop split strategy. The inner loop cross validates the hyperparameters of the model.
            A model is typically built by sampling from a distribution of hyperparameters. It is fit on the inner training fold and evaluated on the inner test fold.
            Of course, the inner loop is nested in the outer loop, so the inner split is performed on the outer training fold.

            Read more about it in the respective documentation of the `CrossValMethod` enum.

        """

        # get values of CrossValMethod enums
        ALLOWED_STRINGS = [method.value for method in CrossValMethod]
        ALLOWED_METHODS = [method for method in CrossValMethod]

        if isinstance(split_out, str) and (split_out not in ALLOWED_STRINGS):
            raise TypeError(
                f"split_out must be a valid CrossValMethod name, was {split_out}. Choose from: "
                + ", ".join(ALLOWED_STRINGS)
                + "."
            )

        if isinstance(split_in, str) and (split_in not in ALLOWED_STRINGS):
            raise TypeError(
                f"split_in must be a valid CrossValMethod name, was {split_in}. Choose from: "
                + ", ".join(ALLOWED_STRINGS)
                + "."
            )

        if not any(
            [
                isinstance(split_out, str),
                isinstance(split_out, CrossValMethod),
                isinstance(split_out, BaseCrossValidator),
                isinstance(split_out, Iterator),
            ]
        ):
            raise TypeError(
                "split_out must be of Type str, CrossValMethod, BaseCrossValidator or Iterator."
            )

        if not any(
            [
                isinstance(split_in, str),
                isinstance(split_in, CrossValMethod),
                isinstance(split_in, BaseCrossValidator),
                isinstance(split_in, Iterator),
            ]
        ):
            raise TypeError(
                "split_in must be of Type str, CrossValMethod, BaseCrossValidator or Iterator."
            )

        if isinstance(split_out, CrossValMethod) and (split_out not in ALLOWED_METHODS):
            raise TypeError("split_out must be a valid CrossValMethod ")

        if isinstance(split_in, CrossValMethod) and (split_in not in ALLOWED_METHODS):
            raise TypeError("split_in must be a valid CrossValMethod")

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

        if isinstance(split_out, str):
            split_out = string_to_crossvalmethod(split_out)
        if isinstance(split_in, str):
            split_in = string_to_crossvalmethod(split_in)
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
        mapping: ModelMappingDict = None,
        yaml_path: str | pathlib.Path = None,
        yaml_string: str = None,
    ):
        """Set your models and related parameters. Pass a ModelMappingDict or pass yaml code or a path to a yaml file.
        The mapping attribute of the class is a ModelMappingDict that contains a ModelConfigDict for each model.
        The class attribute self.config["mapping"] is always updated in this method. 
        Therefore, you can call this method multiple times to add models to the mapping.
        You can also call set_models() with a ModelMappingDict and then call set_models() again with yaml code or a path to a yaml file or after you already called add_models().
        
        
        Args:
          mapping (ModelMappingDict[str, ModelConfigDict]): Dict of model names and model configurations. See ModelMappingDict for more information. (Default value = None)
          yaml_path (str | pathlib.Path): Path to a yaml file containing a model mapping. See flexcv.yaml_parser for more information. (Default value = None)
          yaml_string (str): String containing yaml code. See flexcv.yaml_parser for more information. (Default value = None)

        Returns:
          (CrossValidation): self

        Example:
            In your yaml file:
            ```yaml
            RandomForest:
                model: sklearn.ensemble.RandomForestRegressor
                post_processor: flexcv.model_postprocessing.MixedEffectsPostProcessor
                requires_inner_cv: True
                params:
                    max_depth: !Int
                        low: 1
                        high: 10
            ```
            In your code:
            ```python
            >>> from flexcv import CrossValidation
            >>> cv = CrossValidation()
            >>> cv.set_models(yaml_path="path/to/your/yaml/file")
            ```
            This will automatically read the yaml file and create a ModelMappingDict.
            It even takes care of the imports and instantiates the classes of model, postprocessor and for the optune distributions.
        """
        if not any((mapping, yaml_path, yaml_string)):
            raise ValueError(
                "You must provide either mapping, yaml_path, or yaml_string"
            )

        if sum(bool(x) for x in (mapping, yaml_path, yaml_string)) > 1:
            raise ValueError(
                "You must provide either mapping, yaml_path or yaml_string, not multiple"
            )
        
        if mapping is not None:
            if not isinstance(mapping, ModelMappingDict):
                raise TypeError("mapping must be a ModelMappingDict")
            self.config["mapping"].update(mapping)

        elif yaml_path is not None:
            if not isinstance(yaml_path, str) and not isinstance(
                yaml_path, pathlib.Path
            ):
                raise TypeError("yaml_path must be a string or pathlib.Path")
            self.config["mapping"].update(read_mapping_from_yaml_file(yaml_path))

        elif yaml_string is not None:
            if not isinstance(yaml_string, str):
                raise TypeError("yaml_string must be a string")
            self.config["mapping"].update(read_mapping_from_yaml_string(yaml_string))

        return self

    def set_inner_cv(
        self,
        n_trials: int = 100,
        objective_scorer: ObjectiveScorer = None,
    ):
        """Configure parameters regarding inner cross validation and Optuna optimization.

        Args:
          n_trials (int): Number of trials to sample from the parameter distributions (Default value = 100)
          objective_scorer (ObjectiveScorer): Callable to provide the optimization objective value. Is called during Optuna SearchCV (Default value = None)

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

    def set_lmer(self, predict_known_groups_lmm: bool = True):
        """Configure parameters regarding linear mixed effects regression models.

        Args:
          predict_known_groups_lmm (bool): For use with LMER, whether or not known groups should be predicted (Default value = True)

        Returns:
          (CrossValidation): self

        """
        # check values
        if not isinstance(predict_known_groups_lmm, bool):
            raise TypeError("predict_known_groups_lmm must be a boolean")

        # assign values
        self.config["predict_known_groups_lmm"] = predict_known_groups_lmm
        return self

    def set_merf(
        self,
        add_merf_global: bool = False,
        em_max_iterations: int = 100,
        em_stopping_threshold: float = None,
        em_stopping_window: int = None,
    ):
        """Configure mixed effects parameters.

        Args:
          add_merf_global (bool): If True, the model is passed into the MERF class after it is evaluated, to obtain mixed effects corrected predictions. (Default value = False)
          em_max_iterations (int): For use with EM. Max number of iterations (Default value = 100)
          em_stopping_threshold (float): For use with EM. Threshold of GLL residuals for early stopping (Default value = None)
          em_stopping_window (int): For use with EM. Number of consecutive iterations to be below threshold for early stopping (Default value = None)


        Returns:
          (CrossValidation): self

        """
        if not isinstance(add_merf_global, bool):
            raise TypeError("add_merf must be a boolean")
        if not isinstance(em_max_iterations, int):
            raise TypeError("em_max_iterations must be an integer")
        if em_stopping_threshold and not isinstance(em_stopping_threshold, float):
            raise TypeError("em_stopping_threshold must be a float")
        if em_stopping_window and not isinstance(em_stopping_window, int):
            raise TypeError("em_stopping_window must be an integer")

        # assign values
        self.config["add_merf_global"] = add_merf_global
        self.config["em_max_iterations"] = em_max_iterations
        self.config["em_stopping_threshold"] = em_stopping_threshold
        self.config["em_stopping_window"] = em_stopping_window
        return self

    def set_run(
        self,
        run: NeptuneRun = None,
        diagnostics: bool = False,
        random_seed: int = 42,
    ):
        """

        Args:
          run (NeptuneRun): The run object to use for logging (Default value = None)
          diagnostics (bool): If True, extended diagnostic plots are logged (Default value = False)
          random_seed (int): Seed for random processes (Default value = 42)

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

    def add_model(
        self,
        model_class: object,
        requires_inner_cv: bool = False,
        model_name: str = "",
        post_processor: ModelPostProcessor = None,
        params: dict = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None,
        model_kwargs: dict = None,
        fit_kwargs: dict = None,
        **kwargs,
    ):
        """Add a model to the model mapping dict.
        This method is a convenience method to add a model to the model mapping dict without needing the ModelMappingDict and ModelConfigDict classes.

        Args:
          model_class (object): The model class. Must have a fit() method.
          requires_inner_cv (bool): Whether or not the model requires inner cross validation. (Default value = False)
          model_name (str): The name of the model. Used for logging.
          post_processor (ModelPostProcessor): A post processor to be applied to the model. (Default value = None)
          callbacks (Optional[Sequence[TrainingCallback]]): Callbacks to be passed to the fit method of the model in outer CV. (Default value = None)
          kwargs: A dict of additional keyword arguments that will be passed to the model constructor.
          fit_kwargs: A dict of keyword arguments that will be passed to the model fit method.
          **kwargs: Arbitrary keyword arguments that will be passed to the ModelConfigDict.

        Returns:
            (CrossValidation): self

        Example:
            ```python
            >>> from flexcv import CrossValidation
            >>> from flexcv.models import LinearModel
            >>> cv = CrossValidation()
            >>> cv.add_model(LinearModel, model_name="LinearModel", skip_inner_cv=True)
            ```
            Is equivalent to:
            ```python
            >>> from flexcv import CrossValidation
            >>> from flexcv.models import LinearModel
            >>> from flexcv.model_mapping import ModelMappingDict, ModelConfigDict
            >>> cv = CrossValidation()
            >>> mapping = ModelMappingDict(
            ...     {
            ...         "LinearModel": ModelConfigDict(
            ...             {
            ...                 "model": LinearModel,
            ...                 "skip_inner_cv": True,
            ...             }
            ...         ),
            ...     }
            ... )
            >>> cv.set_models(mapping)
            ```
            As you can see the `add_model()` method is a convenience method to add a model to the model mapping dict without needing the ModelMappingDict and ModelConfigDict classes.
            In cases of multiple models per run, or if you want to reuse the model mapping dict, you should look into the `ModelMappingDict` and the `set_models()` method.

        """

        # check values
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        if not issubclass(model_class, object):
            raise TypeError("model_class must be a class")

        if not isinstance(requires_inner_cv, bool):
            raise TypeError("skip_inner_cv must be a boolean")

        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dict")

        # check if post_processor is a class that inherits ModelPostProcessor object that is not instantiated
        if post_processor and not issubclass(post_processor, ModelPostProcessor):
            raise TypeError("post_processor must be a ModelPostProcessor")

        # params and kwargs may not contain the same keys
        if kwargs is not None and not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dict")

        if (
            kwargs is not None
            and params is not None
            and (set(params.keys()) & set(kwargs.keys()))
        ):
            raise ValueError(
                "params and additional kwargs may not contain the same keys"
            )

        if callbacks is not None and not isinstance(callbacks, Sequence):
            raise TypeError("callbacks must be a Sequence of TrainingCallbacks")

        if requires_inner_cv and params is None:
            warnings.warn(
                f"You did not provide hyperparameters for the model {model_name} but set requires_inner_cv to True.",
                UserWarning,
            )

        if model_kwargs is not None and not isinstance(model_kwargs, dict):
            raise TypeError("model_kwargs must be a dict")

        if fit_kwargs is not None and not isinstance(fit_kwargs, dict):
            raise TypeError("fit_kwargs must be a dict")

        if not requires_inner_cv and params is not None and params != {}:
            requires_inner_cv = True

        if not params:
            params = {}

        if not model_name:
            model_name = model_class.__name__

        if model_kwargs is None:
            model_kwargs = {}

        if fit_kwargs is None:
            fit_kwargs = {}

        if kwargs is None:
            kwargs = {}

        callbacks_dict = {}
        if callbacks is not None:
            callbacks_dict = {"callbacks": callbacks}

        config_dict = {
            "model": model_class,
            "post_processor": post_processor,
            "requires_inner_cv": requires_inner_cv,
            "params": params,
            "callbacks": callbacks_dict,
            "fit_kwargs": fit_kwargs,
            "model_kwargs": model_kwargs,
            **kwargs,
        }

        self.config["mapping"][model_name] = ModelConfigDict(config_dict)
        return self

    def _describe_config(self):
        """This function creates a representation of the config dict for logging purposes. It includes the Model Mapping and a target variable description."""

        mapping_str = " + ".join(self.config["mapping"].keys())
        target_str = self.config["target_name"]
        return f"CrossValidation Summary for Regression of Target {target_str} with Models {mapping_str}"

    def _prepare_before_perform(self):
        """Make preparation steps before performing the cross validation.

        - Checks if a neptune run object has been set. If the user did not provide a neptune run object, a dummy run is instantiated.
        - Checks if the split_out and split_in attributes are set to strings and converts the strings to a CrossValMethod enum.

        Iterates over the ModelMappingDict:
        - Checks if n_trials is set for every model. If not, set to the value of self.config["n_trials"]. If n_trials is not set for a model and not set for the CrossValidation object, it is not used.
        - Checks if "add_merf" is set for a model or if it set globally in the class. In the latter case, the value is set for the model.
        - Checks if the model signature contains "clusters". If so, the model is a mixed effects model and we set a flag "consumes_clusters" to True.

        This method is called by the `perform()` method.
        """
        if isinstance(self.config["split_out"], str):
            self.config["split_out"] = string_to_crossvalmethod(
                self.config["split_out"]
            )

        if isinstance(self.config["split_in"], str):
            self.config["split_in"] = string_to_crossvalmethod(self.config["split_in"])

        if not "run" in self.config:
            self.config["run"] = DummyRun()

        # check if add_merf is set globally
        # iterate over all items in the model mapping
        # if the inenr dict has a key "add_merf" do nothing
        # if it doesnt and add_merf is set globally, set it for the model
        self.config["add_merf_global"] = self.config.setdefault(
            "add_merf_global", False
        )
        self.config["n_trials"] = self.config.setdefault("n_trials", 100)

        for model_key, inner_dict in self.config["mapping"].items():
            if "fit_kwargs" not in inner_dict:
                self.config["mapping"][model_key]["fit_kwargs"] = {}

            if "n_trials" not in inner_dict:
                self.config["mapping"][model_key]["n_trials"] = self.config["n_trials"]

            if "add_merf" not in inner_dict:
                self.config["mapping"][model_key]["add_merf"] = self.config[
                    "add_merf_global"
                ]

            # check model signature for groups and slopes
            # if the model signature contains groups and slopes, the model is a mixed effects model
            model_class = inner_dict["model"]
            model_signature_parameters = inspect.signature(model_class).parameters
            model_fit_signature_parameters = inspect.signature(
                model_class.fit
            ).parameters

            self.config["mapping"][model_key]["consumes_clusters"] = (
                "clusters" in model_fit_signature_parameters
            )
            self.config["mapping"][model_key]["requires_formula"] = (
                "formula" in model_fit_signature_parameters
            )

            if (
                "model_kwargs" not in self.config["mapping"][model_key]
            ):  # TODO test case
                self.config["mapping"][model_key]["model_kwargs"] = {}

            if "n_jobs" in model_signature_parameters:
                self.config["mapping"][model_key]["model_kwargs"][
                    "n_jobs"
                ] = self.config["mapping"][model_key]["n_jobs_model"]

            if "random_state" in model_signature_parameters:
                self.config["mapping"][model_key]["model_kwargs"][
                    "random_state"
                ] = self.config["random_seed"]

    def _log_config(self):
        """Logs the config to Neptune. If None, a Dummy is instantiated.

        Args:
          run (NeptuneRun): The run to log to (Default value = None)

        Returns:
          (CrossValidation): self

        """
        run = self.config["run"]

        # run["Data/DatasetName"] = self.config["dataset_name"] if self.config["dataset_name"] is not None else "None"
        run["Data/TargetName"] = self.config["target_name"]
        run["Data/model_effects"] = self.config["model_effects"]
        run["Data/X"].upload(File.as_html(self.config["X"]))
        run["Data/y"].upload(File.as_html(pd.DataFrame(self.config["y"])))

        if self.config["groups"] is not None:
            run["Data/groups"].upload(File.as_html(pd.DataFrame(self.config["groups"])))
            run["Data/groups_name"] = self.config["groups"].name

        if self.config["slopes"] is not None:
            run["Data/slopes"].upload(File.as_html(pd.DataFrame(self.config["slopes"])))
            run["Data/slopes_name"] = pd.DataFrame(
                self.config["slopes"]
            ).columns.tolist()

        try:
            run["Config/Split/cross_val_method_out"] = self.config["split_out"].value
        except AttributeError:
            run["Config/Split/cross_val_method_out"] = self.config["split_out"]
        try:
            run["Config/Split/cross_val_method_in"] = self.config["split_in"].value
        except AttributeError:
            run["Config/Split/cross_val_method_in"] = self.config["split_in"]

        if self.config["metrics"] is not None:
            run["Config/Split/metrics"] = pformat(self.config["metrics"])
        else:
            run["Config/Split/metrics"] = "default"

        run["Config/Split/n_splits_out"] = self.config["n_splits_out"]
        run["Config/Split/n_splits_in"] = self.config["n_splits_in"]
        run["Config/Split/scale_in"] = self.config["scale_in"]
        run["Config/Split/scale_out"] = self.config["scale_out"]
        run["Config/Split/break_cross_val"] = self.config["break_cross_val"]
        run["ModelMapping"] = pformat(self.config["mapping"])
        run["Config/Optimization/n_trials"] = self.config["n_trials"]
        run["Config/Optimization/objective_scorer"] = pformat(
            self.config["objective_scorer"]
        )
        run["Config/MixedEffects/em_max_iterations"] = self.config["em_max_iterations"]
        run["Config/MixedEffects/em_stopping_threshold"] = self.config[
            "em_stopping_threshold"
        ]
        run["Config/MixedEffects/em_stopping_window"] = self.config[
            "em_stopping_window"
        ]
        run["Config/MixedEffects/predict_known_groups_lmm"] = self.config[
            "predict_known_groups_lmm"
        ]
        run["Config/Run/diagnostics"] = self.config["diagnostics"]
        run["Config/Run/random_seed"] = self.config["random_seed"]
        self._config_logged_ = True

    def _log_results(self):
        """This function logs the results to Neptune."""
        if hasattr(self, "results_"):
            summary_df = self.results_.summary
            self.config["run"]["Results/Summary"].upload(File.as_html(summary_df))
        else:
            logger.warning(
                "You have not called perform() yet. No results to log. Call perform() to log the results."
            )
        self.config["run"]["description"] = self._describe_config()
        self._result_logged_ = True
        return self

    @run_padding
    def perform(self):
        """Perform the cross validation according to the configuration passed by the user.
        Checks if a neptune run object has been set. If the user did not provide a neptune run object, a dummy run is instantiated.
        All logs and plots will be logged to the dummy run and will be lost.
        However, the cross validation results is created and can be returned via the `CrossValidation.results` property.

        Returns:
            (CrossValidation): self
        """
        self._prepare_before_perform()
        self._log_config()
        results = cross_validate(**self.config)
        self._was_performed_ = True
        self.results_ = CrossValidationResults(results)
        self._log_results()
        self._was_logged_ = self._config_logged_ and self._result_logged_
        return self

    def get_results(self) -> CrossValidationResults:
        """Returns a `CrossValidationResults` object. This results object is a wrapper class around the results dict from the `cross_validate` function."""
        if hasattr(self, "results_"):
            return self.results_
        else:
            raise RuntimeError(
                "You must call perform() before you can get the results."
            )

    @property
    def results(self) -> CrossValidationResults:
        """Returns a `CrossValidationResults` object. This results object is a wrapper class around the results dict from the `cross_validate` function."""
        if hasattr(self, "results_"):
            return self.results_
        else:
            raise RuntimeError(
                "You must call perform() before you can get the results."
            )

    @property
    def was_performed(self) -> bool:
        """Returns True if the cross validation was performed."""
        return self._was_performed_

    @property
    def was_logged(self) -> bool:
        """Returns True if the cross validation was logged."""
        return self._was_logged_

    @property
    def cv_description(self) -> str:
        """Returns a string describing the cross validation configuration."""
        return self._describe_config()


if __name__ == "__main__":
    import numpy as np

    from .models import LinearModel
    from .synthesizer import generate_regression

    X, y, group, random_slopes = generate_regression(
        10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
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
        .set_run(DummyRun())
        .perform()
        .get_results()
    )

    n_values = len(results["LinearModel"]["metrics"])
    r2_values = [results["LinearModel"]["metrics"][k]["r2"] for k in range(n_values)]
    print(f"Mean R2: {np.mean(r2_values)}")
    results.summary.to_csv("results.csv")
