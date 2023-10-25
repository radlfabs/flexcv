from dataclasses import dataclass
from typing import Callable, Dict, Type
from pprint import pformat
import logging
import pandas as pd
from neptune.metadata_containers.run import Run as NeptuneRun
from neptune.types import File

from .cv_split import CrossValMethod
from .cv_metrics import MetricsDict
from .cv_objective import ObjectiveScorer
from .cv_results import CrossValidationResults
from .cross_validate import cross_validate
from .funcs import run_padding
from .funcs import add_module_handlers
from .funcs import get_fixed_effects_formula
from .funcs import get_re_formula

logger = logging.getLogger(__name__)
add_module_handlers(logger)


class ModelConfigDict(Dict[str, Type]):
    """A dictionary that maps model configuration names to their corresponding types.
    Usage:
        ```py
        {
            "inner_cv": bool,
                    # this flag can be set to control if a model is used in the inner cross validation.
                    # if set to False, the model will be instantiated in the outer cross validation without hyper parameter optimization.
            "n_trials": int,
                    # number of trials to be used in hyper parameter optimization.
            "n_jobs_model": {"n_jobs": 1},
                    # number of jobs to be used in the model. We use the sklearn convention here.
                    # n_jobs_model is passed to the model constructor as **n_jobs_model
                    # therefore, it MUST be a dictionary with the key "n_jobs" and the value being an integer
                    # if you want to leave it empty, you can pass the empty dict {}.
            "n_jobs_cv": 1,
                    # number of jobs to be used in the inner cross validation/hyper parameter tuning. We use the sklearn convention here.
            "model": BaseEstimator,
                    # pass your sklearn model here. It must be a class, not an instance.
            "params": {},
                    # pass the parameters to be used in the model here. It must be a dictionary of optuna distributions or an empty dict.
            "post_processor": mp.lm_post,
                    # pass the post processor function to be used here. It must be a callable.
            "mixed_model": BaseEstimator,
                    # pass the mixed effects model to be used here. It must be a class, not an instance.
                    # it's fit method must have the same signature as the fit method of the sklearn models.
            "mixed_post_processor": mp.lmer_post,
                    # pass the post processor function to be used here. It must be a callable.
            "mixed_name": "MixedLM"
                    # name of the mixed effects model. It is used to identify the model in the results dictionary.
        }```
    """

    pass


class ModelMappingDict(Dict[str, ModelConfigDict]):
    """A dictionary that maps model names to  model configuration dicts.
    Usage:
    ```py
    model_mapping = ModelMappingDict({
        "LinearModel": ModelConfigDict(
            {...}
        ),
        "SecondModel": ModelConfigDict(
            {...}
        ),
        )
    ```
    """

    pass


@dataclass
class BaseConfigurator:
    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class DataConfigurator(BaseConfigurator):
    """This dataclass is used to pass arguments to the data loader class.
    In __post_init__ we do some input checks, if the data load configuration is plausible.
    This class is used in the construction of CrossValidation.

    Parameters:
    dataset_name (str): Name of the dataset. Configure this for logging purposes.
    target_name (str): Name of the target variable for logging purposes.
    X (pd.DataFrame): Dataframe containing the features.
    y (pd.Series): Series containing the target variable.
    group (pd.Series): Series containing the group variable. If None, the model level is set to "fixed_only".
    # slopes (bool | str | list[str]): Whether to use random slopes. If True, the random slopes are automatically selected from the dataset variable names table. If False, no random slopes are used. If a string, the column based on the string is used as random slopes. If a list of strings, the random slopes are selected from the dataset based on the list of strings.
    slopes # TODO add slopes and assert that slopes variable is in X
    model_level (str)

    Methods:
    log: This method logs the data configuration to Neptune.
    """

    dataset_name: str | list[str] | None = None
    target_name: str | None = None
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    group: pd.Series | None = None
    slopes: pd.Series | pd.DataFrame | None = None
    model_level: str | None = None

    def _check_values(self):
        """This method checks if the passed values are plausible and raises an error if not.
        The method also sets the model level to "mixed" if a group is passed and to "fixed_only" if no group is passed.
        The following checks are performed:
        - If group is not None, model_level must be "mixed".
        - If group is None, model_level must be "fixed_only".
        - If slopes is not None, group must not be None.
        - Length of X and y must match.
        - Length of X and group must match.
        - Length of X and slopes must match.
        - dataset_name must be a string.
        - target_name must be a string.
        """
        if self.group is not None and self.model_level is None:
            self.model_level = "mixed"
            logger.info("Group was passed. Setting model level to 'mixed'.")

        elif self.group is None and self.model_level is None:
            self.model_level = "fixed_only"
            logger.info("No group was passed. Setting model level to 'fixed_only'.")

        ALLOWED_MODEL_LEVEL_STRINGS = ["fixed_only", "mixed"]
        assert (
            self.model_level in ALLOWED_MODEL_LEVEL_STRINGS
        ), f"Model level {self.model_level} not available. Please choose between 'fixed_only' and 'mixed'."

        if self.slopes is not None and self.group is None:
            raise ValueError(
                "Slopes can only be used with a clustering variable but group was None."
            )

        x_length = len(self.X)
        if self.group is not None:
            assert x_length == len(
                self.group
            ), f"Length of X ({x_length}) and group ({len(self.group)}) do not match."

        if self.slopes is not None:
            assert x_length == len(
                self.slopes
            ), f"Length of X ({x_length}) and slopes ({len(self.slopes)}) do not match."

        assert x_length == len(
            self.y
        ), f"Length of X ({x_length}) and y ({len(self.y)}) do not match."

        assert isinstance(
            self.dataset_name, str
        ), f"Dataset name {self.dataset_name} is not a string."

        assert isinstance(
            self.target_name, str
        ), f"Target name {self.target_name} is not a string."

    def log(self, run):
        """This method logs the data configuration to Neptune."""
        run["data/dataset_name"].log(self.dataset_name)
        run["data/target_name"].log(self.target_name)
        run["data/model_level"].log(self.model_level)
        run["data/slopes"].log(self.slopes)
        run["data/X"].upload(File.as_html(self.X))
        run["data/y"].upload(File.as_html(pd.DataFrame(self.y)))
        if self.group is not None:
            run["data/group"].upload(File.as_html(pd.DataFrame(self.group)))
            run["data/group_name"].log(self.group.name)
        if self.slopes is not None:
            run["data/slopes"].upload(File.as_html(pd.DataFrame(self.slopes)))
            run["data/slopes_name"].log(pd.DataFrame(self.slopes).columns.tolist())


@dataclass
class CrossValConfigurator(BaseConfigurator):
    """This dataclass passes the info related to the general cross validation procedure.
    This class is used in the construction of CrossValidation.

    Parameters:
    cross_val_method (CrossValMethod): Outer cross validation method to be used. Must be one of the following: KFOLD, GROUPKFOLD, STRAT, CUSTOM.
    cross_val_method_in (CrossValMethod): Inner cross validation method to be used. Must be one of the following: KFOLD, GROUPKFOLD, STRAT, CUSTOM.
    n_splits (int): Number of splits to be used in the outer cross validation.
    scale_in (bool): Whether to scale the data in the inner cross validation.
    scale_out (bool): Whether to scale the data in the outer cross validation.
    break_cross_val (bool): Whether to break the cross validation if the model does not converge.
    metrics (MetricsDict): Dictionary of metrics to be used in the cross validation. The keys must be strings and the values must be callables.

    Methods:
    log: This method logs the cross validation configuration to Neptune.
    """

    cross_val_method: CrossValMethod = CrossValMethod.KFOLD
    cross_val_method_in: CrossValMethod = CrossValMethod.KFOLD
    n_splits: int = 5
    scale_in: bool = True
    scale_out: bool = True
    break_cross_val: bool = False
    metrics: MetricsDict = None

    def log(self, run):
        """This method logs the cross validation configuration to Neptune."""
        run["cross_val/cross_val_method"].log(self.cross_val_method.value)
        run["cross_val/cross_val_method_in"].log(self.cross_val_method_in.value)
        run["cross_val/n_splits"].log(self.n_splits)
        run["cross_val/scale_in"].log(self.scale_in)
        run["cross_val/scale_out"].log(self.scale_out)
        run["cross_val/break_cross_val"].log(self.break_cross_val)
        run["cross_val/metrics"].log(pformat(self.metrics))


@dataclass
class MixedEffectsConfigurator(BaseConfigurator):
    """This dataclass passes specific attributes to the mixed effects models.
    It is used in the construction of CrossValidation.

    Parameters:
    em_max_iterations (int): Maximum number of iterations to be used in the EM algorithm.
    em_stopping_threshold (float): Threshold to be used in the EM algorithm.
    em_stopping_window (int): Window to be used in the EM algorithm.
    predict_known_groups_lmm (bool): Whether to predict known groups in the linear mixed effects model.

    Methods:
    log: This method logs the mixed effects configuration to Neptune.
    """

    em_max_iterations: int | str = 50
    em_stopping_threshold: float | None = None
    em_stopping_window: int | None = None
    predict_known_groups_lmm: bool = True

    def log(self, run):
        """This method logs the mixed effects configuration to Neptune."""
        run["mixed_effects/em_max_iterations"].log(self.em_max_iterations)
        run["mixed_effects/em_stopping_threshold"].log(self.em_stopping_threshold)
        run["mixed_effects/em_stopping_window"].log(self.em_stopping_window)
        run["mixed_effects/predict_known_groups_lmm"].log(self.predict_known_groups_lmm)


@dataclass
class RunConfigurator:
    """This dataclass passes the attributes related to the run.
    It is used in the construction of CrossValidation.

    Parameters:
    run (NeptuneRun): The Neptune run object.
    neptune_on (bool): Whether to use Neptune.
    diagnostics (bool): Whether to use diagnostics. This will add histograms of the splits.
    """

    run: NeptuneRun = None
    neptune_on: bool = True
    diagnostics: bool = False

    def log(self, run):
        """This method logs the run configuration to Neptune."""
        run["run/neptune_on"].log(self.neptune_on)
        run["run/diagnostics"].log(self.diagnostics)


@dataclass
class OptimizationConfigurator(BaseConfigurator):
    """This dataclass passes the attributes related to the hyper parameter optimization in the inner cv.
    This class is used in the construction of CrossValidation.

    Parameters:
    optuna (bool): Whether to use optuna.
    n_trials (int | str): Number of trials to be used in the hyper parameter optimization. If n_trials is an integer, it is directly used to control, how many trials are sampled from the hyper parameter distribution. You can set n_trials to "mapped" to use the number of trials mapped to the model in the model mapping instead.
    model_selection (str): Whether to use the best model or the last model in the hyper parameter optimization.
    objective_scorer (Callable | ObjectiveScorer): Objective scorer to be used in the hyper parameter optimization.

    Methods:
    log: This method logs the optimization configuration to Neptune.

    """

    optuna: bool = True
    n_trials: int | str = 100
    objective_scorer: Callable | ObjectiveScorer = None

    def log(self, run):
        """This method logs the optimization configuration to Neptune."""
        run["optimization/optuna"].log(self.optuna)
        run["optimization/n_trials"].log(self.n_trials)


@dataclass
class CrossValidation(BaseConfigurator):
    """This dataclass is constructed using the following configuration classes:
    - DataConfigurator
    - CrossValConfigurator
    - RunConfigurator
    - OptimizationConfigurator (optional)
    - MixedEffectsConfigurator (optional)
    It also takes an optional random seed. If no seed is passed, it defaults to 42.
    It is used to pass all the necessary parameters to the cross validation function.

    Methods:
    """

    data_config: DataConfigurator | None = None
    cross_val_config: CrossValConfigurator | None = None
    run_config: RunConfigurator | None = None
    model_mapping: ModelMappingDict | None = None
    optim_config: OptimizationConfigurator | None = None
    mixed_effects_config: MixedEffectsConfigurator | None = None
    random_seed: int = 42

    def __post_init__(self):
        if self.data_config is None:
            self.data_config = DataConfigurator()
        if self.cross_val_config is None:
            self.cross_val_config = CrossValConfigurator()
        if self.run_config is None:
            self.run_config = RunConfigurator()
        if self.optim_config is None:
            self.optim_config = OptimizationConfigurator()
        if self.mixed_effects_config is None:
            self.mixed_effects_config = MixedEffectsConfigurator()

    def set_data(
        self,
        dataset_name: str,
        target_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        group: pd.Series | None = None,
        slopes: pd.Series | pd.DataFrame | None = None,
        model_level: str | None = None,
    ):
        """
        Set the data configuration.

        Parameters:
        -----------
        dataset_name: str
            Name of the dataset. Configure this for logging purposes.
        target_name: str
            Name of the target variable for logging purposes.
        X: pandas DataFrame
            Dataframe containing the features.
        y: pandas Series
            Series containing the target variable.
        group: pandas Series or None, default=None
            Series containing the group variable. If None, the model level is set to "fixed_only".
        slopes: pandas Series or DataFrame or None, default=None
            The random slopes for the model.
        model_level: str or None, default=None
            The level of the model (e.g. 'mixed', 'fixed', 'random').
        """
        self.data_config = DataConfigurator(
            dataset_name=dataset_name,
            target_name=target_name,
            X=X,
            y=y,
            group=group,
            slopes=slopes,
            model_level=model_level,
        )

    def set_cross_val(
        self,
        cross_val_method: CrossValMethod = CrossValMethod.KFOLD,
        cross_val_method_in: CrossValMethod = CrossValMethod.KFOLD,
        n_splits: int = 5,
        scale_in: bool = True,
        scale_out: bool = True,
        break_cross_val: bool = False,
        metrics: MetricsDict = None,
    ):
        """
        Set the cross validation configuration.

        Parameters:
        -----------
        cross_val_method: CrossValMethod, default=CrossValMethod.KFOLD
            The cross validation method to use for the outer loop.
        cross_val_method_in: CrossValMethod, default=CrossValMethod.KFOLD
            The cross validation method to use for the inner loop.
        n_splits: int, default=5
            The number of splits to use for cross validation.
        scale_in: bool, default=True
            Whether to scale the input data for cross validation.
        scale_out: bool, default=True
            Whether to scale the output data for cross validation.
        break_cross_val: bool, default=False
            Whether to break cross validation if a certain condition is met.
        metrics: dict or None, default=None
            The metrics to use for cross validation.

        Returns:
        --------
        None
        """
        self.cross_val_config = CrossValConfigurator(
            cross_val_method=cross_val_method,
            cross_val_method_in=cross_val_method_in,
            n_splits=n_splits,
            scale_in=scale_in,
            scale_out=scale_out,
            break_cross_val=break_cross_val,
            metrics=metrics,
        )

    def set_run(
        self, run: NeptuneRun = None, neptune_on: bool = True, diagnostics: bool = False
    ):
        """This method sets the run configuration.
        Parameters:
        ------------
        run: Run, default=None
            The Neptune run object or a DummyRun.
        neptune_on: bool, default=True
            Whether to use Neptune for logging.
        diagnostics: bool, default=False
            Whether to enable diagnostic logging."""
        self.run_config = RunConfigurator(
            run=run, neptune_on=neptune_on, diagnostics=diagnostics
        )

    def set_optim(
        self,
        optuna: bool = True,
        n_trials: int | str = 100,
        objective_scorer: Callable | ObjectiveScorer = None,
    ):
        """This method sets the optimization configuration."""
        self.optim_config = OptimizationConfigurator(
            optuna=optuna, n_trials=n_trials, objective_scorer=objective_scorer
        )

    def set_mixed_effects(
        self,
        em_max_iterations: int | str = 50,
        em_stopping_threshold: float | None = None,
        em_stopping_window: int | None = None,
        predict_known_groups_lmm: bool = True,
    ):
        """This method sets the mixed effects configuration."""
        self.mixed_effects_config = MixedEffectsConfigurator(
            em_max_iterations=em_max_iterations,
            em_stopping_threshold=em_stopping_threshold,
            em_stopping_window=em_stopping_window,
            predict_known_groups_lmm=predict_known_groups_lmm,
        )

    def _set_params(self):
        """This method constructs a dict of parameters for the cross validation function."""
        self.params = {
            # "X": self.data_config.X,
            # "y": self.data_config.y,
            # "target_name": self.data_config.target_name,
            # "dataset_name": self.data_config.dataset_name,
            "run": self.run_config.run,
            # "group": self.data_config.group,
            # "slopes": self.data_config.slopes,
            "method": self.cross_val_config.cross_val_method,
            "method_in": self.cross_val_config.cross_val_method_in,
            "break_cross_val": self.cross_val_config.break_cross_val,
            "scale_inner_fold": self.cross_val_config.scale_in,
            "scale_outer_fold": self.cross_val_config.scale_out,
            "n_splits": self.cross_val_config.n_splits,
            "random_seed": self.random_seed,
            "model_level": self.data_config.model_level,
            "n_trials": self.optim_config.n_trials,
            "mapping": self.model_mapping,
            "metrics": self.cross_val_config.metrics,
            "objective_scorer": self.optim_config.objective_scorer,
            "em_max_iterations": self.mixed_effects_config.em_max_iterations,
            "em_stopping_threshold": self.mixed_effects_config.em_stopping_threshold,
            "em_stopping_window": self.mixed_effects_config.em_stopping_window,
            "predict_known_groups_lmm": self.mixed_effects_config.predict_known_groups_lmm,
            "diagnostics": self.run_config.diagnostics,
        }

    def __log(self):
        run = self.run_config.run
        self.data_config.log(run)
        self.cross_val_config.log(run)
        self.run_config.log(run)
        self.optim_config.log(run)
        self.mixed_effects_config.log(run)

    def print_log(self):
        """This method prints the configuration to the console."""
        formula = get_fixed_effects_formula(
            target_name=self.data_config.target_name, X_data=self.data_config.X
        )
        logger.info(f"Performing Cross Validation. Formula: {formula}")
        logger.info(
            f"Cross Validation Methods: \n\tOut{self.cross_val_config.cross_val_method.value}\n\tIn{self.cross_val_config.cross_val_method_in.value}"
        )
        logger.info(f"Scale out: {self.cross_val_config.scale_out}")
        logger.info(f"Scale in: {self.cross_val_config.scale_in}")
        logger.info(f"Models: {self.model_mapping.keys()}")
        if self.data_config.model_level == "mixed":
            logger.info(
                f"Mixed Effects Models: {[self.model_mapping[key]['hello'] for key in self.model_mapping.keys()]}"
            )
            if self.data_config.slopes is not None:
                re_formula = get_re_formula(slopes=self.data_config.slopes)
                logger.info(f"RE Formula: {re_formula}")

    @run_padding
    def perform(self, seed=None):
        """This method performs the cross validation.
        The method returns a dictionary containing the results of the cross-validation, organized by machine learning models.
        usage:
        ```py
        cv = CrossValidation()
        results = cv.perform()
        df = results.summary
        print(df)
        """
        self.data_config._check_values()
        self._set_params()
        if seed is not None:
            self.random_seed = seed
            self.params["random_seed"] = seed
            logger.info("Seed changed to {seed}.")
        self.__log()
        return CrossValidationResults(cross_validate(**self.params))


if __name__ == "__main__":
    cv = CrossValidation()
    cv.data_config(
        dataset_name="dataset_name",
        target_name="target_name",
        X=pd.DataFrame(),
        y=pd.Series(),
        group=pd.Series(),
        slopes=pd.Series(),
        model_level="mixed",
    )
    cv.set_data(
        dataset_name="dataset_name",
        target_name="target_name",
        X=pd.DataFrame(),
        y=pd.Series(),
        group=pd.Series(),
        slopes=pd.Series(),
        model_level="mixed",
    )
    from flexcv.cv_split import CrossValMethod

    results = cv.perform()
    print(cv.data_config)
