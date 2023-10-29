from typing import Dict, Type

from .funcs import empty_func

"""
- Model mapping is a dictionary of dictionaries, where the keys of the outer dictionary are the names of the models
- it is used to map the model names to the model classes, hyperparameter distributions, post-processing functions, etc.
- it is used in the cross_val.py file to create the models by iterating over the outer dictionary

MODEL_MAPPING = {
    "model_name_1": {
        "inner_cv": bool,  # determines whether to use inner CV for hyperparameter tuning
        "n_jobs_model": dict,  # dict {"n_jobs": int} or empty {} to pass to model constructor
        "n_jobs_cv": int,  # number of jobs to use for inner CV, should be -1 or 1
        "model": model_class,  # model class to use
        "params": dict,  # dict of hyperparameter distributions to use for optuna
        "post_processor": post_processor_function,  # function to use for post-processing
        "level_4_model": level_4_model_class,  # model class to use for level 4 model
        "level_4_post_processor": level_4_post_processor_function  # function to use for level 4 post-processing
    },
    "model_name_2": {
        "inner_cv": bool,
        "n_jobs_model": dict,
        "n_jobs_cv": int,
        "model": model_class,
        "params": dict,
        "post_processor": post_processor_function,
        "level_4_model": level_4_model_class,
        "level_4_post_processor": level_4_post_processor_function
    },
    # ...
}
"""


class ModelConfigDict(Dict[str, Type]):
    """A dictionary that maps model configuration names to their corresponding types.
    To make working with this custom Dict-like class easy, we re-implemented the __init__ method to set some default key-value pairs for us.
    If you don't pass them, it will set
    inner_cv = False
    n_trials = 100
    n_jobs = {"n_jobs": 1}
    n_jobs_cv = 1
    params = {}

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

    def __init__(self, mapping=None):
        if mapping is None:
            mapping = {}
        super().__init__(mapping)
        self._set_defaults()

    def _set_defaults(self):
        if not hasattr(self, "inner_cv"):
            self["inner_cv"] = False
        if not hasattr(self, "n_trials"):
            self["n_trials"] = 100
        if not hasattr(self, "n_jobs"):
            self["n_jobs"] = {"n_jobs": 1}
        if not hasattr(self, "n_jobs_cv"):
            self["n_jobs_cv"] = 1
        if not hasattr(self, "params"):
            self["params"] = {}
        if not hasattr(self, "post_processor"):
            self["post_processor"] = empty_func
        if hasattr(self, "mixed_model") and not hasattr(self, "mixed_name"):
            self["mixed_name"] = self["mixed_model"].__repr__()


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


from sklearn.base import BaseEstimator


def make_model_config_from_estimator(
    fixed_effects_model: BaseEstimator,
    mixed_effects_model: BaseEstimator = None,
    fixed_post_processing_fn: callable = None,
    mixed_post_processing_fn: callable = None,
) -> ModelConfigDict:
    """Creates a model configuration dictionary from a sklearn model.
    This function is used to create a model configuration dictionary from a sklearn model.
    It is used in the `CrossValidation` class to create the model configuration dictionary from the model mapping dictionary.
    Args:
        model: A sklearn model.
    Returns:
        A model configuration dictionary.
    """

    if not mixed_effects_model:
        mixed_effects_model = None

    return ModelConfigDict(
        {
            "inner_cv": False,
            "n_trials": 100,
            "n_jobs_model": {"n_jobs": 1},
            "n_jobs_cv": 1,
            "model": fixed_effects_model.__class__,
            "params": {},
            "post_processor": fixed_post_processing_fn
            if fixed_post_processing_fn
            else empty_func,
            "level_4_model": mixed_effects_model
            if mixed_effects_model
            else BaseEstimator,
            "level_4_post_processor": mixed_post_processing_fn
            if mixed_post_processing_fn
            else empty_func,
            "level_4_name": mixed_effects_model.__class__.__name__
            if mixed_effects_model
            else "MixedModel",
        }
    )


MIXED_TO_BASE_MODEL_MAPPING: Dict[str, str] = {
    "MixedLM": "LinearModel",
    "MERF": "RandomForest",
    "XGBEM": "XGBoost",
    "EarthEM": "MARS",
    "SVREM": "SVR",
}


if __name__ == "__main__":
    # test default values
    mymodel = ModelConfigDict()
    print(mymodel.__repr__)
    print()
