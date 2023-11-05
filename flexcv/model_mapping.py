"""
This module contains the ModelConfigDict and ModelMappingDict classes.
They can be used to construct a mapping which can be passed to flexcv.interface.CrossValidation
Usage:
    ```python
    model_mapping = ModelMappingDict({
        "ModelName1": ModelConfigDict(
            {
                "inner_cv": False,
                "n_trials": 100,
                "n_jobs_model": {"n_jobs": 1},
                "n_jobs_cv": 1,
                "model": model_1,
                "params": {},
                "post_processor": model_1_post_func,
                "mixed_model": mixed_model_1,
                "mixed_post_processor": mixed_model_1_post_func,
                "mixed_name": "MixedModel1"
            }
        ),
        
        "ModelName2": ModelConfigDict(
            {
                "inner_cv": True,
                "n_trials": 100,
                "n_jobs_model": {"n_jobs": 1},
                "n_jobs_cv": 1,
                "model": model_2,
                "params": {},
                "post_processor": model_2_post_func,
                "mixed_model": mixed_model_2,
                "mixed_post_processor": mixed_model_2_post_func,
                "mixed_name": "MixedModel2"
            },
        ),
    }
    )
    ```
"""

from typing import Dict, Type

from .utilities import empty_func


class ModelConfigDict(Dict[str, Type]):
    """A dictionary that maps model configuration names to their corresponding types.
    To make working with this custom Dict-like class easy, we re-implemented the __init__ method to set some default key-value pairs for us.
    If you don't pass them, it will set
    inner_cv = False
    n_trials = 100
    n_jobs = {"n_jobs": 1}
    n_jobs_cv = 1

    Usage:
        ```python
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
            }
        ```
    """

    def __init__(self, mapping=None):
        if mapping is None:
            mapping = {}
        super().__init__(mapping)
        self._set_defaults()

    def _set_defaults(self) -> None:
        """Sets default values for the model configuration dict. This allows us to use the dict without having to pass all the keys every time."""
        # check if dict key exists, if not, set default value
        self._check_key_set_default("inner_cv", False)
        self._check_key_set_default("n_trials", 100)
        self._check_key_set_default("n_jobs", {"n_jobs": 1})
        self._check_key_set_default("n_jobs_cv", 1)
        self._check_key_set_default("params", {})
        self._check_key_set_default("post_processor", empty_func)

        if self._has_key("mixed_model") and not self._has_key("mixed_name"):
            self["mixed_name"] = self["mixed_model"].__repr__()

    def _has_key(self, key) -> bool:
        """Method to check if a key exists in the dict.

        Args: 
          key: The key to check for.

        Returns:
          bool: True if the key exists, False otherwise.
        """
        try:
            self[key]
            return True
        except KeyError:
            return False

    def _check_key_set_default(self, key, default) -> None:
        """Checks if a key exists in the dict and sets a default value if it doesn't.

        Args:
          key: The key to check for.
          default: The default value to set if the key doesn't exist.

        Returns:
          None
        """
        if not self._has_key(key):
            self[key] = default


class ModelMappingDict(Dict[str, ModelConfigDict]):
    """A dictionary that maps model names to  model configuration dicts.
    Usage:
        ```python
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


def map_backwards(mapping) -> dict:
    """Maps a model mapping backwards:
    From the mixed effects model to the fixed effects model.

    Args:
      mapping: The model mapping to map backwards.

    Returns:
      dict: The reversed mapped model mapping.
    """
    # reduce the nested mapping to key: value["mixed_name"]
    reduced_mapping = {key: value["mixed_name"] for key, value in mapping.items()}
    # invert the mapping
    return {value: key for key, value in reduced_mapping.items()}


if __name__ == "__main__":
    # test default values
    mymodel = ModelConfigDict()
    print(mymodel.__repr__)
    print()
