"""
This module contains the ModelConfigDict and ModelMappingDict classes.
They can be used to construct a mapping which can be passed to flexcv.interface.CrossValidation
Usage:
    ```python
    model_mapping = ModelMappingDict({
        "ModelName1": ModelConfigDict(
            {
                "requires_inner_cv": False,
                "n_trials": 100,
                "n_jobs_model": 1,
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
                "requires_inner_cv": True,
                "n_trials": 100,
                "n_jobs_model": 1,
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


class ModelConfigDict(Dict[str, Type]):
    """A dictionary that maps model configuration names to their corresponding types.

    Default Values:
        To make working with this custom Dict-like class easy, we re-implemented the __init__ method to set some default key-value pairs for us.
        If you don't pass them, it will set

        - requires_inner_cv = False
        - n_jobs_model = 1
        - n_jobs_cv = 1
        - params = {}

    Usage:
        ```python
            {  # TODO update this to correct defaults
                "requires_inner_cv": bool,
                        # this flag can be set to control if a model is used in the inner cross validation.
                        # if set to False, the model will be instantiated in the outer cross validation without hyper parameter optimization.
                "n_trials": int,
                        # number of trials to be used in hyper parameter optimization.
                "n_jobs_model": 1,
                        # number of jobs to be used in the model. We use the sklearn convention here. We use the sklearn convention here.
                        # If your model does not support n_jobs, you can pass False here.
                "n_jobs_cv": 1,
                        # number of jobs to be used in the inner cross validation/hyper parameter tuning. We use the sklearn convention here.
                "model": BaseEstimator,
                        # pass your sklearn model here. It must be a class, not an instance.
                "params": {},
                        # pass the parameters to be used in the model here. It must be a dictionary of optuna distributions or an empty dict.
                "post_processor": flexcv.model_postprocessing.ModelPostProcessor,
                        # pass the post processor class to be used here. It must inherit from the flexcv.model_postprocessing.ModelPostProcessor abstract base class.
            }
        ```
        See also:
            For information on possible optuna distributions, see:
            https://optuna.readthedocs.io/en/stable/reference/distributions.html
    """

    def __init__(self, mapping=None):
        if mapping is None:
            mapping = {}
        super().__init__(mapping)
        self._set_defaults()

    def _set_defaults(self) -> None:
        """Sets default values for the model configuration dict. This allows us to use the dict without having to pass all the keys every time."""
        # check if dict key exists, if not, set default value
        self._check_key_set_default("requires_inner_cv", False)
        self._check_key_set_default("n_jobs_model", -1)
        self._check_key_set_default("n_jobs_cv", -1)
        self._check_key_set_default("params", {})

    def _has_key(self, key) -> bool:
        """Method to check if a key exists in the dict.

        Args:
          key (str | int) : The key to check for.

        Returns:
          (bool): True if the key exists, False otherwise.
        """
        try:
            self[key]
            return True
        except KeyError:
            return False

    def _check_key_set_default(self, key, default) -> None:
        """Checks if a key exists in the dict and sets a default value if it doesn't.

        Args:
          key (str | int): The key to check for.
          default (str | int):  The default value to set if the key doesn't exist.

        Returns:
          (None)
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


if __name__ == "__main__":
    # test default values
    mymodel = ModelConfigDict()
    print(mymodel.__repr__)
    print()
