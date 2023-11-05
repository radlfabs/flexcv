import logging

import numpy as np
import pandas as pd
from neptune.types import File
import neptune

from .interface import CrossValidation
from .utilities import add_module_handlers
from .run import Run

logger = logging.getLogger(__name__)


def init_repeated_runs(n_repeats, neptune_credentials) -> tuple:
    """Initialize a repeated run and n_repeats children runs.
    The children runs are linked to the parent run via the "HostRun" key.
    The children runs are initialized with the same credentials as the parent run.

    Args:
      n_repeats(int): Number of repeated runs.
      use_neptune(bool): Whether to use neptune or not.
      neptune_credentials(dict): Credentials for neptune.

    Returns:
      tuple[Run, list[Run]]: parent_run (Run), children_runs (list of n_repeats runs)

    """
    parent_run = neptune.init_run(**neptune_credentials)
    children_runs = [neptune.init_run(**neptune_credentials) for _ in range(n_repeats)]
    return parent_run, children_runs


def try_mean(x):
    """Mean function that handles NaN values for cases where the input contains string "NaN".
    If all elements in x are equal to "NaN", -99 is returned.
    If x contains "NaN" and other values, -999 is returned.

    Args:
      x: Input array.

    Returns:
      float: Mean of the input array.

    """
    try:
        return np.mean(x)
    except ValueError:
        # entered if x contains str "NaN"
        # check if all elements in x are equal to "NaN"
        if np.all([element == "NaN" for element in x]):
            return -99
        else:
            return -999
            
            
def aggregate_(repeated_runs) -> pd.DataFrame:
    """Aggregate the results of repeated runs into a single DataFrame.
    Therefore, the nested dict structure of the results is flattened.
    First, the model results are averaged over folds of individual runs.
    Second, the repeated (indicidual) runs are averaged.
    The summary statistics are returned as a DataFrame with the following structure:
    index: [aggregate]_[metric_name]
    columns: [model_name]
    values: [metric_value]

    Args:
      repeated_runs: List of results of repeated runs.

    Returns:
      pd.DataFrame: Summary statistics of repeated runs.

    """

    model_keys = list(repeated_runs[0].keys())

    result_keys = repeated_runs[0][model_keys[0]]["metrics"].keys()
    results = []
    for result_key in result_keys:
        tmp_df = pd.DataFrame(
            [
                pd.Series(
                    [
                        try_mean(run[model_key]["metrics"][result_key])
                        for model_key in model_keys
                    ],
                    index=model_keys,
                )
                for run in repeated_runs
            ]
        ).agg(["mean", "std"])
        tmp_df.index = [f"{result_key}_{index}" for index in tmp_df.index]
        results.append(tmp_df)
    return pd.concat(results)


class RepeatedResult():
    """Class for results of repeated cross-validation.
    Implements a summary property that returns a DataFrame with aggregated metrics.

    """
    def __init__(self, df):
        """Constructor method for RepeatedResult class.
        
        Args:
            df(pd.DataFrame): DataFrame with aggregated metrics.
        """
        self._summary_df = df
    
    @property
    def summary(self):
        """Summary property that returns a DataFrame with aggregated metrics.

        Args:

        Returns:
          pd.DataFrame: DataFrame with aggregated metrics.

        """
        return self._summary_df


class RepeatedCV(CrossValidation):
    """Class for repeated cross-validation. Inherits from CrossValidation.
    Implements
    - set_n_repeats method that sets the number of repeated runs
    - set_seeds method that sets the random seeds for the repeated runs
    - set_neptune method that sets the neptune credentials for the repeated runs and initializes them
    - in contrast to CrossValidation, the set_run method takes in a parent run and a list of children runs
    - perform method that performs repeated cross-validation


    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_repeats = 3
        self.seeds = None

    def set_n_repeats(self, n_repeats: int):
        """Set the number of repeats for repeated cross-validation.

        Args:
          n_repeats(int): Number of repeated runs.
          n_repeats: int: 

        Returns:
          RepeatedCV: self

        """
        self.n_repeats = n_repeats
        return self
    
    def set_neptune(self, neptune_credentials):
        """Set your neptune credentials and initialize the repeated runs automatically.
        If you do not want to pass your credentials explicitly, use set_run and call init_repeated_runs from outside the class yourself.

        Args:
          neptune_credentials(dict): Your API token and project name.

        Returns:
          RepeatedCV: self

        """
        self.neptune_credentials = neptune_credentials
        parent_run, children_runs = init_repeated_runs(self.n_repeats, neptune_credentials)
        self.parent_run = parent_run
        self.children_runs = children_runs
        return self
    
    def set_seeds(self, seeds=None, generator_seed=42):
        """Set the seeds for the repeated runs.
        If no seeds are passed, a list of seeds is generated randomly.
        The random seed for the random seed generation can be set via the generator_seed argument.
        Therefore, the random seed for the repeated runs is set to the same value for each repeated run and is deterministic.

        Args:
          seeds: (Default value = None)
          generator_seed: (Default value = 42)

        Returns:
          RepeatedCV: self

        """
        np.random.seed(generator_seed)
        self.seeds = np.random.randint(42000, size=self.n_repeats).tolist()
        self.seeds = seeds
        return self
    
    def set_run(self, parent_run=None, children_run=None):
        """Use this method if you want to pass your own parent run and children runs.

        Args:
          parent_run: (Default value = None)
          children_run: (Default value = None)

        Returns:
          RepeatedCV: self

        """
        self.parent_run = parent_run
        self.children_run = children_run
        return self
    
    def _perform_repeats(self):
        """Performs repeated cross-validation.

        Args:

        Returns:
          pd.DataFrame: DataFrame with aggregated metrics.

        """
        add_module_handlers(logger)
        repeated_run = self.parent_run  # neptune.init_run(**your_credentials)
        repeated_id = repeated_run["sys/id"].fetch()
        desc = f"Instance of repeated run {repeated_id}."

        run_ids = []
        run_results = []
        for seed, inner_run in zip(self.seeds, self.children_runs):
            inner_id = inner_run["sys/id"].fetch()
            inner_run["HostRun"] = repeated_id
            inner_run["seed"] = seed

            results = (
                self
                .set_run(run=inner_run, random_seed=seed)
                .perform()
                .get_results()
            )

            # append the run id and the run metric to the lists
            run_ids.append(inner_id)
            run_results.append(results)

        # run_dfs have the same column and index names and we
        df = aggregate_(run_results)

        # log the repeated run results to neptune
        repeated_id = repeated_run["sys/id"].fetch()
        repeated_run["summary"].upload(File.as_html(df))
        repeated_run[
            "sys/description"
        ] = f"Host run for repeated runs with {self.n_repeats} repeats. run_ids: {run_ids}"
        repeated_run["RelatedRuns"] = ", ".join(run_ids)
        repeated_run["seeds"] = self.seeds
        repeated_run["mapping"] = self.config["mapping"]
        repeated_run.stop()
        return df
    
    def perform(self):
        """Wrapper method to perform repeated cross-validation.

        Args:

        Returns:
          RepeatedCV: self

        """
        if self.seeds is None:
            self.set_seeds()
        summary_df = self._perform_repeats()
        self._summary_df = summary_df
        return self

    def get_results(self):
        """Returns the results of repeated cross-validation.

        Args:

        Returns:
          RepeatedResult: RepeatedResult object with summary property that returns a DataFrame with aggregated metrics.

        """
        return RepeatedResult(self._summary_df)


if __name__ == "__main__":
    
  from flexcv.synthesizer import generate_regression
  from flexcv.models import LinearModel
  from flexcv.model_mapping import ModelConfigDict, ModelMappingDict

  # make sample data
  X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)

  # create a model mapping
  model_map = ModelMappingDict(
      {
          "LinearModel": ModelConfigDict(
              {
                  "model": LinearModel,
              }
          ),
      }
  )

  credentials = {}
  
  rcv = (
      RepeatedCV()
      .set_data(X, y, group, dataset_name="ExampleData")
      .set_models(model_map)
      .set_n_repeats(3)
      .set_neptune(credentials)
      .perform()
      .get_results()
  )

  rcv.summary.to_excel("repeated_cv.xlsx")  # save dataframe to excel file