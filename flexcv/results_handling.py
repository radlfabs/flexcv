"""
This module provides a CrossValidationResults class which can be used to summarize the results of CrossValidation.perform().
"""


import operator

import numpy as np
import pandas as pd


def add_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add summary statistics to a pandas DataFrame.
    Calculates the mean, median and standard deviation on copies slices of the original data and adds them as rows to the DataFrame.
    This makes sure, that the summary statistics are not sequentially dependent on each other.

    Args:
      df: pd.DataFrame: Input data.

    Returns:
      (pd.DataFrame): Data with added summary statistics

    """
    original_fold_slice = df.copy(deep=True)
    df.loc["mean"] = original_fold_slice.mean(skipna=True)
    df.loc["median"] = original_fold_slice.median(skipna=True)
    df.loc["std"] = original_fold_slice.std(skipna=True)
    return df


class CrossValidationResults(dict):
    """A summary of the results of CrossValidation.perform().
    Cross validate returns a dictionary of results for each model with the form:
    ```python
    {
        "model_name_1": {
            "model": [model_1_fold_1, model_1_fold_2, ...],
            "parameters": [params_1_fold_1, params_1_fold_2, ...],
            "metrics": [
                {
                    "metric_1_fold_1": metric_value_1_fold_1,
                    "metric_2_fold_1": metric_value_2_fold_1,
                        ...
                },
                {
                    "metric_1_fold_2": metric_value_1_fold_2,
                    "metric_2_fold_2": metric_value_2_fold_2,
                    ...
                },
            ],
            "y_pred": [y_pred_1_fold_1, y_pred_1_fold_2, ...],
            "y_test": [y_test_1_fold_1, y_test_1_fold_2, ...],
            "y_pred_train": [y_pred_train_1_fold_1, y_pred_train_1_fold_2, ...],
            "y_train": [y_train_1_fold_1, y_train_1_fold_2, ...],
        },
        "model_name_2": {
            ...
        },
        ...
    }
    ```
    This class is a wrapper around this dictionary which provides a summary of the results.

    - `_make_summary` computes the mean, median and standard deviation of the metrics for each model.
    - `_make_summary` is called the first time the summary property is accessed and the result is cached.
    - `_get_model` returns the model instance corresponding to the given model name.

    Properties:
        summary (pd.DataFrame): Summary of the results.

    """

    def __init__(self, results_dict):
        super().__init__(results_dict)
        self._summary = None
        # TODO add some kind of id or description

    def __repr__(self):
        return f"CrossValidationResults {pformat_dict(self)}"

    @property
    def summary(self):
        """Property that returns a pandas dataframe with the fold values, mean, median and standard deviation of the metrics for each model."""
        if self._summary is None:
            self._summary = self._make_summary()
        return self._summary

    def _make_summary(self):
        """Creates pandas dataframe with the fold values, mean, median and standard deviation of the metrics for each model.
        Columns: model names
        Multiindex from tuples: (fold id, metric)
        1. It reorders the data from
        ```python
        "metrics": [
                {
                    "metric_1_fold_1": metric_value_1_fold_1,
                    "metric_2_fold_1": metric_value_2_fold_1,
                        ...
                },
                {
                    "metric_1_fold_2": metric_value_1_fold_2,
                    "metric_2_fold_2": metric_value_2_fold_2,
                    ...
                },
            ],
        ```
        to the form
        ```python
        "metrics": {
            "metric_1": [metric_value_1_fold_1, metric_value_1_fold_2, ...],
            "metric_2": [metric_value_2_fold_1, metric_value_2_fold_2, ...],
            ...
        }
        ```

        """
        model_dfs = []
        for model in self.keys():
            metrics_dfs = []
            for metric in self[model]["folds_by_metrics"].keys():
                # collect the values for each fold for single metric
                values_df = pd.DataFrame(
                    self[model]["folds_by_metrics"][metric], columns=[metric]
                )
                values_df.index.name = "Fold"
                values_df = add_summary_stats(values_df)
                metrics_dfs.append(values_df)
            # concatenate the dataframes for each metric
            model_df = pd.concat(metrics_dfs, axis=1)
            # reshape to long format and add model name as column
            new_df = model_df.reset_index().melt(
                id_vars="Fold", var_name="Metric", value_name=model
            )
            # set the index to fold and metric
            new_df = new_df.set_index(["Fold", "Metric"])
            model_dfs.append(new_df)
        # concatenate the dataframes for each model
        summary_df = pd.concat(model_dfs, axis=1)
        return summary_df

    def get_best_model_by_metric(
        self, model_name=None, metric_name="mse", direction="min"
    ):
        """Returns the model with the best metric value for the given metric.
        Direction can be "min" or "max" and determines whether the best model is the one with the lowest or highest metric value.
        E.g. for MSE, direction should be "min" and for R2, direction should be "max".

        Args:
          model_name (str): Name of the model (Default value = None)
          metric_name (str): Name for the metric (Default value = "mse")
          direction (str): Minimize or maximize. (Default value = "min")

        Returns:
            (object): The model with the best metric value for the given metric.
        """
        assert direction in [
            "min",
            "max",
        ], f"direction must be 'min' or 'max', got {direction}"
        arg_func = np.argmin if direction == "min" else np.argmax
        op_func = operator.lt if direction == "min" else operator.gt

        best_model_name = None
        best_metric_value = None
        best_metric_index = None

        iter_dict = (
            self.items() if model_name is None else [(model_name, self[model_name])]
        )

        for model_name, model_results in iter_dict:
            metric_values = model_results["metrics"][metric_name]
            metric_index = arg_func(metric_values)
            metric_value = metric_values[metric_index]

            if best_metric_value is None or op_func(metric_value, best_metric_value):
                best_metric_value = metric_value
                best_metric_index = metric_index
                best_model_name = model_name

        return self[best_model_name]["model"][best_metric_index]

    def get_predictions(self, model_name, fold_id):
        """Returns the predictions for the given model and fold.

        Args:
          model_name (str): The name of the model.
          fold_id (int): The id of the fold.

        Returns:
          (array-like): The predictions for the given model and fold.

        """
        return self[model_name]["y_pred"][fold_id]

    def get_true_values(self, model_name, fold_id):
        """Returns the true values for the given model and fold.

        Args:
          model_name (str): The name of the model.
          fold_id (int): The id of the fold.

        Returns:
            (array-like): The true values for the given model and fold.
        """
        return self[model_name]["y_test"][fold_id]

    def get_training_predictions(self, model_name, fold_id):
        """Returns the predictions for the given model and fold.

        Args:
          model_name (str): The name of the model.
          fold_id (int): The id of the fold.

        Returns:
            (array-like): The training predictions for the given model and fold.
        """
        return self[model_name]["y_pred_train"][fold_id]

    def get_training_true_values(self, model_name, fold_id):
        """Returns the true values for the given model and fold.

        Args:
          model_name (str): The name of the model.
          fold_id (int): The id of the fold.

        Returns:
            (array-like): The training true values for the given model and fold.
        """
        return self[model_name]["y_train"][fold_id]

    def get_params(self, model_name, fold_id):
        """Returns the parameters for the given model and fold.
        If the key is not found, returns None and will not raise an error.

        Args:
          model_name (str): The name of the model.
          fold_id (int): The id of the fold.

        Returns:
            (dict): The parameters for the given model and fold.
        """
        return self[model_name].get("parameters", [None])[fold_id]


class MergedSummary(CrossValidationResults):
    """ """

    def __init__(
        self,
        cv_results_1,
        cv_results_2,
    ):
        self.cv_results_1 = cv_results_1
        self.cv_results_2 = cv_results_2

    @property
    def summary(self):
        """ """
        return self._merge()

    def _merge(self):
        """Merges the two summaries dataframes into one."""
        return pd.concat([self.cv_results_1.summary, self.cv_results_2.summary], axis=1)

    def __add__(self, other):
        raise NotImplementedError


if __name__ == "__main__":
    pass
