"""
This module contains functions for logging results to Neptune.ai.
The functions are called during performing cross validation and are used to construct the results metrics dict.
"""

from dataclasses import dataclass
from pprint import pformat
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
import neptune.integrations.optuna as npt_utils
import seaborn as sns
from neptune.types import File
from optuna.study import Study

from .metrics import METRICS, MetricsDict


class CustomNeptuneCallback(npt_utils.NeptuneCallback):
    """This class inherits from NeptuneCallback and overrides the __call__ method.
    The __call__ method is called after each trial and logs the best trial and the plots.
    The override is necessary because logging each trial is not feasible for multiple models, folds and trials.
    It would hit Neptune's namespace limits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, study, trial):
        """Logs only the best trial and the plots.
        Args:
          study (optuna.study): Optuna study object.
          trial (optuna.trial): Optuna trial object.

        Returns:
          (None)
        """
        self._log_best_trials(study)
        self._log_plots(study, trial)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def log_diagnostics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    run,
    effects: str,
    cluster_train: pd.Series = None,
    cluster_test: pd.Series = None,
    namestring: str = "out",
) -> None:
    """Logs histograms of the features and target for diagnostic purposes to neptune.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        run: Neptune run object.
        effects (str): Type of effects to be used. Either "fixed" or "mixed".
        cluster_train (pd.Series, optional): Training clustering or grouping variable. Defaults to None.
        cluster_test (pd.Series, optional): Testing clustering or grouping variable. Defaults to None.
        namestring (str, optional): A string to pass to logging. Use to separate folds: use "in" or "out". Defaults to "out".

    Returns:
        None
    """
    # function definitions

    def get_df_hist_fig(df: pd.DataFrame) -> plt.Figure:
        """Generates a histogram for each column in the dataframe.

        Args:
            df (pd.DataFrame): The data to plot histograms for.

        Returns:
            plt.Figure: The figure object containing the histograms as subplots.
        """
        fig, axes = plt.subplots(len(df.columns), 1, figsize=(5, 15))
        ax = axes.flatten()

        for i, col in enumerate(df.columns):
            sns.histplot(df[col], ax=ax[i])  # histogram call
            ax[i].set_title(col)
            # remove scientific notation for both axes
            ax[i].ticklabel_format(style="plain", axis="both")
        return fig

    def get_series_hist_fig(ser: pd.Series) -> plt.Figure:
        """Get a histogram for a series.

        Args:
            ser (pd.Series): The data to plot a histogram for.

        Returns:
            plt.Figure: The figure object containing the histogram.
        """
        fig, ax = plt.subplots()
        sns.histplot(ser, ax=ax)
        return fig

    # log number of samples in each fold
    run[f"diagnostics/fold_{namestring}/n_samples_train"].append(len(X_train))
    run[f"diagnostics/fold_{namestring}/n_samples_test"].append(len(X_test))

    fig = get_df_hist_fig(X_train)
    run[f"diagnostics/fold_{namestring}/train_histograms/X"].append(fig)
    del fig
    plt.close()
    fig = get_series_hist_fig(y_train)
    run[f"diagnostics/fold_{namestring}/train_histograms/y"].append(fig)
    del fig
    plt.close()
    fig = get_df_hist_fig(X_test)
    run[f"diagnostics/fold_{namestring}/test_histograms/X"].append(fig)
    del fig
    plt.close()
    fig = get_series_hist_fig(y_test)
    run[f"diagnostics/fold_{namestring}/test_histograms/y"].append(fig)
    del fig
    plt.close()

    # log groups in each fold
    if effects == "mixed":
        run[f"diagnostics/fold_{namestring}/groups_train"].append(
            str(cluster_train.unique().tolist())
        )
        run[f"diagnostics/fold_{namestring}/groups_test"].append(
            str(cluster_test.unique().tolist())
        )
