from pprint import pformat
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from neptune.types import File
import neptune.integrations.optuna as npt_utils
from optuna.study import Study

from .cv_metrics import MetricsDict, METRICS


class CustomNeptuneCallback(npt_utils.NeptuneCallback):
    """This class inherits from NeptuneCallback and overrides the __call__ method.
    The __call__ method is called after each trial and logs the best trial and the plots.
    The override is necessary because logging each trial is not feasible for multiple models, folds and trials.
    It would hit Neptune's namespace limits."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, study, trial):
        self._log_best_trials(study)
        self._log_plots(study, trial)


def log_diagnostics(
    X_train,
    X_test,
    y_train,
    y_test,
    run,
    effects,
    cluster_train: pd.Series = None,
    cluster_test: pd.Series = None,
    namestring="out",
):
    """This function makes histograms of the features and target for diagnostic purposes.
    All outputs are logged to neptune and the function returns None."""

    def get_df_hist_fig(df):
        fig, axes = plt.subplots(len(df.columns), 1, figsize=(5, 15))
        ax = axes.flatten()

        for i, col in enumerate(df.columns):
            sns.histplot(df[col], ax=ax[i])  # histogram call
            ax[i].set_title(col)
            # remove scientific notation for both axes
            ax[i].ticklabel_format(style="plain", axis="both")
        return fig

    def get_series_hist_fig(ser):
        fig, ax = plt.subplots()
        sns.histplot(ser, ax=ax)
        return fig

    # log number of samples in each fold
    run[f"diagnostics/fold_{namestring}/n_samples_train"].log(len(X_train))
    run[f"diagnostics/fold_{namestring}/n_samples_test"].log(len(X_test))

    fig = get_df_hist_fig(X_train)
    run[f"diagnostics/fold_{namestring}/train_histograms/X"].log(fig)
    del fig
    plt.close()
    fig = get_series_hist_fig(y_train)
    run[f"diagnostics/fold_{namestring}/train_histograms/y"].log(fig)
    del fig
    plt.close()
    fig = get_df_hist_fig(X_test)
    run[f"diagnostics/fold_{namestring}/test_histograms/X"].log(fig)
    del fig
    plt.close()
    fig = get_series_hist_fig(y_test)
    run[f"diagnostics/fold_{namestring}/test_histograms/y"].log(fig)
    del fig
    plt.close()

    # log groups in each fold
    if effects == "mixed":
        run[f"diagnostics/fold_{namestring}/groups_train"].log(
            str(cluster_train.unique().tolist())
        )
        run[f"diagnostics/fold_{namestring}/groups_test"].log(
            str(cluster_test.unique().tolist())
        )


def log_single_model_single_fold(
    y_test,
    y_pred,
    y_train,
    y_pred_train,
    model_name,
    best_model,
    best_params,
    k,
    run,
    results_all_folds,
    study: Study | None,
    metrics: MetricsDict = METRICS,
):
    """
    This function calculates R², MSE and MAE per default based on the y_true and y_pred passed.
    The metrics are appended to the dict containing results from all folds and this dict is also returned.
    The metrics and are logged to Neptune.ai as well.
    The parameters are tracked and the model is uploaded as a pickled File object.
    Finally, the index corresponding to the model with the median R² value is saved to the dict as well.

    Returns:
    dict
    """

    def res_vs_fitted_plot(y_test, y_pred):
        fig = plt.figure()
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted Values")
        return fig

    if metrics is None:
        metrics = METRICS

    inner_cv_exists = True
    if not study is None:
        # log inner train and test MSEs
        df = study.trials_dataframe()
        best_idx = study.best_trial.number
        # since optuna logs the negative scorer we have to change sign to make it a positive MSE
        mse_in_train = -df.iloc[best_idx]["user_attrs_mean_train_score"]
        mse_in_test = -df.iloc[best_idx]["user_attrs_mean_test_score"]
        of = -df.iloc[best_idx]["user_attrs_mean_OF_score"]
    else:
        inner_cv_exists = False
        mse_in_train = np.nan
        mse_in_test = np.nan
        of = np.nan

    # calculate metrics
    eval_metrics = {}
    for metric_name, metric_func in metrics.items():
        eval_metrics[metric_name] = metric_func(y_test, y_pred)
        eval_metrics[metric_name + "_train"] = metric_func(y_train, y_pred_train)

    eval_metrics["mse_in_test"] = mse_in_test
    eval_metrics["mse_in_train"] = mse_in_train

    # update the existing dict, make a new entry if the key does not exist or append to the list if it does
    # this is necessary because of the nested structure
    if model_name not in results_all_folds:
        results_all_folds[model_name] = {
            "model": [],
            "parameters": [],
            "metrics": [],
            "folds_by_metrics": {},
            "y_pred": [],
            "y_test": [],
            "y_pred_train": [],
            "y_train": [],
        }

    # store the fold values by metric
    for metric_name in eval_metrics.keys():
        results_all_folds[model_name]["folds_by_metrics"].setdefault(
            metric_name, []
        ).append(eval_metrics[metric_name])

    # store the metrics by fold
    results_all_folds[model_name]["metrics"].append(eval_metrics)

    # store the model and its parameters
    results_all_folds[model_name]["model"].append(best_model)
    results_all_folds[model_name]["parameters"].append(best_params)
    # store all the predictions and the true values
    results_all_folds[model_name]["y_pred"].append(y_pred)
    results_all_folds[model_name]["y_test"].append(y_test)
    results_all_folds[model_name]["y_pred_train"].append(y_pred_train)
    results_all_folds[model_name]["y_train"].append(y_train)

    # log metrics to neptune
    for key, value in eval_metrics.items():
        run[f"{model_name}/{key}"].log(value)
    if inner_cv_exists:
        run[f"{model_name}/MSE_IN"].log(mse_in_test)
        run[f"{model_name}/MSETRAIN_IN"].log(mse_in_train)
    else:
        run[f"{model_name}/MSE_IN"].log("N/A")
        run[f"{model_name}/MSETRAIN_IN"].log("N/A")
    run[f"{model_name}/OF"].log(of)
    run[f"{model_name}/Model/{k}"].upload(File.as_pickle(best_model))
    run[f"{model_name}/BestParams/{k}"] = pformat(best_params)
    run[f"{model_name}/ResPlot/{k}"].log(res_vs_fitted_plot(y_test, y_pred))
    plt.close()
    return results_all_folds


@dataclass
class SingleModelFoldResult:
    """This dataclass holds results corresponding to a single model fit in a single outer fold.
    This class is used in cross_validate to pass arguments to the model postprocessor and to easily log results.
    """

    k: int
    model_name: str
    best_model: Any
    best_params: dict | Any
    y_pred: pd.Series
    y_test: pd.Series
    X_test: pd.DataFrame
    y_train: pd.Series
    y_pred_train: pd.Series
    X_train: pd.DataFrame
    fit_result: Any
