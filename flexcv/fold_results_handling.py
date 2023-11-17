from dataclasses import dataclass
from pprint import pformat
from typing import Any
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import neptune.integrations.sklearn as npt_utils
import numpy as np
import pandas as pd
from neptune.types import File
from neptune.utils import stringify_unsupported
from optuna.study import Study

from .metrics import METRICS, MetricsDict

logger = logging.getLogger(__name__)

@dataclass
class SingleModelFoldResult:
    """This dataclass is used to store the fold data as well as the predictions of a single model in a single fold.
    It's make_results method is used to evaluate the model with the metrics and log the results to Neptune.

    Attributes:
        k (int): The fold number.
        model_name (str): The name of the model.
        best_model (object): The best model after inner cv or the model when skipping inner cv.
        best_params (dict | Any): The best parameters.
        y_pred (pd.Series): The predictions of the model.
        y_test (pd.Series): The test data.
        X_test (pd.DataFrame): The test data.
        y_train (pd.Series): The train data.
        y_pred_train (pd.Series): The predictions of the model.
        X_train (pd.DataFrame): The train data.
        fit_result (Any): The result of the fit method of the model.
        fit_kwargs (dict): Additional keyword arguments to pass to the fit method. (default: None)

    """

    k: int
    model_name: str
    best_model: object
    best_params: dict | Any
    y_pred: pd.Series
    y_test: pd.Series
    X_test: pd.DataFrame
    y_train: pd.Series
    y_pred_train: pd.Series
    X_train: pd.DataFrame
    fit_result: Any
    fit_kwargs: dict = None

    def make_results(
        self,
        run,
        results_all_folds,
        study: Study | None,
        metrics: MetricsDict = METRICS,
    ):
        """This method is used to evaluate the model with the metrics and log the results to Neptune.

        Args:
          run (neptune.run): Neptune run object.
          results_all_folds (dict): Dictionary containing the results of all models and folds.
          study (optuna.study): Optuna study object.
          metrics (dict): Dictionary containing the metrics to be evaluated.

        Returns:
          (dict): Dictionary containing the results of all models and folds.
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
            df = study.trials_dataframe()
            best_idx = study.best_trial.number
            mse_in_train = -df.iloc[best_idx]["user_attrs_mean_train_score"]
            mse_in_test = -df.iloc[best_idx]["user_attrs_mean_test_score"]
            of = -df.iloc[best_idx]["user_attrs_mean_OF_score"]
        else:
            inner_cv_exists = False
            mse_in_train = np.nan
            mse_in_test = np.nan
            of = np.nan

        eval_metrics = {}
        for metric_name, metric_func in metrics.items():
            eval_metrics[metric_name] = metric_func(self.y_test, self.y_pred)
            eval_metrics[metric_name + "_train"] = metric_func(
                self.y_train, self.y_pred_train
            )

        eval_metrics["mse_in_test"] = mse_in_test
        eval_metrics["mse_in_train"] = mse_in_train

        if self.model_name not in results_all_folds:
            results_all_folds[self.model_name] = {
                "model": [],
                "parameters": [],
                "metrics": [],
                "folds_by_metrics": {},
                "y_pred": [],
                "y_test": [],
                "y_pred_train": [],
                "y_train": [],
            }

        for metric_name in eval_metrics.keys():
            results_all_folds[self.model_name]["folds_by_metrics"].setdefault(
                metric_name, []
            ).append(eval_metrics[metric_name])

        results_all_folds[self.model_name]["metrics"].append(eval_metrics)
        results_all_folds[self.model_name]["model"].append(self.best_model)
        results_all_folds[self.model_name]["parameters"].append(self.best_params)
        results_all_folds[self.model_name]["y_pred"].append(self.y_pred)
        results_all_folds[self.model_name]["y_test"].append(self.y_test)
        results_all_folds[self.model_name]["y_pred_train"].append(self.y_pred_train)
        results_all_folds[self.model_name]["y_train"].append(self.y_train)

        for key, value in eval_metrics.items():
            run[f"{self.model_name}/{key}"].append(value)

        run[f"{self.model_name}/ObjectiveValue"].append(of)
        
        # saving the model
        # check if model has method save_raw -> important for xgboost
        
        try:
            self.best_model.save_model(f"{self.model_name}_{self.k}.json")
            run[f"{self.model_name}/Model/{self.k}"].upload(f"{self.model_name}_{self.k}.json")
            Path(f"{self.model_name}_{self.k}.json").unlink()
        except (AttributeError, KeyError):
            # AttributeError is raised when model has no method save_raw
            # KeyError is raised when model has method save_raw but no raw_format='json'
            run[f"{self.model_name}/Model/{self.k}"].upload(File.as_pickle(self.best_model))

        try:
            run[f"{self.model_name}/Parameters/"] = stringify_unsupported(
                npt_utils.get_estimator_params(self.best_model)
            )
        except (RuntimeError, TypeError):
            # is raised when model is not a scikit-learn model
            run[f"{self.model_name}/Parameters/"].append(pformat(self.best_params))

        run[f"{self.model_name}/ResidualPlot/"].append(
            res_vs_fitted_plot(self.y_test, self.y_pred)
        )

        try:
            run[
                f"{self.model_name}/RegressionSummary/{self.k}"
            ] = npt_utils.create_regressor_summary(
                self.best_model, self.X_train, self.X_test, self.y_train, self.y_test
            )
        except (KeyError, TypeError, RuntimeError, AssertionError) as e:
            # is raised when model is not a scikit-learn model
            logger.info(
                f"Regression summary not available for model {self.model_name}. Skipping.\n{e}")
            run[f"{self.model_name}/RegressionSummary/{self.k}"] = f"Not available: {e}"

        plt.close()
        return results_all_folds
